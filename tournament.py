import argparse
import itertools
import os
import random
import sys

import chess
import chess.engine
import chess.pgn
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, 'gnn-chess'))
sys.path.append(os.path.join(HERE, 'ours_gpu_extras'))
from graph_builder import GraphBuilder
from plays import load_model, model_move
from search import search_move

from bayes_elo import fit_bayeselo


class Player:
    def __init__(self, name, pick_move, known_elo=None):
        self.name = name
        self.pick_move = pick_move
        self.known_elo = known_elo


def unique_name(name, taken):
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f'{name}#{n}', n + 1
    return candidate


def make_model_player(spec, device, cache, taken):
    """spec is 'path' (policy only) or 'path@depth' (policy + minimax search)."""
    path, _, depth = spec.partition('@')
    if path not in cache:
        cache[path] = (load_model(path, device), GraphBuilder(device))
    model, graph_builder = cache[path]
    stem = os.path.splitext(os.path.basename(path))[0]
    if depth:
        name = unique_name(f'{stem}+search{depth}', taken)
        return Player(name, lambda board: search_move(model, graph_builder, board, max_depth=int(depth)))
    return Player(unique_name(stem, taken), lambda board: model_move(model, graph_builder, board))


def make_stockfish_player(path, elo, movetime, engines):
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engines.append(engine)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': elo})
    limit = chess.engine.Limit(time=movetime)
    return Player(f'Stockfish{elo}', lambda board: engine.play(board, limit).move, known_elo=elo)


def random_opening(n_plies, rng):
    while True:
        board = chess.Board()
        for _ in range(n_plies):
            board.push(rng.choice(list(board.legal_moves)))
            if board.is_game_over():
                break
        if not board.is_game_over():
            return list(board.move_stack)


def play_game(white, black, max_plies, opening):
    board = chess.Board()
    for move in opening:
        board.push(move)
    result = None
    while result is None and not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= max_plies:
            result = '1/2-1/2'
            break
        mover, other = (white, black) if board.turn == chess.WHITE else (black, white)
        move = mover.pick_move(board)
        if move not in board.legal_moves:
            result = '0-1' if mover is white else '1-0'  # illegal move forfeits
            break
        board.push(move)
    if result is None:
        result = board.result(claim_draw=True)

    game = chess.pgn.Game.from_board(board)
    game.headers['White'], game.headers['Black'], game.headers['Result'] = white.name, black.name, result
    return result, game


def main():
    parser = argparse.ArgumentParser(description='Round-robin tournament rated with BayesElo, anchored to Stockfish.')
    parser.add_argument('--stockfish', required=True, help='path to stockfish executable')
    parser.add_argument('--stockfish-elo', type=int, nargs='+', default=[1320, 1500],
                        help='known Elo level(s) for Stockfish; each level is a separate anchored player')
    parser.add_argument('--model', action='append', default=None,
                        help="model player, 'path' or 'path@depth' for search; repeat for several players")
    parser.add_argument('--games', type=int, default=10, help='games per pairing per round (colors alternate)')
    parser.add_argument('--rounds', type=int, default=1, help='number of round-robin rounds')
    parser.add_argument('--movetime', type=float, default=0.1, help='seconds per Stockfish move')
    parser.add_argument('--max-plies', type=int, default=300, help='games longer than this are drawn')
    parser.add_argument('--opening-plies', type=int, default=4,
                        help='random moves played before the players take over (0 = always the start position)')
    parser.add_argument('--seed', type=int, default=None, help='seed for the random openings (default: random)')
    parser.add_argument('--prior-std', type=float, default=300.0, help='BayesElo prior std (Elo) for fitted players')
    parser.add_argument('--pgn', default='tournament.pgn', help='where to save every game')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2 ** 32)
    print(f'opening seed: {seed}', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    taken, cache, engines = set(), {}, []
    players = []
    try:
        for spec in args.model or [os.path.join('gnn-chess', 'chessgnn.pt')]:
            players.append(make_model_player(spec, device, cache, taken))
            taken.add(players[-1].name)
        for elo in args.stockfish_elo:
            players.append(make_stockfish_player(args.stockfish, elo, args.movetime, engines))

        # Stockfish-vs-Stockfish games say nothing new: their Elos are already known.
        pairings = [(a, b) for a, b in itertools.combinations(range(len(players)), 2)
                    if not (players[a].known_elo and players[b].known_elo)]
        pair_tally = {pair: [0, 0, 0] for pair in pairings}  # [wins_a, wins_b, draws]
        games_played = {pair: 0 for pair in pairings}

        with open(args.pgn, 'w') as pgn_file:
            for round_number in range(1, args.rounds + 1):
                for a, b in pairings:
                    for _ in range(args.games):
                        # games_played persists across rounds so colors stay balanced
                        white, black = (a, b) if games_played[(a, b)] % 2 == 0 else (b, a)
                        # a game and its color-swapped twin share one random opening
                        opening_rng = random.Random(f'{seed}-{a}-{b}-{games_played[(a, b)] // 2}')
                        opening = random_opening(args.opening_plies, opening_rng)
                        games_played[(a, b)] += 1
                        result, game = play_game(players[white], players[black], args.max_plies, opening)
                        print(game, file=pgn_file, end='\n\n')
                        pgn_file.flush()

                        if result == '1/2-1/2':
                            pair_tally[(a, b)][2] += 1
                        elif (result == '1-0') == (white == a):
                            pair_tally[(a, b)][0] += 1
                        else:
                            pair_tally[(a, b)][1] += 1
                    w, l, d = pair_tally[(a, b)]
                    print(f'round {round_number}: {players[a].name} vs {players[b].name}  '
                          f'cumulative +{w} -{l} ={d}', flush=True)
    finally:
        for engine in engines:
            engine.quit()

    games = [(a, b, w, l, d) for (a, b), (w, l, d) in pair_tally.items()]
    anchors = {i: p.known_elo for i, p in enumerate(players) if p.known_elo}
    prior_rating = sum(anchors.values()) / len(anchors)
    ratings, kappa = fit_bayeselo(games, len(players), anchors=anchors,
                                  prior_rating=prior_rating, prior_std=args.prior_std)

    scores = [0.0] * len(players)
    played = [0] * len(players)
    for a, b, w, l, d in games:
        scores[a] += w + d / 2
        scores[b] += l + d / 2
        played[a] += w + l + d
        played[b] += w + l + d

    print(f'\n{len(games)} pairings, {args.rounds} round(s), {args.games} games each. Games saved to {args.pgn}')
    print(f'{"Player":<28}{"Elo":>8}{"Games":>8}{"Score":>8}')
    for i in sorted(range(len(players)), key=lambda i: -ratings[i]):
        tag = '  (fixed)' if i in anchors else ''
        score = f'{scores[i] / played[i]:.2f}' if played[i] else '-'
        print(f'{players[i].name:<28}{ratings[i]:>8.0f}{played[i]:>8}{score:>8}{tag}')
    print(f'draw parameter kappa = {kappa:.2f}')


if __name__ == '__main__':
    main()
