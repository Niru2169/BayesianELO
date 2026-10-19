import argparse
import math
import os
import sys

import chess
import chess.engine
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ours_gpu_extras'))
from graph_builder import GraphBuilder
from plays import load_model, model_move
from search import search_move


def play_game(model, graph_builder, engine, limit, model_is_white: bool, use_search: bool, depth: int) -> float:
    board = chess.Board()
    model_color = chess.WHITE if model_is_white else chess.BLACK

    while not board.is_game_over(claim_draw=True):
        if board.turn == model_color:
            if use_search:
                move = search_move(model, graph_builder, board, max_depth=depth)
            else:
                move = model_move(model, graph_builder, board)
            if move not in board.legal_moves:
                return 0.0
        else:
            move = engine.play(board, limit).move
        board.push(move)

    result = board.result(claim_draw=True)
    if result == '1/2-1/2':
        return 0.5
    white_won = result == '1-0'
    model_won = white_won == model_is_white
    return 1.0 if model_won else 0.0


def elo_diff(score: float, n_games: int) -> float:
    score = min(max(score, 0.5 / n_games), 1 - 0.5 / n_games)
    return -400 * math.log10(1 / score - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stockfish', required=True, help='path to stockfish executable')
    parser.add_argument('--model', default='chessgnn.pt')
    parser.add_argument('--elo', type=int, default=1320, help='Stockfish Elo level')
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--movetime', type=float, default=0.1, help='seconds per Stockfish move')
    parser.add_argument('--search', action='store_true', help='use shallow minimax on top of the policy')
    parser.add_argument('--depth', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, graph_builder = load_model(args.model, device), GraphBuilder(device)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': args.elo})
    limit = chess.engine.Limit(time=args.movetime)

    try:
        total_score, wins, draws, losses = 0.0, 0, 0, 0
        for game_number in range(args.games):
            model_is_white = game_number % 2 == 0
            score = play_game(model, graph_builder, engine, limit, model_is_white, args.search, args.depth)
            total_score += score
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1
            print(f'game {game_number + 1}/{args.games}: model scored {score}', flush=True)

        average_score = total_score / args.games
        print(f'Result: +{wins} ={draws} -{losses}, score {average_score:.2f}')
        estimate = args.elo + elo_diff(average_score, args.games)
        print(f'Estimated model Elo: {estimate:.0f} (rough, Stockfish scale)')
    finally:
        engine.quit()


if __name__ == '__main__':
    main()
