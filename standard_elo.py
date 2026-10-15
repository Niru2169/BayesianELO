import argparse
import re

import chess.pgn


def expected_score(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def main():
    parser = argparse.ArgumentParser(description='Classic sequential Elo over a PGN, Stockfish<N> players fixed at N.')
    parser.add_argument('--pgn', default='tournament.pgn')
    parser.add_argument('--k', type=float, default=16, help='K-factor')
    args = parser.parse_args()

    results = []
    with open(args.pgn) as f:
        while (game := chess.pgn.read_game(f)) is not None:
            white, black, result = game.headers['White'], game.headers['Black'], game.headers['Result']
            results.append((white, black, {'1-0': 1.0, '0-1': 0.0}.get(result, 0.5)))

    fixed = {}
    for white, black, _ in results:
        for name in (white, black):
            match = re.fullmatch(r'Stockfish(\d+)', name)
            if match:
                fixed[name] = float(match.group(1))
    start = sum(fixed.values()) / len(fixed) if fixed else 1500.0

    ratings = {name: fixed.get(name, start) for white, black, _ in results for name in (white, black)}
    for white, black, white_score in results:
        change = args.k * (white_score - expected_score(ratings[white], ratings[black]))
        if white not in fixed:
            ratings[white] += change
        if black not in fixed:
            ratings[black] -= change

    print(f'{"Player":<28}{"Elo":>8}')
    for name, rating in sorted(ratings.items(), key=lambda item: -item[1]):
        print(f'{name:<28}{rating:>8.0f}{"  (fixed)" if name in fixed else ""}')


if __name__ == '__main__':
    main()
