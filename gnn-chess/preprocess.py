import argparse
import os
import sys

import chess
import chess.pgn
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ours_gpu_extras'))
from labels import build_rows, record_game, save_shard

RESULTS = {'1-0': 0, '1/2-1/2': 1, '0-1': 2}
GAMES_PER_SHARD = 2000
OPENING_DROP_MAX = 0.98
OPENING_DROP_SCALE = 0.075


def keep_prob(ply: int, sample_rate: float) -> float:
    opening_drop = max(0.0, OPENING_DROP_MAX - (OPENING_DROP_SCALE * ply) ** 2)
    return sample_rate * (1.0 - opening_drop)


def sample_plies(n_moves: int, sample_rate: float, rng) -> set:
    return {ply for ply in range(n_moves) if rng.random() < keep_prob(ply, sample_rate)}


def process_game(game: chess.pgn.Game, sample_rate: float, rng):
    result = RESULTS.get(game.headers.get('Result', ''))
    if result is None:
        return None

    moves = list(game.mainline_moves())
    n_moves = len(moves)
    if n_moves < 2:
        return None

    board = game.board()
    sampled_plies = sample_plies(n_moves, sample_rate, rng)
    move_infos, states = record_game(board, moves, sampled_plies)
    return build_rows(move_infos, states, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pgn')
    parser.add_argument('--out', default='data')
    parser.add_argument('--sample-rate', type=float, default=0.1, help='per-position keep probability')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    games_in_shard = 0
    shard_id = 0
    n_positions = 0

    with open(args.pgn, 'r', encoding='utf-8', errors='replace') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_rows = process_game(game, args.sample_rate, rng)
            if game_rows:
                rows.extend(game_rows)
            games_in_shard += 1

            if games_in_shard == GAMES_PER_SHARD:
                if save_shard(rows, args.out, shard_id):
                    shard_id += 1
                    n_positions += len(rows)
                rows, games_in_shard = [], 0

    if save_shard(rows, args.out, shard_id):
        shard_id += 1
        n_positions += len(rows)
    print(f'done: {n_positions:,} positions in {shard_id} shards -> {args.out}')


if __name__ == '__main__':
    main()
