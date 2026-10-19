"""
PGN -> compact training shards (.npz), parallel across CPU cores. Run once.

    python preprocess.py ../ours/games.pgn --out data --sample-rate 0.1 --workers 20

Every position (board BEFORE a move) gets all labels derived from the game record, following the
reference's data preparation:
  next move, legal moves, legal captures, pieces that move / are captured in the next 10 plies,
  give-check / get-checked within 1/3/5 own plies, plies till end, final result, state repetition.
Positions are subsampled (--sample-rate) and early-opening positions are down-weighted with the
reference's filter  p(drop) = max(0, 0.98 - (0.075*ply)^2).

Games are split into chunks and processed by a process pool (--workers). Existing shards are
skipped, so re-running resumes where it left off. Each chunk gets its own RNG (seed + chunk id),
so results do not depend on the worker count.
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import chess
import chess.pgn
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ours_gpu_extras'))
from labels import build_rows, record_game, rows_to_arrays

RESULTS = {'1-0': 0, '1/2-1/2': 1, '0-1': 2}
GAMES_PER_SHARD = 2000
OPENING_DROP_MAX = 0.98
OPENING_DROP_SCALE = 0.075


def keep_prob(ply: int, sample_rate: float) -> float:
    opening_drop = max(0.0, OPENING_DROP_MAX - (OPENING_DROP_SCALE * ply) ** 2)
    return sample_rate * (1.0 - opening_drop)


def sample_plies(n_moves: int, sample_rate: float, rng) -> set:
    return {ply for ply in range(n_moves) if rng.random() < keep_prob(ply, sample_rate)}


def process_game(game: chess.pgn.Game, sample_rate: float, rng, min_elo: int):
    result = RESULTS.get(game.headers.get('Result', ''))
    if result is None or game.headers.get('Variant', 'Standard') not in ('Standard', ''):
        return None
    if min_elo:
        try:
            if min(int(game.headers['WhiteElo']), int(game.headers['BlackElo'])) < min_elo:
                return None
        except (KeyError, ValueError):
            return None
    moves = list(game.mainline_moves())
    n_moves = len(moves)
    if n_moves < 2:
        return None

    board = game.board()
    if board.chess960:
        return None
    sampled_plies = sample_plies(n_moves, sample_rate, rng)
    move_infos, states = record_game(board, moves, sampled_plies)
    return build_rows(move_infos, states, result)


def work(args):
    chunk_id, text, out_dir, rate, seed, min_elo = args
    rng = np.random.default_rng(seed + chunk_id)
    stream = io.StringIO(text)
    rows, games = [], 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        r = process_game(game, rate, rng, min_elo)
        if r:
            rows.extend(r)
            games += 1
    arrays = rows_to_arrays(rows)
    if arrays is not None:
        tmp = os.path.join(out_dir, f'.shard_{chunk_id:06d}.tmp.npz')
        np.savez(tmp, **arrays)
        os.replace(tmp, os.path.join(out_dir, f'shard_{chunk_id:06d}.npz'))
    return chunk_id, games, len(rows)


def iter_chunks(path, games_per_chunk, max_games):
    buf, n_in, total = [], 0, 0
    chunk_id = 0
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        for line in f:
            if line.startswith('[Event '):
                if n_in == games_per_chunk:
                    yield chunk_id, ''.join(buf)
                    chunk_id += 1
                    buf, n_in = [], 0
                if max_games is not None and total >= max_games:
                    break
                n_in += 1
                total += 1
            buf.append(line)
    if buf:
        yield chunk_id, ''.join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pgn')
    ap.add_argument('--out', default='data')
    ap.add_argument('--sample-rate', type=float, default=0.1, help='per-position keep probability')
    ap.add_argument('--max-games', type=int, default=None)
    ap.add_argument('--games-per-chunk', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument('--min-elo', type=int, default=0, help='drop games where either player is below this')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0, done_pos, done_chunks = time.time(), 0, 0
    pending = set()

    def reap(block):
        nonlocal pending, done_pos, done_chunks
        if not pending:
            return
        d, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=None if block else 0)
        for fut in d:
            cid, games, npos = fut.result()
            done_pos += npos
            done_chunks += 1
            if done_chunks % 10 == 0:
                print(f'[{time.time() - t0:7.0f}s] chunks={done_chunks} positions={done_pos:,}', flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        for cid, text in iter_chunks(a.pgn, a.games_per_chunk, a.max_games):
            if os.path.exists(os.path.join(a.out, f'shard_{cid:06d}.npz')):
                continue  # resume support
            while len(pending) >= a.workers * 3:
                reap(True)
            pending.add(ex.submit(work, (cid, text, a.out, a.sample_rate, a.seed, a.min_elo)))
            reap(False)
        while pending:
            reap(True)
    print(f'done: {done_pos:,} new positions in {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
