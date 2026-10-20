import glob
import os

import numpy as np
import torch

from chessutil import decode_move

SHARD_KEYS = ['board', 'ep', 'castle', 'turn', 'half', 'rep', 'next_from', 'next_to',
              'moved', 'captured', 'chk', 'ptte', 'result', 'legal_len', 'legal_data']


def _state_hashes(board, turn, castle, ep):
    """Compact 64-bit hash of (board, turn, castle, ep) for memory-efficient dedup.

    Hash column-by-column (each step is a transient 8-byte-per-row array) rather
    than casting the whole (N, 64) board to uint64, which would need 8x the board
    in RAM. A structured ``np.unique(..., axis=0)`` over tens of millions of rows
    needs several GB; this needs ~200 MB.
    """
    n = board.shape[0]
    h = np.zeros(n, dtype=np.uint64)
    prime = np.uint64(0x100000001B3)  # FNV-1a 64-bit prime
    for i in range(64):
        h = h * prime
        h ^= board[:, i].astype(np.uint64)
    h ^= turn.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    h ^= castle.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= (ep.astype(np.uint64) + 1) * np.uint64(0x165667B19E3779F9)
    return h


def load_shards(files: list, strict_dedup: bool = True) -> dict:
    parts = {key: [] for key in SHARD_KEYS}
    for file in files:
        with np.load(file) as shard:
            for key in SHARD_KEYS:
                parts[key].append(shard[key])
    raw = {key: np.concatenate(arrays) for key, arrays in parts.items()}

    if not strict_dedup:
        return raw

    # Strict deduplication across the composite state signature
    # (board [64], turn [1], castle [1], ep [1]).
    hashes = _state_hashes(raw['board'], raw['turn'], raw['castle'], raw['ep'])
    _, unique_indices = np.unique(hashes, return_index=True)
    unique_indices = np.sort(unique_indices)

    # Rebuild the variable-length legal_data slices for the kept positions,
    # chunked so the flat index arrays stay small (a single 633M-element int64
    # index would need ~5 GB).
    lengths = raw['legal_len'].astype(np.int64)
    starts = np.cumsum(lengths) - lengths
    kept_starts = starts[unique_indices]
    kept_lens = lengths[unique_indices]
    parts = []
    CHUNK = 500_000
    for s in range(0, len(unique_indices), CHUNK):
        ks = kept_starts[s:s + CHUNK]
        kl = kept_lens[s:s + CHUNK]
        tot = int(kl.sum())
        if tot == 0:
            continue
        group_starts = np.cumsum(kl) - kl
        within = np.arange(tot, dtype=np.int64) - np.repeat(group_starts, kl)
        flat = np.repeat(ks, kl) + within
        parts.append(raw['legal_data'][flat])
    legal_data = np.concatenate(parts) if parts else np.empty(0, dtype=raw['legal_data'].dtype)

    # Build the deduped dict while popping from `raw`, so the old arrays are
    # freed as we go instead of both sets being resident at once.
    deduped = {}
    for key in SHARD_KEYS:
        if key == 'legal_data':
            continue
        deduped[key] = raw.pop(key)[unique_indices]
    deduped['legal_data'] = legal_data
    return deduped


class GPUData:
    def __init__(self, arrays: dict, device):
        self.device = device
        self.n = len(arrays['board'])
        # keep legal_data as uint16 on the CPU; it is cast to int32 on the GPU
        # (CUDA cannot index a UInt16 tensor, and casting on the CPU would double
        # its size for no benefit).
        self.tensors = {key: torch.from_numpy(array).to(device) for key, array in arrays.items()}
        self.tensors['legal_data'] = self.tensors['legal_data'].to(torch.int32)

        moves_per_position = self.tensors['legal_len'].long()
        self.first_move_of_position = torch.cumsum(moves_per_position, 0) - moves_per_position

    def _legal_moves(self, indices: torch.Tensor):
        n_positions = indices.numel()
        moves_per_position = self.tensors['legal_len'][indices].long()

        position_of_move = torch.repeat_interleave(
            torch.arange(n_positions, device=self.device), moves_per_position)
        first_move_in_batch = torch.cumsum(moves_per_position, 0) - moves_per_position
        move_number_within_position = (
            torch.arange(position_of_move.numel(), device=self.device) - first_move_in_batch[position_of_move])
        flat_index = self.first_move_of_position[indices][position_of_move] + move_number_within_position

        codes = self.tensors['legal_data'][flat_index].long()
        return position_of_move, codes

    def batch(self, indices: torch.Tensor) -> dict:
        t = self.tensors
        legal_position, codes = self._legal_moves(indices)
        legal_from, legal_to, legal_is_capture = decode_move(codes)

        return {
            'board': t['board'][indices], 'ep': t['ep'][indices], 'castle': t['castle'][indices],
            'turn': t['turn'][indices], 'half': t['half'][indices], 'rep': t['rep'][indices],
            'next_from': t['next_from'][indices].long(), 'next_to': t['next_to'][indices].long(),
            'moved': t['moved'][indices], 'captured': t['captured'][indices],
            'chk': t['chk'][indices].long(), 'ptte': t['ptte'][indices], 'result': t['result'][indices],
            'legal_position': legal_position, 'legal_from': legal_from, 'legal_to': legal_to,
            'legal_is_capture': legal_is_capture,
        }


def load_data(data_dir: str, device, val_shards: int = 2):
    files = sorted(glob.glob(os.path.join(data_dir, 'shard_*.npz')))
    if not files:
        raise SystemExit(f'no training shards in {data_dir} - run preprocess.py first')
    val_shards = min(val_shards, len(files) - 1)
    train_files = files[:-val_shards] if val_shards else files
    val_files = files[-val_shards:] if val_shards else files
    train = GPUData(load_shards(train_files, strict_dedup=True), device)
    val = GPUData(load_shards(val_files, strict_dedup=True), device)
    return train, val
