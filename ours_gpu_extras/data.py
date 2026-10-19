import glob
import os

import numpy as np
import torch

from chessutil import decode_move

SHARD_KEYS = ['board', 'ep', 'castle', 'turn', 'half', 'rep', 'next_from', 'next_to',
              'moved', 'captured', 'chk', 'ptte', 'result', 'legal_len', 'legal_data']


def load_shards(files: list, strict_dedup: bool = True) -> dict:
    parts = {key: [] for key in SHARD_KEYS}
    for file in files:
        with np.load(file) as shard:
            for key in SHARD_KEYS:
                parts[key].append(shard[key])
    raw = {key: np.concatenate(arrays) for key, arrays in parts.items()}

    if not strict_dedup:
        return raw

    # Strict deduplication across (board [64], turn [1], castle [1], ep [1])
    state_sig = np.concatenate([
        raw['board'],
        raw['turn'][:, None],
        raw['castle'][:, None],
        raw['ep'][:, None].view(np.uint8),
    ], axis=1)
    _, unique_indices = np.unique(state_sig, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)

    # Rebuild variable-length legal_data slices for unique positions
    lengths = raw['legal_len']
    starts = np.cumsum(lengths) - lengths
    kept_legal_slices = [
        raw['legal_data'][starts[i]:starts[i] + lengths[i]] for i in unique_indices
    ]

    deduped = {
        key: raw[key][unique_indices] for key in SHARD_KEYS if key != 'legal_data'
    }
    deduped['legal_data'] = (
        np.concatenate(kept_legal_slices) if kept_legal_slices else np.empty(0, dtype=np.uint16)
    )
    return deduped


class GPUData:
    def __init__(self, arrays: dict, device):
        self.device = device
        self.n = len(arrays['board'])
        arrays = dict(arrays, legal_data=arrays['legal_data'].astype(np.int32))
        self.tensors = {key: torch.from_numpy(array).to(device) for key, array in arrays.items()}

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
        raise FileNotFoundError(f'no shard_*.npz files in {data_dir!r} - run preprocess.py first')

    # Always keep at least one shard for training: a small PGN yields fewer shards
    # than the default hold-out asks for, and an empty train set is a hard error.
    n_val = max(0, min(val_shards, len(files) - 1))
    train = GPUData(load_shards(files[:len(files) - n_val], strict_dedup=True), device)
    if n_val:
        val = GPUData(load_shards(files[len(files) - n_val:], strict_dedup=True), device)
    else:
        # Single shard: evaluate on the training data. The loss curve is still
        # meaningful at this size; a real hold-out needs at least two shards.
        print(f'warning: only {len(files)} shard available - evaluating on the training data')
        val = train
    return train, val
