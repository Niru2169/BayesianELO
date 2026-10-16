import glob
import os

import numpy as np
import torch

from chessutil import decode_move

SHARD_KEYS = ['board', 'ep', 'castle', 'turn', 'half', 'rep', 'next_from', 'next_to',
              'moved', 'captured', 'chk', 'ptte', 'result', 'legal_len', 'legal_data']


def load_shards(files: list) -> dict:
    parts = {key: [] for key in SHARD_KEYS}
    for file in files:
        with np.load(file) as shard:
            for key in SHARD_KEYS:
                parts[key].append(shard[key])
    return {key: np.concatenate(arrays) for key, arrays in parts.items()}


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
    train = GPUData(load_shards(files[:-val_shards]), device)
    val = GPUData(load_shards(files[-val_shards:]), device)
    return train, val
