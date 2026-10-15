import functools

import numpy as np
import torch

PIECE_ORDER = ['rook', 'knight', 'bishop', 'queen', 'king', 'pawn']
CHESS_TO_PROJECT_TYPE = {1: 5, 2: 1, 3: 2, 4: 0, 5: 3, 6: 4}
PIECE_TYPE_OF_CODE = [0] + [CHESS_TO_PROJECT_TYPE[(code - 1) % 6 + 1] for code in range(1, 13)]


def _offsets():
    squares = np.arange(64)
    rows, cols = squares // 8, squares % 8
    dy = rows[None, :] - rows[:, None]
    dx = cols[None, :] - cols[:, None]
    return dx, dy


def _rook(dx, dy):
    return (dx == 0) | (dy == 0)


def _knight(dx, dy):
    return (dx ** 2 + dy ** 2 == 5) | ((np.abs(dx) + np.abs(dy)) == 0)


def _bishop(dx, dy):
    return np.abs(dx) == np.abs(dy)


def _queen(dx, dy):
    return _rook(dx, dy) | _bishop(dx, dy)


def _king(dx, dy):
    return dx ** 2 + 2 * dy ** 2 < 5


def _pawn(dx, dy):
    return 2 * dx ** 2 + dy ** 2 < 5


MOBILITY_FUNCS = [_rook, _knight, _bishop, _queen, _king, _pawn]


@functools.lru_cache(maxsize=None)
def mobility_table(device) -> torch.Tensor:
    dx, dy = _offsets()
    table = np.stack([func(dx, dy) for func in MOBILITY_FUNCS])
    return torch.tensor(table, device=device)


def piece_types(board: torch.Tensor) -> torch.Tensor:
    return torch.tensor(PIECE_TYPE_OF_CODE, device=board.device)[board]


def build_adjacency(board: torch.Tensor) -> torch.Tensor:
    squares = torch.arange(64, device=board.device)
    piece_type = piece_types(board)
    occupied = board > 0

    adjacency = mobility_table(board.device)[piece_type, squares] & occupied[..., None]
    adjacency = adjacency | torch.eye(64, dtype=torch.bool, device=board.device)
    return adjacency | adjacency.transpose(1, 2)


def board_to_coo(board: torch.Tensor):
    position, from_square, to_square = build_adjacency(board).nonzero(as_tuple=True)
    return position, from_square, to_square
