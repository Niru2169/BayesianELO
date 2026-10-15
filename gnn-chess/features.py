import torch

from graph_utils import piece_types

NODE_FEATURES = 12
GLOBAL_FEATURES = 8
BOARD_CENTRE = 3.5
SQUARE_COORD_SCALE = 0.3
HALFMOVE_SCALE = 0.04
REPETITION_SCALE = 0.5
PLIES_DECAY = 0.2


def node_features(board: torch.Tensor, en_passant: torch.Tensor) -> torch.Tensor:
    n_positions = board.shape[0]
    squares = torch.arange(64, device=board.device)

    piece_type = piece_types(board)
    occupied = board > 0
    is_white = (board >= 1) & (board <= 6)
    is_black = board >= 7

    piece_one_hot = torch.nn.functional.one_hot(piece_type, 6).float() * occupied[..., None]
    square_colour = ((squares // 8 + squares % 8) % 2).float()
    rank = SQUARE_COORD_SCALE * ((squares // 8).float() - BOARD_CENTRE)
    file = SQUARE_COORD_SCALE * ((squares % 8).float() - BOARD_CENTRE)
    en_passant_one_hot = (squares[None, :] == en_passant.long()[:, None]).float()

    return torch.cat([
        piece_one_hot,
        is_black.float()[..., None],
        is_white.float()[..., None],
        square_colour.expand(n_positions, 64)[..., None],
        rank.expand(n_positions, 64)[..., None],
        file.expand(n_positions, 64)[..., None],
        en_passant_one_hot[..., None],
    ], dim=-1).reshape(n_positions * 64, NODE_FEATURES)


def _castle_right(castle: torch.Tensor, weight: int) -> torch.Tensor:
    return ((castle // weight) % 2).float()


def global_features(castle, turn, halfmove, repetition) -> torch.Tensor:
    castle = castle.long()
    turn = turn.long()
    return torch.stack([
        (turn == 1).float(),
        (turn == 0).float(),
        _castle_right(castle, 1),
        _castle_right(castle, 2),
        _castle_right(castle, 4),
        _castle_right(castle, 8),
        halfmove.float() * HALFMOVE_SCALE,
        repetition.float() * REPETITION_SCALE,
    ], dim=-1)


def plies_till_end_transform(n_plies: torch.Tensor) -> torch.Tensor:
    return torch.exp(-PLIES_DECAY * (n_plies.float() - 1.0))
