import torch

from graph_utils import piece_types

NODE_FEATURES = 12
GLOBAL_FEATURES = 8
EDGE_GEO_FEATURES = 14  # geometric & directional edge features (AlphaGateau style)
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


def edge_geometric_features(from_square: torch.Tensor, to_square: torch.Tensor) -> torch.Tensor:
    """14-dim spatial/directional features for every directed edge (AlphaGateau style).

    Rank/file offsets, normalized Manhattan/Chebyshev distances, self-loop and
    knight-jump indicators, and an 8-way compass one-hot.
    """
    r1, c1 = (from_square // 8).float(), (from_square % 8).float()
    r2, c2 = (to_square // 8).float(), (to_square % 8).float()
    dr = (r2 - r1) / 7.0
    dc = (c2 - c1) / 7.0

    abs_dr, abs_dc = (r2 - r1).abs(), (c2 - c1).abs()
    manhattan = (abs_dr + abs_dc) / 14.0
    chebyshev = torch.maximum(abs_dr, abs_dc) / 7.0

    is_self = (from_square == to_square).float()
    is_knight = ((abs_dr * abs_dr + abs_dc * abs_dc) == 5).float()

    is_ortho = (abs_dr == 0) | (abs_dc == 0)
    is_diag = (abs_dr == abs_dc) & (abs_dr > 0)

    dir_n = ((r2 > r1) & (abs_dc == 0)).float()
    dir_s = ((r2 < r1) & (abs_dc == 0)).float()
    dir_e = ((c2 > c1) & (abs_dr == 0)).float()
    dir_w = ((c2 < c1) & (abs_dr == 0)).float()
    dir_ne = (is_diag & (r2 > r1) & (c2 > c1)).float()
    dir_nw = (is_diag & (r2 > r1) & (c2 < c1)).float()
    dir_se = (is_diag & (r2 < r1) & (c2 > c1)).float()
    dir_sw = (is_diag & (r2 < r1) & (c2 < c1)).float()

    return torch.stack([
        dr, dc, manhattan, chebyshev, is_self, is_knight,
        dir_n, dir_s, dir_e, dir_w, dir_ne, dir_nw, dir_se, dir_sw,
    ], dim=-1)
