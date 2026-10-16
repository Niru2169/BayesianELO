import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn-chess'))
from chessutil import MAX_COUNT, board_array, castle_bits, ep_square
from features import global_features, node_features
from graph_utils import board_to_coo

MAX_EDGES_PER_POSITION = 64 * 64


def make_edge_key(position, from_square, to_square):
    return position * MAX_EDGES_PER_POSITION + from_square * 64 + to_square


def edge_lookup(edge_key: torch.Tensor, query_keys: torch.Tensor):
    index = torch.searchsorted(edge_key, query_keys).clamp_(max=edge_key.numel() - 1)
    return index, edge_key[index] == query_keys


def find_move_edges(graph: dict, position, from_square, to_square):
    edge_key = graph['edge_key']
    edge_index, is_edge = edge_lookup(edge_key, make_edge_key(position, from_square, to_square))
    legal_in = torch.zeros(edge_key.numel(), device=edge_key.device)
    legal_in[edge_index[is_edge]] = 1.0
    return edge_index, is_edge, legal_in


class GraphBuilder:
    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def build(self, board, en_passant, castle, turn, halfmove, repetition) -> dict:
        n_positions = board.shape[0]
        board = board.long()
        position, from_square, to_square = board_to_coo(board)

        return dict(
            node_features=node_features(board, en_passant),
            global_features=global_features(castle, turn, halfmove, repetition),
            receiver=position * 64 + from_square,
            sender=position * 64 + to_square,
            edge_position=position,
            edge_key=make_edge_key(position, from_square, to_square),
            edges_per_position=torch.bincount(position, minlength=n_positions),
            n_positions=n_positions,
        )

    def build_from_boards(self, boards: list, repetitions: list) -> dict:
        def to_tensor(values):
            return torch.tensor(values, device=self.device)

        return self.build(
            torch.from_numpy(np.stack([board_array(b) for b in boards])).to(self.device),
            to_tensor([ep_square(b) for b in boards]),
            to_tensor([castle_bits(b) for b in boards]),
            to_tensor([int(b.turn) for b in boards]),
            to_tensor([min(b.halfmove_clock, MAX_COUNT) for b in boards]),
            to_tensor([min(r, MAX_COUNT) for r in repetitions]))
