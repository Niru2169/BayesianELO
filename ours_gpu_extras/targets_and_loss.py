import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn-chess'))
from features import plies_till_end_transform
from graph_builder import edge_lookup, make_edge_key
from layers import scatter_and_aggregate


def _legal_move_labels(graph: dict, batch: dict):
    edge_key = graph['edge_key']
    legal_keys = make_edge_key(batch['legal_position'], batch['legal_from'], batch['legal_to'])
    edge_index, edge_exists = edge_lookup(edge_key, legal_keys)

    legal = torch.zeros(edge_key.numel(), device=edge_key.device)
    legal[edge_index[edge_exists]] = 1.0

    is_capture = edge_exists & (batch['legal_is_capture'] == 1)
    captures = torch.zeros(edge_key.numel(), device=edge_key.device)
    captures[edge_index[is_capture]] = 1.0
    return legal, captures


def _packed_to_square_flags(packed: torch.Tensor) -> torch.Tensor:
    powers_of_two = 2 ** torch.arange(63, device=packed.device)
    flags = torch.zeros(packed.shape[0], 64, device=packed.device)
    flags[:, :63] = ((packed[:, None] // powers_of_two) % 2).float()
    flags[:, 63] = (packed < 0).float()
    return flags.flatten()


def _check_flag(check_flags: torch.Tensor, flag: int) -> torch.Tensor:
    return ((check_flags // 2 ** flag) % 2).float()


def make_targets(graph: dict, batch: dict):
    n_positions = graph['n_positions']
    device = graph['edge_key'].device

    legal, captures = _legal_move_labels(graph, batch)

    next_move_keys = make_edge_key(torch.arange(n_positions, device=device), batch['next_from'], batch['next_to'])
    next_move_edge, next_move_found = edge_lookup(graph['edge_key'], next_move_keys)

    check_flags = batch['chk']
    targets = {
        'legal_moves': legal, 'legal_captures': captures,
        'next_move_edge': next_move_edge, 'next_move_found': next_move_found,
        'next_piece_moved': _packed_to_square_flags(batch['moved']),
        'next_piece_captured': _packed_to_square_flags(batch['captured']),
        'checking_in_1': _check_flag(check_flags, 0), 'checking_in_3': _check_flag(check_flags, 1),
        'checked_in_1': _check_flag(check_flags, 3), 'checked_in_3': _check_flag(check_flags, 4),
        'plies_till_end': plies_till_end_transform(batch['ptte']),
        'final_result': batch['result'].long(),
    }
    return targets, legal


def _binary_crossentropy(name, outputs, targets, graph):
    return F.binary_cross_entropy_with_logits(outputs[name].float(), targets[name])


def _binary_crossentropy_per_position(name, outputs, targets, graph):
    per_edge = F.binary_cross_entropy_with_logits(outputs[name].float(), targets[name], reduction='none')
    sums = scatter_and_aggregate(per_edge, graph['edge_position'], graph['n_positions'], 'sum')
    return (sums / graph['edges_per_position'].float()).mean()


def _sigmoid_mae(name, outputs, targets, graph):
    return (torch.sigmoid(outputs[name].float()) - targets[name]).abs().mean()


def _categorical_crossentropy(name, outputs, targets, graph):
    return F.cross_entropy(outputs[name].float(), targets[name])


def _softmax_over_edges(name, outputs, targets, graph):
    scores = outputs[name].float()
    edge_position, n_positions = graph['edge_position'], graph['n_positions']

    max_per_position = scatter_and_aggregate(scores, edge_position, n_positions, 'max')
    sum_exp = scatter_and_aggregate(torch.exp(scores - max_per_position[edge_position]),
                                    edge_position, n_positions, 'sum')
    log_sum_exp = torch.log(sum_exp) + max_per_position
    cross_entropy = log_sum_exp - scores[targets['next_move_edge']]
    found = targets['next_move_found'].float()
    return (cross_entropy * found).sum() / found.sum().clamp(min=1)
