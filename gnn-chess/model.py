import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import GLOBAL_FEATURES, NODE_FEATURES
from layers import (
    concat_edge_with_ends, concat_edges_ends_only, concat_globals_with_anything,
    global_average_pooling, scatter_and_aggregate, scatter_and_softmax,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ours_gpu_extras'))
from targets_and_loss import (
    _binary_crossentropy, _binary_crossentropy_per_position, _categorical_crossentropy,
    _sigmoid_mae, _softmax_over_edges,
)

EMBED_DIM = 48
N_BLOCKS = 10
LN_EPS = 1e-3
LEAKY_SLOPE = 0.3
EDGE_HIDDEN = 32
GAT_HIDDEN = 48
GAT_HEADS = 24
GAT_HEAD_OUT = 2
HEAD_HIDDEN_1 = 72
HEAD_HIDDEN_2 = 32

LOSSES = {
    'next_piece_moved': 'binary_crossentropy',
    'next_piece_captured': 'binary_crossentropy',
    'legal_moves': 'binary_crossentropy_per_position',
    'legal_captures': 'binary_crossentropy_per_position',
    'checked_in_1': 'binary_crossentropy',
    'checked_in_3': 'binary_crossentropy',
    'checking_in_1': 'binary_crossentropy',
    'checking_in_3': 'binary_crossentropy',
    'plies_till_end': 'sigmoid_mae',
    'final_result': 'categorical_crossentropy',
    'next_move': 'softmax_over_edges',
}


def leaky(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, LEAKY_SLOPE)


def layer_norm() -> nn.LayerNorm:
    return nn.LayerNorm(EMBED_DIM, eps=LN_EPS)


class MultiHeadGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.att1 = nn.Linear(2 * EMBED_DIM, GAT_HIDDEN)
        self.att2 = nn.Linear(GAT_HIDDEN, GAT_HEADS)
        self.proj = nn.Linear(2 * EMBED_DIM, GAT_HEADS * GAT_HEAD_OUT, bias=False)
        self.bias = nn.Parameter(torch.zeros(GAT_HEADS * GAT_HEAD_OUT))

    def forward(self, v, receivers, senders):
        n_nodes = v.shape[0]
        e = concat_edges_ends_only(v, receivers, senders)
        e_a = leaky(self.att2(leaky(self.att1(e))))
        e_a = scatter_and_softmax(e_a, receivers, n_nodes)
        e = self.proj(e).view(-1, GAT_HEADS, GAT_HEAD_OUT)
        v = scatter_and_aggregate(e_a[..., None] * e, receivers, n_nodes, 'sum')
        return leaky(v.flatten(1) + self.bias)


class InputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1, self.v2 = nn.Linear(NODE_FEATURES, EMBED_DIM), nn.Linear(EMBED_DIM, EMBED_DIM)
        self.e0, self.e1 = nn.Linear(NODE_FEATURES, EMBED_DIM), nn.Linear(2 * EMBED_DIM + 1, EMBED_DIM)
        self.u0, self.u1 = nn.Linear(NODE_FEATURES, EMBED_DIM), nn.Linear(EMBED_DIM + GLOBAL_FEATURES, EMBED_DIM)
        self.ln_v, self.ln_e, self.ln_u = layer_norm(), layer_norm(), layer_norm()

    def forward(self, graph, legal_moves):
        pos_feats, glob_feats = graph['node_features'], graph['global_features']
        receivers, senders = graph['receiver'], graph['sender']

        v = leaky(self.v2(leaky(self.v1(pos_feats))))
        v = self.ln_v(v)

        e = leaky(self.e0(pos_feats))
        e = concat_edges_ends_only(e, receivers, senders)
        e = torch.cat([e, legal_moves[:, None]], -1)
        e = self.ln_e(leaky(self.e1(e)))

        u = leaky(self.u0(pos_feats))
        u = global_average_pooling(u, graph['n_positions'])
        u = self.ln_u(leaky(self.u1(torch.cat([u, glob_feats], -1))))
        return v, e, u


class FullBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(3 * EMBED_DIM, EDGE_HIDDEN)
        self.e2 = nn.Linear(EMBED_DIM + EDGE_HIDDEN, EMBED_DIM)
        self.gat = MultiHeadGAT()
        self.ln_v, self.ln_e, self.ln_u = layer_norm(), layer_norm(), layer_norm()

    def forward(self, v_prev, e_prev, u_prev, graph):
        receivers, senders = graph['receiver'], graph['sender']

        e = leaky(self.e1(concat_edge_with_ends(v_prev, e_prev, receivers, senders)))
        e = leaky(self.e2(concat_globals_with_anything(u_prev, e, graph['edge_position'])))

        v = scatter_and_aggregate(e, receivers, v_prev.shape[0], 'mean')
        v = self.gat(v, receivers, senders)

        u = global_average_pooling(v, graph['n_positions'])

        v = v_prev + v
        e = e_prev + e
        u = u_prev + u

        return self.ln_v(v), self.ln_e(e), self.ln_u(u)


class MLPHead(nn.Module):
    def __init__(self, out: int = 1):
        super().__init__()
        self.a = nn.Linear(EMBED_DIM, HEAD_HIDDEN_1)
        self.b = nn.Linear(HEAD_HIDDEN_1, HEAD_HIDDEN_2)
        self.c = nn.Linear(HEAD_HIDDEN_2, out)

    def forward(self, x):
        return self.c(leaky(self.b(leaky(self.a(x)))))


class ChessGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = InputBlock()
        self.blocks = nn.ModuleList([FullBlock() for _ in range(N_BLOCKS)])
        self.heads = nn.ModuleDict({
            name: MLPHead(3 if name == 'final_result' else 1) for name in LOSSES
        })

    def forward(self, graph, legal_moves):
        v, e, u = self.inp(graph, legal_moves)
        for block in self.blocks:
            v, e, u = block(v, e, u, graph)

        h = self.heads
        return {
            'next_piece_moved': h['next_piece_moved'](v).squeeze(-1),
            'next_piece_captured': h['next_piece_captured'](v).squeeze(-1),
            'next_move': h['next_move'](e).squeeze(-1),
            'legal_moves': h['legal_moves'](e).squeeze(-1),
            'legal_captures': h['legal_captures'](e).squeeze(-1),
            'checked_in_1': h['checked_in_1'](u).squeeze(-1),
            'checked_in_3': h['checked_in_3'](u).squeeze(-1),
            'checking_in_1': h['checking_in_1'](u).squeeze(-1),
            'checking_in_3': h['checking_in_3'](u).squeeze(-1),
            'plies_till_end': h['plies_till_end'](u).squeeze(-1),
            'final_result': h['final_result'](u),
        }



LOSS_FUNCTIONS = {
    'binary_crossentropy': _binary_crossentropy,
    'binary_crossentropy_per_position': _binary_crossentropy_per_position,
    'sigmoid_mae': _sigmoid_mae,
    'categorical_crossentropy': _categorical_crossentropy,
    'softmax_over_edges': _softmax_over_edges,
}


def compute_loss(outputs: dict, targets: dict, graph: dict):
    losses = {}
    for head, kind in LOSSES.items():
        losses[head] = LOSS_FUNCTIONS[kind](head, outputs, targets, graph)
    total = sum(losses.values())
    return total, losses
