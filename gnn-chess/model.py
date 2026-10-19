import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import EDGE_GEO_FEATURES, GLOBAL_FEATURES, NODE_FEATURES
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

# Multi-task loss curriculum (Zieba 2023, Fig. 6.3): primary heads stay fixed,
# useful lookahead auxiliaries stay fixed, and the scaffold heads that are
# learned trivially early (or over-regularize late) decay out with progress.
PRIMARY_HEADS = {'next_move': 1.5, 'final_result': 1.5}
USEFUL_AUX_HEADS = {'next_piece_moved': 0.5, 'next_piece_captured': 0.5, 'plies_till_end': 0.5}
DISCONTINUED_HEADS = {
    'legal_moves', 'legal_captures',
    'checked_in_1', 'checked_in_3', 'checking_in_1', 'checking_in_3',
}


def leaky(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, LEAKY_SLOPE)


def layer_norm() -> nn.LayerNorm:
    return nn.LayerNorm(EMBED_DIM, eps=LN_EPS)


class GATEAULayer(nn.Module):
    """Edge-feature-aware multi-head graph attention (Rigaux & Kashima, AlphaGateau).

    Attention logits and value projections use the full edge-node triplet
    ``[e_feat, v_receiver, v_sender]`` (``3 * EMBED_DIM``), so the updated edge
    representation participates in the same block's message passing.
    """

    def __init__(self):
        super().__init__()
        self.att1 = nn.Linear(3 * EMBED_DIM, GAT_HIDDEN)
        self.att2 = nn.Linear(GAT_HIDDEN, GAT_HEADS)
        self.proj = nn.Linear(3 * EMBED_DIM, GAT_HEADS * GAT_HEAD_OUT, bias=False)
        self.bias = nn.Parameter(torch.zeros(GAT_HEADS * GAT_HEAD_OUT))

    def forward(self, v: torch.Tensor, e_feat: torch.Tensor,
                receivers: torch.Tensor, senders: torch.Tensor) -> torch.Tensor:
        n_nodes = v.shape[0]
        triplet = concat_edge_with_ends(v, e_feat, receivers, senders)
        e_a = leaky(self.att2(leaky(self.att1(triplet))))
        e_a = scatter_and_softmax(e_a, receivers, n_nodes)
        messages = self.proj(triplet).view(-1, GAT_HEADS, GAT_HEAD_OUT)
        v_out = scatter_and_aggregate(e_a[..., None] * messages, receivers, n_nodes, 'sum')
        return leaky(v_out.flatten(1) + self.bias)


class CNNStem(nn.Module):
    """Lightweight 2-layer 3x3 residual CNN stem on the 8x8 grid (hybrid CNN+GNN).

    Captures local spatial patterns (pawn shields, adjacent squares) that have no
    mobility edge between them, before the embeddings are flattened back into nodes.
    """

    def __init__(self, channels: int = EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, v: torch.Tensor, n_positions: int) -> torch.Tensor:
        x = v.view(n_positions, 8, 8, EMBED_DIM).permute(0, 3, 1, 2).contiguous()
        residual = x
        out = leaky(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = leaky(out + residual)
        return out.permute(0, 2, 3, 1).contiguous().view(n_positions * 64, EMBED_DIM)


class InputBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1, self.v2 = nn.Linear(NODE_FEATURES, EMBED_DIM), nn.Linear(EMBED_DIM, EMBED_DIM)
        self.cnn_stem = CNNStem(EMBED_DIM)
        self.e0 = nn.Linear(NODE_FEATURES, EMBED_DIM)
        self.e1 = nn.Linear(2 * EMBED_DIM + 1 + EDGE_GEO_FEATURES, EMBED_DIM)
        self.u0, self.u1 = nn.Linear(NODE_FEATURES, EMBED_DIM), nn.Linear(EMBED_DIM + GLOBAL_FEATURES, EMBED_DIM)
        self.ln_v, self.ln_e, self.ln_u = layer_norm(), layer_norm(), layer_norm()

    def forward(self, graph, legal_moves):
        pos_feats, glob_feats = graph['node_features'], graph['global_features']
        receivers, senders = graph['receiver'], graph['sender']
        n_positions = graph['n_positions']

        v = leaky(self.v2(leaky(self.v1(pos_feats))))
        v = self.cnn_stem(v, n_positions)
        v = self.ln_v(v)

        e = leaky(self.e0(pos_feats))
        e = concat_edges_ends_only(e, receivers, senders)
        if 'edge_geo_features' in graph:
            e = torch.cat([e, legal_moves[:, None], graph['edge_geo_features']], dim=-1)
        else:
            e = torch.cat([e, legal_moves[:, None]], dim=-1)
        e = self.ln_e(leaky(self.e1(e)))

        u = leaky(self.u0(pos_feats))
        u = global_average_pooling(u, n_positions)
        u = self.ln_u(leaky(self.u1(torch.cat([u, glob_feats], -1))))
        return v, e, u


class FullBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(3 * EMBED_DIM, EDGE_HIDDEN)
        self.e2 = nn.Linear(EMBED_DIM + EDGE_HIDDEN, EMBED_DIM)
        self.gat = GATEAULayer()
        # Virtual dummy-node / global-to-node broadcast (ScalableAlphaZero style)
        self.u_to_v = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.ln_v, self.ln_e, self.ln_u = layer_norm(), layer_norm(), layer_norm()

    def forward(self, v_prev, e_prev, u_prev, graph):
        receivers, senders = graph['receiver'], graph['sender']
        n_positions = graph['n_positions']

        e = leaky(self.e1(concat_edge_with_ends(v_prev, e_prev, receivers, senders)))
        e = leaky(self.e2(concat_globals_with_anything(u_prev, e, graph['edge_position'])))

        v_agg = scatter_and_aggregate(e, receivers, v_prev.shape[0], 'mean')
        v_gat = self.gat(v_agg, e, receivers, senders)
        u_broadcast = self.u_to_v(u_prev).unsqueeze(1).expand(n_positions, 64, EMBED_DIM).reshape(-1, EMBED_DIM)
        v = v_gat + u_broadcast

        u = global_average_pooling(v, n_positions)

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


def compute_loss(outputs: dict, targets: dict, graph: dict, progress: float = 0.0):
    """Multi-task loss with a dynamic curriculum over the auxiliary heads.

    ``progress`` in [0.0, 1.0] (``current_step / total_steps``) decays the
    scaffold heads down to a 0.02 floor so they warm up the representation early
    without over-regularizing late training.
    """
    losses = {}
    weighted_total = 0.0
    aux_decay = max(0.02, (1.0 - progress) ** 2)

    for head, kind in LOSSES.items():
        raw_loss = LOSS_FUNCTIONS[kind](head, outputs, targets, graph)
        losses[head] = raw_loss

        if head in PRIMARY_HEADS:
            weight = PRIMARY_HEADS[head]
        elif head in USEFUL_AUX_HEADS:
            weight = USEFUL_AUX_HEADS[head]
        elif head in DISCONTINUED_HEADS:
            weight = 0.25 * aux_decay
        else:
            weight = 1.0

        weighted_total = weighted_total + weight * raw_loss

    return weighted_total, losses
