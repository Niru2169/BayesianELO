import math
import os
import sys

import chess
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn-chess'))
from model import compute_loss
from targets_and_loss import make_targets

GRAD_CLIP = 5.0
MAX_EVAL_BATCHES = 50
MAX_WARMUP_STEPS = 500
LR_FLOOR = 0.05
LR_RANGE = 0.95


def batch_to_chess_boards(batch: dict, max_boards: int = 8) -> list[chess.Board]:
    """Reconstructs `chess.Board` instances from GPU batch tensors for search distillation."""
    boards = []
    n = min(max_boards, batch['board'].shape[0])
    board_np = batch['board'][:n].cpu().numpy()
    turn_np = batch['turn'][:n].cpu().numpy()
    castle_np = batch['castle'][:n].cpu().numpy()
    ep_np = batch['ep'][:n].cpu().numpy()
    half_np = batch['half'][:n].cpu().numpy()

    for i in range(n):
        b = chess.Board(None)
        for sq, code in enumerate(board_np[i]):
            if code > 0:
                color = chess.WHITE if code <= 6 else chess.BLACK
                ptype = int(code) if code <= 6 else int(code) - 6
                b.set_piece_at(sq, chess.Piece(ptype, color))
        b.turn = chess.WHITE if turn_np[i] == 1 else chess.BLACK
        c_bits = int(castle_np[i])
        castling_fen = (
            ('K' if c_bits & 1 else '') +
            ('Q' if c_bits & 2 else '') +
            ('k' if c_bits & 4 else '') +
            ('q' if c_bits & 8 else '')
        ) or '-'
        b.set_castling_fen(castling_fen)
        if ep_np[i] >= 0:
            b.ep_square = int(ep_np[i])
        b.halfmove_clock = int(half_np[i])
        if b.is_valid():
            boards.append(b)
    return boards


def forward_batch(model, graph_builder, data, indices: torch.Tensor):
    batch = data.batch(indices)
    graph = graph_builder.build(batch['board'], batch['ep'], batch['castle'],
                                batch['turn'], batch['half'], batch['rep'])
    targets, legal_in = make_targets(graph, batch)
    outputs = model(graph, legal_in)
    return outputs, targets, graph


@torch.no_grad()
def evaluate(model, graph_builder, data, batch_size: int) -> float:
    model.eval()
    shuffled = torch.randperm(data.n, device=data.device)[: batch_size * MAX_EVAL_BATCHES]
    total_loss, n_batches = 0.0, 0
    for start in range(0, shuffled.numel(), batch_size):
        outputs, targets, graph = forward_batch(model, graph_builder, data, shuffled[start:start + batch_size])
        loss, _ = compute_loss(outputs, targets, graph, progress=1.0)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / n_batches


def make_lr_schedule(optimizer, total_steps: int):
    warmup_steps = min(MAX_WARMUP_STEPS, total_steps // 10 + 1)

    def lr_multiplier(step):
        warmup = min(1.0, (step + 1) / warmup_steps)
        progress = min(1.0, step / total_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        decay = LR_FLOOR + LR_RANGE * cosine
        return warmup * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def train_step(model, optimizer, scheduler, graph_builder, data, indices: torch.Tensor,
               progress: float = 0.0):
    outputs, targets, graph = forward_batch(model, graph_builder, data, indices)
    loss, losses = compute_loss(outputs, targets, graph, progress=progress)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    return loss, losses


def save_checkpoint(model, path: str):
    torch.save({'model': model.state_dict()}, path)
