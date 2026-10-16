import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn-chess'))
from model import compute_loss
from targets_and_loss import make_targets

GRAD_CLIP = 5.0
MAX_EVAL_BATCHES = 50
MAX_WARMUP_STEPS = 500
LR_FLOOR = 0.05
LR_RANGE = 0.95


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
        loss, _ = compute_loss(outputs, targets, graph)
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


def train_step(model, optimizer, scheduler, graph_builder, data, indices: torch.Tensor):
    outputs, targets, graph = forward_batch(model, graph_builder, data, indices)
    loss, losses = compute_loss(outputs, targets, graph)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    return loss, losses


def save_checkpoint(model, path: str):
    torch.save({'model': model.state_dict()}, path)
