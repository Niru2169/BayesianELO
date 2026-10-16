import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ours_gpu_extras'))
from data import load_data
from graph_builder import GraphBuilder
from training_utils import evaluate, make_lr_schedule, save_checkpoint, train_step
from model import ChessGNN

LEARNING_RATE = 5e-4
LOG_EVERY = 200
EVAL_EVERY = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save', default='chessgnn.pt')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available - check the torch install'
    device = torch.device('cuda')

    print('loading data ...', flush=True)
    train_data, val_data = load_data(args.data, device)
    print(f'train positions: {train_data.n:,}   val positions: {val_data.n:,}', flush=True)

    graph_builder = GraphBuilder(device)
    model = ChessGNN().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=LEARNING_RATE)

    steps_per_epoch = train_data.n // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scheduler = make_lr_schedule(optimizer, total_steps)

    step = 0
    model.train()
    while step < total_steps:
        shuffled = torch.randperm(train_data.n, device=device)
        for start in range(0, steps_per_epoch * args.batch_size, args.batch_size):
            if step >= total_steps:
                break
            indices = shuffled[start:start + args.batch_size]
            loss, losses = train_step(model, optimizer, scheduler, graph_builder, train_data, indices)
            step += 1

            if step % LOG_EVERY == 0:
                print(f'step {step}/{total_steps}  loss {loss.item():.4f}  '
                      f'next_move {losses["next_move"].item():.3f}  '
                      f'lr {scheduler.get_last_lr()[0]:.2e}', flush=True)
            if step % EVAL_EVERY == 0 or step == total_steps:
                print(f'  [val] loss {evaluate(model, graph_builder, val_data, args.batch_size):.4f}', flush=True)
                save_checkpoint(model, args.save)

    save_checkpoint(model, args.save)
    print(f'saved {args.save}')


if __name__ == '__main__':
    main()
