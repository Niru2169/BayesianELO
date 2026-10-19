# train.py explained

`train.py` is the training loop: it loads the shards made by `preprocess.py`, repeatedly builds graphs from random batches of positions,
runs `ChessGNN`, computes the 11-part loss, and updates the weights. The output is `chessgnn.pt`, which `plays.py`, `play.py` and `visualize_move.py` load.

```
data/shard_*.npz ─ data.py ─> GPU tensors ─┐
                                           │  every step
   random 256 positions ── GPUData.batch ──┼─> graph_builder.build ─> graph ─┐
                                           └─> make_targets ──────> targets, legal_in
                                                                              ▼
                                       model(graph, legal_in) ─> outputs ─> compute_loss ─> backward ─> clip ─> NAdam ─> LR schedule
```

`train.py` itself is short; the real work is in `ours_gpu_extras/{data,graph_builder,targets_and_loss,training_utils}.py` and `model.py`. This document covers all of them as they appear in a training step.

Command (from the README): `python train.py --data data --epochs 3 --batch-size 256 --save chessgnn.pt`

---------------------------------------------------------------------------------------------------

## 1. `train.py` line by line

### Imports / constants
- `sys.path.append(.../ours_gpu_extras)` makes `data`, `graph_builder`, `training_utils` importable. (Gotcha: `ours_gpu` also contains a *folder* called `data/`.
  It has no `__init__.py`, so Python prefers the real `data.py` module found in `ours_gpu_extras`. It works, but don't add an `__init__.py` to that folder.)
- `LEARNING_RATE = 5e-4`, `LOG_EVERY = 200` steps, `EVAL_EVERY = 2000` steps.

### Arguments
| flag | default | meaning |
|------|---------|---------|
| `--data` | `data` | folder with `shard_*.npz` |
| `--epochs` | 3 | passes over the training set |
| `--batch-size` | 256 | positions per step |
| `--save` | `chessgnn.pt` | checkpoint path |

### Setup
```python
assert torch.cuda.is_available()        # training is GPU-only (RTX 5090 laptop GPU on this machine)
train_data, val_data = load_data(args.data, device)      # all of it copied to GPU memory
graph_builder = GraphBuilder(device)
model     = ChessGNN().to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=5e-4)
steps_per_epoch = train_data.n // batch_size             # 4,764,749 // 256 = 18,612
total_steps     = steps_per_epoch * epochs               # 55,836 for 3 epochs
scheduler = make_lr_schedule(optimizer, total_steps)
```

### The loop
```python
step = 0; model.train()
while step < total_steps:
    shuffled = torch.randperm(train_data.n, device=device)           # new random order every epoch
    for start in range(0, steps_per_epoch*batch_size, batch_size):   # whole batches only; the remainder is dropped
        indices = shuffled[start:start+batch_size]
        loss, losses = train_step(...)
        step += 1
        every 200 steps : print step, total loss, next_move loss, current lr
        every 2000 steps (and the last): print validation loss; save checkpoint
save_checkpoint(model, args.save)
```
Logging calls `loss.item()` only every 200 steps (each `.item()` forces a GPU->CPU sync, so avoiding it on other steps keeps the GPU busy).
`model.train()` has no practical effect: the network has no dropout or batch-norm, so train and eval modes compute the same thing.

---------------------------------------------------------------------------------------------------

## 2. Loading the data (`ours_gpu_extras/data.py`)

`load_data(data_dir, device, val_shards=2)`:
- Sort all `shard_*.npz`; the **last 2 shards** become validation (49,384 positions), the **rest** training (4,764,749).
- `load_shards` reads each file and concatenates each of the 15 fields across shards (`board, ep, castle, turn, half, rep, next_from, next_to, moved, captured, chk, ptte, result, legal_len, legal_data`).

`GPUData.__init__`: converts `legal_data` from uint16 to int32 (so arithmetic works), copies **every array to the GPU once**.
Rough GPU footprint: boards 4.8M x 64 B ≈ 0.3 GB, legal moves 4.8M x ~28 x 4 B ≈ 0.5 GB, small columns ≈ 0.2 GB, total about 1 GB of the 24 GB card. After that, no data moves between CPU and GPU during training.
It also precomputes `first_move_of_position = cumsum(legal_len) - legal_len`: where each position's legal-move list starts inside the flat `legal_data` array.

### `GPUData.batch(indices)` - gather one batch entirely on the GPU
Per-position fields are just `t[key][indices]`. The only tricky part is the ragged legal-move lists (`_legal_moves`). Toy example, a batch of 3 positions with 2, 3 and 1 legal moves:
```
moves_per_position      = [2, 3, 1]
position_of_move        = [0,0, 1,1,1, 2]                       (repeat_interleave)
first_move_in_batch     = [0, 2, 5]                              (cumsum - self)
move_number_in_position = arange(6) - first_move_in_batch[position_of_move] = [0,1, 0,1,2, 0]
flat_index              = first_move_of_position[indices][position_of_move] + move_number_in_position
codes                   = legal_data[flat_index]
```
Then `decode_move(codes)` gives `legal_from, legal_to, legal_is_capture` (`to = code % 64`, `from = (code//64) % 64`, `capture = code // 4096`).
Returned dict: board, ep, castle, turn, half, rep, next_from/next_to (as int64), moved, captured, chk, ptte, result, and the four `legal_*` lists (each with one entry per legal move in the batch).

---------------------------------------------------------------------------------------------------

## 3. One training step - `training_utils.train_step`

```python
outputs, targets, graph = forward_batch(model, graph_builder, data, indices)
loss, losses = compute_loss(outputs, targets, graph)
optimizer.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
scheduler.step()
return loss, losses
```

### 3a. `forward_batch`
```python
batch   = data.batch(indices)
graph   = graph_builder.build(batch['board'], batch['ep'], batch['castle'], batch['turn'], batch['half'], batch['rep'])
targets, legal_in = make_targets(graph, batch)
outputs = model(graph, legal_in)
```

### 3b. `GraphBuilder.build` (`graph_builder.py`) - board numbers -> graph tensors (no gradients)
1. `board_to_coo(board)`: builds the `(P, 64, 64)` boolean adjacency (piece mobility patterns, self-loops, symmetrised) and returns its `nonzero()` triple `(position, from_square, to_square)`.
   `nonzero` lists entries in row-major order, so edges come out **sorted by (position, from, to)**.
2. `node_features(board, ep)` -> `(P*64, 12)`; `global_features(castle, turn, half, rep)` -> `(P, 8)`.
3. `receiver = position*64 + from_square`, `sender = position*64 + to_square`, `edge_position = position`.
4. `edge_key = position*4096 + from*64 + to`. It is sorted, so `torch.searchsorted` can find any (position, from, to) quickly, which is how moves are matched to edges.
5. `edges_per_position = bincount(position)` (used to normalise `legal_moves` losses).

Example sizes for 256 positions: 16,384 nodes, roughly 150,000 edges (~600 per position).

### 3c. `make_targets` (`targets_and_loss.py`) - labels aligned to the graph
| target | how it's computed | shape |
|--------|------------------|-------|
| `legal_moves` | `edge_key` lookup of every legal (pos, from, to); `1.0` at the matching edges, 0 elsewhere | `(E,)` |
| `legal_captures` | same, but only those legal moves whose code had the capture flag | `(E,)` |
| `next_move_edge`, `next_move_found` | lookup of the played move (`arange(P)`, `next_from`, `next_to`) -> edge index and whether it exists as an edge | `(P,)` |
| `next_piece_moved`, `next_piece_captured` | unpack the 64-bit masks into 64 flags per position, flattened | `(P*64,)` |
| `checking_in_1`, `checking_in_3`, `checked_in_1`, `checked_in_3` | bits 0, 1, 3, 4 of `chk` | `(P,)` |
| `plies_till_end` | `exp(-0.2 * (ptte - 1))` | `(P,)` |
| `final_result` | class id 0/1/2 | `(P,)` |

`edge_lookup` = `searchsorted` then verify the key at that index is equal; `edge_exists` tells which lookups truly matched.

**Important**: `make_targets` also returns the `legal` vector, and `forward_batch` passes that same vector to the model as `legal_in`
(the model's input bit). So the model is told which edges are legal as an *input*, and the `legal_moves` head is trained to output that very bit - an easy head. Likewise `legal_captures` is just "legal and capture".

### 3d. Forward and loss
`model(graph, legal_in)` -> 11 outputs; `compute_loss` -> the sum of 11 losses (details in `model_EXPLAINED.md`). The most important term is `next_move`: cross-entropy where the "classes" are all edges of the position and the correct class is the edge of the move that was actually played.

### 3e. Backward, clipping, update
- `zero_grad(set_to_none=True)`: clears old gradients (setting them to `None` is cheaper than zeroing).
- `loss.backward()`: backprop through the heads, the 10 blocks, the input block (the scatter/gather ops are differentiable).
- `clip_grad_norm_(..., 5.0)`: if the gradient vector over all 269,925 parameters has length > 5, scale it down to 5. Prevents the occasional huge update from destabilising training.
- `optimizer.step()`: **NAdam** (Adam with Nesterov momentum), default betas (0.9, 0.999), no weight decay. Adam keeps a running average of each weight's gradient and of its square, so each weight gets its own step size.
- `scheduler.step()`: advance the learning-rate schedule by one step.

---------------------------------------------------------------------------------------------------

## 4. Learning-rate schedule (`make_lr_schedule`)

```
warmup_steps = min(500, total_steps // 10 + 1)               # = 500 here
multiplier(step) = min(1, (step+1)/warmup_steps)  *  ( 0.05 + 0.95 * 0.5 * (1 + cos(pi * step/total_steps)) )
lr = 5e-4 * multiplier
```
- **Warm-up**: the first 500 steps climb linearly from ~0 to 5e-4 (random weights + big steps = unstable, so start gently).
- **Cosine decay**: after that, a smooth fall from 5e-4 down to a floor of 5% (2.5e-5) at the end.

Values for this run (55,836 steps), computed from the formula:

| step | 0 | 99 | 249 | 499 | 10,000 | 18,612 (end epoch 1) | 27,918 | 37,224 (end epoch 2) | 46,530 | 55,835 |
|------|---|----|-----|-----|--------|----------------------|--------|----------------------|--------|--------|
| lr | 1.0e-6 | 1.0e-4 | 2.5e-4 | 5.0e-4 | 4.6e-4 | 3.8e-4 | 2.6e-4 | 1.4e-4 | 5.7e-5 | 2.5e-5 |

The scheduler is `LambdaLR`, which multiplies the optimizer's base lr (5e-4) by `multiplier(step)` every time `scheduler.step()` is called. The log line prints `scheduler.get_last_lr()`.

---------------------------------------------------------------------------------------------------

## 5. Validation - `training_utils.evaluate`

Every 2000 steps (and at the end):
```python
model.eval()
shuffled = randperm(val.n)[: batch_size * 50]            # a fresh random subset: 50 batches = 12,800 positions
for each batch: forward_batch → compute_loss → add loss
model.train()
return average total loss
```
It runs under `@torch.no_grad()` (no gradients, less memory). Only the **total** (sum of 11 losses) is reported, not individual heads.
Because the subset is re-drawn every time, the validation number is a bit noisy (it is a sample of the 49,384 validation positions).
The model is not early-stopped or reverted from it; validation only gives you the number and a checkpoint is saved right after.

## 6. Checkpoints

`save_checkpoint(model, path)` -> `torch.save({'model': model.state_dict()}, path)`.
`plays.load_model` does `ChessGNN().load_state_dict(torch.load(path)['model'])`.

Note: the current `chessgnn.pt` in this folder contains four keys (`model`, `opt`, `sched`, `step`), i.e. it was written by an earlier version of `train.py` that also saved optimizer, scheduler and step.
The current code saves only `model` and has no resume option, so running `train.py` again starts from random weights and **overwrites** `chessgnn.pt` (use `--save other.pt` to keep the old one).

---------------------------------------------------------------------------------------------------

## 7. What the log looks like and what to watch

```
loading data ...
train positions: 4,764,749   val positions: 49,384
step 200/55836  loss X.XXXX  next_move Y.YYY  lr 2.5e-04...
  [val] loss X.XXXX              (every 2000 steps; checkpoint saved)
saved chessgnn.pt
```
- `loss` is the sum of 11 terms, so it is larger than any single head. `next_move` is printed separately because it is the one that matters for playing strength.
- A uniform guess over a position's ~600 edges would give `next_move ≈ ln(600) ≈ 6.4`; a position with 30 legal moves where the model is perfectly uncertain between them gives `ln(30) ≈ 3.4`. Since the legal bit is an input, the model can in principle get well below the 6.4 level just by learning to favour legal edges, and further improvement means learning which legal moves players actually choose. (I did not run training, so I have no actual loss curve to quote.)
- Training time per step is dominated by the edge tensors (≈150k x 144 floats in each block); batch size 256 fits easily on the 24 GB GPU.

## 8. Things worth knowing
- No data augmentation, no mixed precision (`autocast`), no `torch.compile`, no gradient accumulation. Losses call `.float()` on outputs, which is a no-op in fp32.
- All 11 losses are weighted 1.0. Heads with larger raw loss values (like `next_move`) naturally contribute larger gradients.
- One epoch = one pass over 4.76M positions, but they come from only a fraction of each game's positions (sampling in `preprocess.py`), so positions from the same game are correlated but are drawn randomly across the dataset.
- The same positions are used in every epoch with identical labels; a model with 270k parameters on 4.8M positions is unlikely to overfit heavily in 3 epochs.
