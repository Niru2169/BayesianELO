# ChessGNN - PyTorch / GPU port

A graph neural network that plays chess. A batch of positions is merged into one disjoint graph
(one node per square, no edges between different positions), and the whole pipeline runs on the GPU.

## How it works
```
games.pgn --data prep--> shards (positions + labels) --train.py--> chessgnn.pt --plays.py--> moves

board ──graph_utils.py──> edges (pieces' mobility rules)
      ──features.py────> node features (12) + global features (8)
      ──model.py───────> input block -> 10 attention blocks -> 11 output heads
```
- Each square is a node. Two squares are connected if some piece type could move between them
  (rook, bishop, knight, queen, king and pawn patterns), plus every square connects to itself.
- The network scores every edge as a possible move. At play time only the legal ones are considered.
- Training predicts all 11 heads at once; the loss is the sum of the 11 head losses (`LOSSES` and `compute_loss` in `model.py`).

## Setup
```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install numpy python-chess
```

## Run
```
# 1) once: PGN -> shards
.venv\Scripts\python preprocess.py games.pgn --out data --sample-rate 0.1

# 2) train (whole dataset is held on the GPU)
.venv\Scripts\python train.py --data data --epochs 3 --batch-size 256 --save chessgnn.pt

# 3) play (add --search for minimax on top of the network)
.venv\Scripts\python plays.py chessgnn.pt --model-plays black

# 4) optional: estimate Elo against Stockfish
.venv\Scripts\python play.py --stockfish "C:\stockfish\stockfish.exe" --elo 1320 --games 10
```

## Files
- `graph_utils.py` the 6 piece-mobility rules, adjacency matrix and edge list
- `features.py` node features, global features
- `layers.py` scatter / softmax / concat operations used by the model
- `model.py` input block, full (attention) block, multi-head attention, output heads, `LOSSES` and the loss (`compute_loss`: the sum of the 11 head losses)
- `preprocess.py` PGN -> shards: result filter, random subsampling, opening down-weighting
- `train.py` training loop
- `plays.py` play against the model (greedy policy, or `--search`)
- `play.py` plays the model against Stockfish to estimate Elo

Supporting code lives in the sibling folder `..\ours_gpu_extras`: move encoding (`chessutil.py`), batched graph
building (`graph_builder.py`), training targets (`targets_and_loss.py`), training helpers (`training_utils.py`),
data loading (`data.py`), label computation for preprocessing (`labels.py`) and the minimax search (`search.py`).
