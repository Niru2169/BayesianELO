# ChessGNN (ours_gpu) - how the model works, start to finish

Everything below was checked against the code in `ours_gpu/` and `ours_gpu_extras/`, and the numbers come from actually running
`visualize_move.py` on its built-in position (`r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w`)
with the move **Rb1 (a1b1)** as the worked example.

Total parameters: **269,925** (input block 11,952 + 10 blocks x 19,304 + 11 heads 64,933).

(Per-file deep dives: `layers_EXPLAINED.md`, `model_EXPLAINED.md`, `preprocess_EXPLAINED.md`, `train_EXPLAINED.md`.)

---------------------------------------------------------------------------------------------------

## 0. The one idea you need first

**A move is never fed to the network as a token, a name, or an ID.**
"Rb1" is only ever represented as **an edge in a graph**: a directed arrow from square a1 to square b1.
The network never sees the text "Rb1". It sees:

1. the numbers describing square a1 (a white rook, rank/file coordinates),
2. the numbers describing square b1 (empty, light/dark colour, coordinates),
3. one extra bit saying "this edge is a legal move: 1 / not: 0".

Those get multiplied through the weights and out comes a **single number (a score) for that edge**.
The move with the best score among legal edges is played. That is the whole trick.

The integer "move code" in `chessutil.py` (`from*64 + to (+4096 if capture)`) is **bookkeeping only**: it is how
legal moves and training labels are stored in the dataset and looked up. It is never an input to a weight.

---------------------------------------------------------------------------------------------------

## 1. The three parallel streams (node / edge / global) and where they join

The model keeps three vectors, each 48 numbers wide, updated together through 11 stages (input block + 10 full blocks):

| stream | one per...        | count for 1 position | what it represents                         | symbol |
|--------|-------------------|----------------------|--------------------------------------------|--------|
| node   | square            | 64                   | "what is going on at this square"          | `v`    |
| edge   | (from, to) pair   | 608 here (varies)    | "the move/relationship from sq A to sq B"  | `e`    |
| global | whole position    | 1                    | "summary of the whole board + game state"  | `u`    |

**Yes, they join - constantly.** They only look separate in the input block. In every one of the 10 blocks:

```
  edge  <-  its own previous value + node[from] + node[to] + global         (nodes and global flow INTO edges)
  node  <-  average of all edges pointing at it, then attention over neighbours   (edges flow INTO nodes)
  global<-  average over all 64 nodes                                        (nodes flow INTO global)
```
then each stream gets a residual add and a LayerNorm. The node stream never reads the global vector directly;
it gets global information indirectly via the edges (edge update uses `u`, node update uses edges).
Edges are the "hub" - which is why the move score is read from the **edge** vector.

Which stream feeds which head:
- `next_move`, `legal_moves`, `legal_captures`  <- **edge** vector (608 outputs)
- `next_piece_moved`, `next_piece_captured`     <- **node** vector (64 outputs)
- `checked_in_1/3`, `checking_in_1/3`, `plies_till_end`, `final_result` <- **global** vector (1 output each, 3 for result)

---------------------------------------------------------------------------------------------------

## 2. Chess -> numbers (exact conversion)

### 2a. The board
`board_array()` gives 64 integers, index = `rank*8 + file` (a1=0, b1=1, ... h1=7, a2=8, ... h8=63):
`0` empty, `1..6` = white P N B R Q K, `7..12` = black P N B R Q K.

### 2b. Node features - 12 numbers per square (`features.py: node_features`)
| idx | name        | value |
|-----|-------------|-------|
| 0-5 | piece one-hot | rook, knight, bishop, queen, king, pawn (all 0 if empty) - colour NOT in here |
| 6   | is_black    | 1 if a black piece is on the square |
| 7   | is_white    | 1 if a white piece is on the square |
| 8   | sq_colour   | (rank+file) % 2 |
| 9   | rank        | 0.3 * (rank - 3.5)   -> -1.05 ... +1.05 |
| 10  | file        | 0.3 * (file - 3.5)   -> -1.05 ... +1.05 |
| 11  | en_passant  | 1 if this is the legal en-passant target square |

Real rows from the run:
```
a1 (R)   rook=1  is_white=1  rank=-1.05  file=-1.05          -> [1,0,0,0,0,0, 0,1, 0,-1.05,-1.05, 0]
b1 (.)   sq_colour=1  rank=-1.05  file=-0.75, rest 0          -> [0,0,0,0,0,0, 0,0, 1,-1.05,-0.75, 0]
```
So the model has no notion of "a1" or "b1" as names. It only knows coordinates as two small real numbers.

### 2c. Global features - 8 numbers per position (`features.py: global_features`)
`[white_to_move, black_to_move, castle_WK, castle_WQ, castle_BK, castle_BQ, halfmove_clock*0.04, repetition_count*0.5]`
For the demo position: `[1, 0, 0, 0, 0, 0, 0.0, 0.0]` (no castling rights left, halfmove 0, no repetition).
The board is NOT flipped for black; colour is just given by `is_white / is_black` and `white_to_move`.

### 2d. Edges - the graph (`graph_utils.py`)
Every square is a node. A directed edge `from -> to` exists if ANY of these hold:
- `from == to` (self loop; 64 of them),
- the piece standing on `from` could *geometrically* reach `to` with its type's pattern
  (rook = same row/col, bishop = diagonal, knight = L, queen = both, king = within 1 (and 2 along the rank, which covers castling), pawn = forward 1-2 and diagonals),
- the **reverse** of the above (the adjacency matrix is symmetrised).

Important properties (verified):
- The pattern ignores blocking pieces. a1's rook has edges to c1..h1 even though Bc1 is in the way. Legality is NOT encoded in the graph.
- Empty squares have no outgoing mobility edges of their own, but they receive edges from the pieces that could reach them (and via symmetry send them back).
- Pawn pattern is colour-blind (includes backwards); colour only enters through the features.
- Demo position: **608 directed edges** (a square has between 4 and 23 incoming edges, mean 9.5). a1 has 16 neighbours: itself, b1-h1, a2, b2, a3-a8.
- Receiver = from-square, sender = to-square. `edge #1` is `a1 -> b1`: receiver=0, sender=1. That is "Rb1".

### 2e. The legal-move bit
`find_move_edges()` looks up each legal (from, to) in the sorted `edge_key` array (`position*4096 + from*64 + to`) and writes a `1.0` on that edge.
Result: a vector of 608 numbers, 37 of them 1 (the 37 legal moves), the rest 0. This is the only place the model is told what is legal.
Promotions: queen/rook/bishop/knight promotion collapse into the same (from,to) edge; play code always promotes to queen.

### 2f. Putting it in the batch
Many positions are merged into ONE big disjoint graph: node index = `position*64 + square`. Edges never cross between positions.
All the tensor shapes below are for one position (N = 1): V = 64 nodes, E = 608 edges.

---------------------------------------------------------------------------------------------------

## 3. The forward pass, shape by shape (worked example: Rb1)

`W` = weight matrix, `b` = bias, `lk()` = leaky_relu with slope 0.3, `LN` = LayerNorm(48).

### 3a. Input block (`model.py: InputBlock`) - makes the first v, e, u

```
NODE STREAM
 node feats x (64 x 12)
   -> v1: Linear 12->48   (W 48x12, b 48)    -> lk   (64 x 48)
   -> v2: Linear 48->48   (W 48x48, b 48)    -> lk   (64 x 48)
   -> LN_v                                          = v0   (64 x 48)

EDGE STREAM
 node feats x (64 x 12)
   -> e0: Linear 12->48 (W 48x12)            -> lk   (64 x 48)   (a separate per-square embedding, not shared with v1)
   -> for each edge: [ e0[from] | e0[to] ]                         (608 x 96)
   -> append legal bit                                              (608 x 97)
   -> e1: Linear 97->48 (W 48x97, b 48)      -> lk   (608 x 48)
   -> LN_e                                          = e0'  (608 x 48)

GLOBAL STREAM
 node feats x (64 x 12)
   -> u0: Linear 12->48 (W 48x12)            -> lk   (64 x 48)
   -> mean over the 64 squares                              (1 x 48)
   -> append the 8 global features                          (1 x 56)
   -> u1: Linear 56->48 (W 48x56, b 48)      -> lk   (1 x 48)
   -> LN_u                                          = u0'  (1 x 48)
```
Concretely, for Rb1 the first multiplication is a1's 12 numbers times row 0 of `inp.v1.weight`:
`1.0*0.1171 (rook) + 1.0*0.3741 (is_white) + (-1.05)*(-0.2194) (rank) + (-1.05)*0.1686 (file) + bias(-0.0302) = +0.5143`
(all other terms are 0 because those features are 0). 48 such dot products make the first 48-number description of a1.

The edge a1->b1 becomes `[ e0(a1) 48 | e0(b1) 48 | legal=1 ]` = 97 numbers -> 48 numbers. Here the move meets the graph: its two ends
(rook on a1, empty light square on b1) and the "legal" bit are fused.

### 3b. One full block (`model.py: FullBlock`) - repeated 10 times with different weights

Inputs: `v_prev (64x48)`, `e_prev (608x48)`, `u_prev (1x48)`. For Rb1: edge #1, from-square a1.

```
(a) EDGE UPDATE                                   <- nodes + global flow into edges
    cat1 = [ e_prev | v_prev[from] | v_prev[to] ]       (608 x 144)
    e1   = lk( Linear 144->32 )                          (608 x 32)     W 32x144
    cat2 = [ u_prev | e1 ]                               (608 x 80)     (u_prev copied to every edge of its position)
    e_new= lk( Linear 80->48 )                           (608 x 48)     W 48x80

(b) EDGE -> NODE
    v_mean[i] = average of e_new over every edge whose receiver is i      (64 x 48)
      (a1 has 16 incoming edges, so v_mean[a1] = mean of 16 vectors)

(c) 24-HEAD GRAPH ATTENTION over each node's neighbours
    g    = [ v_mean[receiver] | v_mean[sender] ]         (608 x 96)
    score= lk( Linear 48->24 ( lk( Linear 96->48 (g) ) ) )  (608 x 24)  W 48x96 then W 24x48
    attn = softmax of score over all edges with the SAME receiver, separately for each of the 24 heads
    val  = Linear 96->48, no bias (g), reshaped 24 heads x 2 numbers   (608 x 24 x 2)   W 48x96
    out[i,h,:] = sum over incoming edges ( attn[edge,h] * val[edge,h,:] )              (64 x 24 x 2)
    v_gat = lk( flatten(out) 48 + bias )                 (64 x 48)

(d) NODE -> GLOBAL
    u_new = mean over the 64 squares of v_gat            (1 x 48)

(e) RESIDUAL + LAYERNORM (each stream separately)
    v = LN_v( v_prev + v_gat )     e = LN_e( e_prev + e_new )     u = LN_u( u_prev + u_new )
```
Parameters per block: e1 4,640 + e2 3,888 + attention 10,488 (att1 4,656 + att2 1,176 + proj 4,608 + bias 48) + 3 LayerNorms 288 = **19,304**.

In block 1 for Rb1 the attention for the receiver a1 (its 16 neighbours) shows, for example, head 10 putting 94% weight on g1 and head 13 putting 43%
on a3. Head 0's raw scores run from -0.42 (c1) to +2.17 (a3) and the softmax turns them into weights that sum to 1.
After 10 blocks every vector has been mixed with neighbours up to ~10 hops away through the geometric graph.

### 3c. The move-scoring head (`model.py: MLPHead`, name `next_move`)

Applied to **every** edge vector of the last block (608 x 48):
```
 e (608 x 48) -> a: Linear 48->72 (W 72x48) -> lk -> b: Linear 72->32 (W 32x72) -> lk -> c: Linear 32->1 (W 1x32) -> score (608)
```
5,897 parameters. For Rb1 the edge's 48 numbers go through those three layers and come out as **logit +19.826** (the last bias is +21.33; it is added to
every edge, so only the remaining -1.50 distinguishes Rb1 from other moves).

### 3d. From scores to a move (`plays.py: model_move`, `visualize_move.py` section 8)
```
 scores of the 608 edges -> keep only the 37 legal ones (others = -inf) -> argmax        = the move played
 softmax over those 37 (only for display / for search) -> probabilities
```
Demo position: top move h3 (logit 25.89, 30.4%), then Bg5 (28.5%), a3 (9.2%) ...; **Rb1 is ranked 21st of 37 with 0.07%**.
Softmax never changes the ordering, so plain argmax of the logits gives the same move.

---------------------------------------------------------------------------------------------------

## 4. All 11 heads and what they are trained to predict

All are trained simultaneously; total loss = plain sum of the 11 losses (`model.py: compute_loss`).

| head | reads | output per position | target (from the game) | loss |
|------|-------|---------------------|------------------------|------|
| next_move | edge | 1 per edge | the edge of the move actually played | softmax cross-entropy over ALL edges of the position |
| legal_moves | edge | 1 per edge | 1 if the move is legal | BCE averaged per position |
| legal_captures | edge | 1 per edge | 1 if legal AND a capture | BCE averaged per position |
| next_piece_moved | node | 1 per square | piece that started on this square moves within the next 10 plies | BCE |
| next_piece_captured | node | 1 per square | piece that started here is captured within the next 10 plies | BCE |
| checking_in_1 | global | 1 | side to move gives check next move | BCE |
| checking_in_3 | global | 1 | side to move gives check within its next 3 moves | BCE |
| checked_in_1 | global | 1 | opponent gives check on the reply | BCE |
| checked_in_3 | global | 1 | opponent gives check within its next 3 moves | BCE |
| plies_till_end | global | 1 | `exp(-0.2*(plies_left-1))` | MAE of `sigmoid(output)` |
| final_result | global | 3 | white win / draw / black win of the real game | cross-entropy |

Notes: `legal_moves` is an easy head because the legal bit is already an input to the edge stream. In training the `next_move` softmax runs over all edges
(legal or not); only at play time are illegal edges masked out.

---------------------------------------------------------------------------------------------------

## 5. How the training data is made (`preprocess.py`, `ours_gpu_extras/labels.py`)

1. Read `games.pgn`, keep games with a result (`1-0`, `1/2-1/2`, `0-1`) and at least 2 moves.
2. Randomly keep a fraction of plies; early plies are kept much less (`keep_prob(ply) = rate * (1 - max(0, 0.98 - (0.075*ply)^2))`).
3. For each kept position store: board array, ep square, castle bits, side to move, halfmove clock, repetition count, the move played (from,to),
   the full list of legal move codes, bitmasks of pieces that move / get captured in the next 10 plies (pieces are tracked from their *original* square even after moving),
   check flags, plies remaining, game result.
4. Save in shards of 2000 games (`data/shard_*.npz`, 189 here). The last 2 shards are the validation set.

## 6. How it trains (`train.py`, `ours_gpu_extras/training_utils.py`)
- Whole dataset sits on the GPU. Per step: 256 positions -> `GraphBuilder` merges them into one graph -> forward -> sum of 11 losses -> backward.
- Optimizer NAdam, lr 5e-4, linear warm-up (<=500 steps), cosine decay down to 5% of lr, gradient-norm clip 5.0, 3 epochs, validation every 2000 steps.

## 7. How it plays
- `plays.py`: pure policy, argmax of legal-edge logits.
- `search.py` (`--search`): for each position the net gives (a) move probabilities (softmax over legal edges) and (b) a value
  `P(white win) - P(black win)` from the `final_result` head. A tree is built to depth 4, branches with path probability < 0.005 are pruned, values are backed up
  with minimax, and the root move maximises `value + 2.0 * policy_prob` (sign-adjusted for the side to move). Checkmate/stalemate/50-move are scored exactly.
- `play.py`: games versus Stockfish with limited Elo to estimate the model's strength.

---------------------------------------------------------------------------------------------------

## 8. Things worth knowing / possible weaknesses (observations, not bugs)
- The graph is purely geometric and blocker-blind; legality comes only from the single input bit.
- Black pieces are not mirrored; the model must learn both colours' pawn directions from `is_white/is_black` and the turn bit.
- Block 10's node and global outputs are only used by the node/global heads; `next_move` reads the edge vector, which in block 10 is built from block 9's node/global values.
- `ours_gpu_extras/{graph_builder,targets_and_loss,training_utils,search}.py` add `..\ours_gpu\gnn-chess` to `sys.path`. That folder does not exist,
  so the imports work only because the scripts are run from `ours_gpu/` (which is already first on the path). Harmless, but stale.
- Not reviewed: the original Keras/TensorFlow implementation (`chessgnn/`, `notebooks/`, `ours/`) and `ours_gpuaa/` (tournament / Elo scripts).

---------------------------------------------------------------------------------------------------

## 9. Backpropagation - how the weights actually get updated (the backward flow)

Section 3 was the **forward** pass: numbers go in, one loss number comes out. Training asks one question about that loss number:
*"If I nudge this particular weight up a tiny bit, does the loss go up or down, and by how much?"* That answer is the **gradient** of the loss with respect to
that weight, written `dL/dw`. Backpropagation is the efficient way to get that answer for all 269,925 parameters at once: walk the forward pass **in reverse**,
and at every operation use the chain rule to turn "how the loss depends on my output" into "how the loss depends on my input and on my weights".

In code this is the line `loss.backward()` in `training_utils.train_step`. PyTorch recorded every operation during the forward pass
(each tensor remembers which operation made it), so `backward()` just replays that record backwards. Nothing in `model.py` has hand-written backward code.

### 9.1 The chain rule on one neuron (real numbers from the Rb1 run)

Take the very first neuron on square a1 (`inp.v1`, neuron 0). Forward:
```
pre = w·x + b = 1.0*0.1171 + 1.0*0.3741 + (-1.05)*(-0.2194) + (-1.05)*0.1686 + (-0.0302) = +0.5143
y   = leaky(pre) = +0.5143            (pre >= 0, so leaky passes it through unchanged)
```
Suppose the layers above tell us `dL/dy = g` (the loss would change by `g` per unit increase of this neuron's output). Backward, step by step:
```
dL/dpre  = dL/dy * leaky'(pre) = g * 1.0          (leaky' is 1 if pre >= 0, else 0.3)
dL/db    = dL/dpre            = g
dL/dw_i  = dL/dpre * x_i      -> rook: g*1.0,  is_white: g*1.0,  rank: g*(-1.05),  file: g*(-1.05),  all zero-features: 0
dL/dx_i  = dL/dpre * w_i      -> passed further down (here x is the raw input, so it stops; no one trains the input)
```
So **a weight's gradient = (error signal arriving at its neuron) x (the input that weight multiplied)**. A weight attached to an input that was 0 gets no gradient
from this square (that is why the features that are 0 for a1, like "knight", leave their weights untouched by this example). Chain rule = multiply these local factors along the path from the loss down to the weight.

### 9.2 The backward rule of every operation used in this model

(`g` = the gradient arriving from above, i.e. `dL/d(output)`.)

| forward operation | where | backward: gradient sent to the inputs / parameters |
|---|---|---|
| `y = W x + b` (Linear) | everywhere | `dW = g xᵀ`, `db = g`, `dx = Wᵀ g`. For a batch of N rows (X is N x in, G is N x out): `dW = Gᵀ X` (out x in), `db = G.sum(rows)`, `dX = G W` |
| `leaky(x)` | after most Linears | `g * (1 if x >= 0 else 0.3)` |
| `LayerNorm` `y = γ·x̂ + β` | end of every stage | `dβ = Σ g`, `dγ = Σ g·x̂`, and `dx = (1/σ)·( γg − mean(γg) − x̂·mean(γg·x̂) )` (the mean/std of the vector also depend on x, which is why the extra terms appear) |
| residual add `v = LN(v_prev + v_gat)` | every block | the gradient reaching the sum is copied **unchanged** to both `v_prev` and `v_gat` |
| concat `[a \| b \| c]` | `concat_*` in layers.py | cut the gradient into the same pieces; each piece goes back to where it came from |
| gather `x.index_select(0, idx)` | putting node/global vectors on edges | **scatter-add**: every edge adds its gradient into the row it copied. A node used by 16 edges receives the sum of 16 gradients |
| scatter-sum `index_add_` | GAT weighted sum | **gather**: each edge receives the gradient of the receiver it was summed into |
| scatter-mean | edges -> nodes | gather, divided by the number of edges in that group (`g[recv] / count`) |
| global mean pool over 64 squares | nodes -> global | every one of the 64 squares receives `g / 64` |
| `att * val` (elementwise product) | GAT | `d att = g · val`, `d val = g · att` (product rule) |
| grouped softmax `α = softmax(score)` per receiver and head | GAT | `d score_k = α_k · ( dα_k − Σ_j α_j dα_j )`, sums taken over the neighbours of the same receiver |
| `view` / `flatten` / `squeeze` | reshapes | reshape the gradient back |

Gather and scatter being each other's backward is the key link to `layers.py`: **the forward path that sends information along an edge is the same path that carries
the gradient back along that edge**, just in the opposite direction.

### 9.3 Where the backward pass starts: the gradient of each loss

`compute_loss` returns `total = sum of 11 losses`. Because it's a plain sum, the derivative of `total` with respect to each individual loss is exactly 1,
so every head starts its own backward pass with a clean, unscaled gradient. The first gradient is on the head outputs (the logits):

| loss | gradient on each output logit `z` |
|---|---|
| `next_move` (softmax cross-entropy over a position's edges) | `p_k - 1[k is the played edge]`, where `p` is the softmax over **all** edges of the position, divided by the number of positions. The played edge is pulled **up** by `1 - p`; every other edge is pushed **down** by its own `p_k` |
| binary cross-entropy (`legal_*`, `next_piece_*`, check heads) | `sigmoid(z) - target`, divided by the number of elements averaged |
| `final_result` (3-way cross-entropy) | `softmax(z) - one_hot(result)`, divided by the number of positions |
| `plies_till_end` (MAE of sigmoid) | `sign(sigmoid(z) - target) * sigmoid(z)*(1 - sigmoid(z))`, divided by the count |

Concrete example for `next_move`: if the game move in this position was h2h3, the edge h2->h3 gets gradient `p - 1` (negative: raise its score), and the edge for Rb1
(a1->b1) gets `+p_Rb1` (positive: lower its score). Edges that already have almost zero probability (most of the 608) get almost no gradient. That is exactly how a
confident wrong answer gets corrected hard and a near-correct answer gets nudged only a little.

### 9.4 The full backward flow through the model

The backward pass is the forward diagram read from the bottom up:

```
 loss = L_next_move + L_legal_moves + L_legal_captures + L_piece_moved + L_piece_captured + L_checks(4) + L_plies + L_result
   │ (each term has derivative 1)
   ▼
 11 heads:   MLPHead c ← b ← a   (gradients for their own weights; each head only receives gradient from its own loss)
   │
   ├── edge heads (next_move, legal_moves, legal_captures)   → dL/de   (summed if several heads feed the same e)
   ├── node heads (next_piece_moved, next_piece_captured)    → dL/dv
   └── global heads (checks, plies, result)                  → dL/du
   ▼
 Block 10 (reverse order of its forward steps):
   LN_v, LN_e, LN_u  backward         ← gradient on the block outputs v, e, u
   residual:  gradient copied to  v_prev/e_prev/u_prev  AND into the new parts v_gat/e_new/u_new
   u_new = mean(v_gat)                → every node gets dL/du_new / 64
   v_gat = leaky(GAT(v_mean)+bias)    → GAT backward: weighted-sum → (dα, dval) → softmax backward → att2 → att1 → proj
                                        → gradient for g = [v_mean[recv] | v_mean[send]] → scatter-add back to nodes (v_mean)
   v_mean = mean of e_new over receivers → gather onto edges (÷count): joins the gradient already on e_new
   e_new = leaky(e2([u | e1]))        → e2 backward: weights; gradient for u (scatter-add over the edges of each position) and for e1
   e1 = leaky(Linear144→32 [e_prev | v_prev[recv] | v_prev[send]])
                                      → e1 backward: weights; the 144-gradient is cut into 3 pieces:
                                        e_prev, and scatter-added into the node rows v_prev[from] and v_prev[to]
   ▼
 Block 9 receives gradient on (v, e, u) that came from block 10's v_prev, e_prev, u_prev  … and so on down to block 1
   ▼
 Input block: LN → Linear layers (v1, v2, e0, e1, u0, u1) get their weight gradients.
   The gradient reaching the raw features (node features, 8 globals, the legal bit) is just discarded - they're inputs, not parameters.
```

Three things the diagram shows:
1. **The residual connections make a highway for gradients.** Because a residual add copies the gradient unchanged to `x_prev`, the gradient reaching block 1 is
   (identity path through all 10 blocks) + (paths through each block's layers). Without it, a gradient passing through 10 blocks x ~5 layers could vanish or explode; with it, early blocks still get a strong signal.
2. **A single move's loss touches a wide neighbourhood.** For `next_move` of the edge a1->b1, block 10's `e` gets the gradient; `e1`'s backward
   sends part of it to the node rows a1 and b1 of block 9; those nodes' gradients go to every edge that touched them via the previous block's mean and attention, and so on.
   After 10 blocks the loss on one edge reaches the weights through squares up to ~10 hops away (in the piece-mobility graph).
3. **Edges are the hub in the backward pass as well.** Node and global gradients enter the edge computations of the previous block only through the edge update (`cat1` and `cat2`).

### 9.5 Weight sharing: gradients are summed over every place the weight was used

A weight matrix like `inp.v1` is applied to all 64 squares of all 256 positions of a batch (16,384 rows). Its single gradient is the **sum over all those rows**
(`dW = Gᵀ X` adds one outer product per row). The same for `blocks.k.e1`: summed over all ~150,000 edges in the batch. So a weight "hears" from every square or edge that used it,
and the update is the average-direction compromise across all of them. Each of the 10 blocks has its own separate matrices, so sharing happens only across squares/edges/positions, not across blocks.

### 9.6 Which parameters get gradient from which loss (verified by running it)

I ran one backward pass on 64 validation positions with the checkpoint in this folder:
- Backpropagating **only** `next_move` gives a gradient of `None` (no gradient at all) to the other 10 heads, and to `blocks.9.gat.*` and `blocks.9.ln_v` / `ln_u`.
  Reason: `next_move` reads `e` of block 10, and block 10's `e` is computed from block 9's `v` and `u`; block 10's own node and global outputs are used only by the node/global heads.
- The trunk (input block and blocks 1-9) gets gradient from **all 11** losses added together. This is the multi-task learning: the check/result/capture heads shape the same shared weights that the move head uses.
- Gradient norms with all 11 losses on (a rough picture of how much each part is pushed): `inp.v1.weight` 1.10, block 1 `e1.weight` 1.58, block 10 `e1.weight` 0.89, block 10 `gat.proj` 0.76.

### 9.7 After the gradients exist: clip, then update (`train_step`)

```
optimizer.zero_grad(set_to_none=True)          # old gradients would otherwise be ADDED to the new ones
loss.backward()                                 # section 9.1-9.6: fills p.grad for every parameter
clip_grad_norm_(model.parameters(), 5.0)        # measure the length of the whole gradient vector; if > 5, scale all of it down
optimizer.step()                                # NAdam
scheduler.step()                                # learning rate for the next step
```
In my test batch the total gradient norm was **13.8**, so it would be scaled by 5/13.8 ≈ 0.36. Clipping preserves the *direction* and caps the *size*.

**NAdam update (per weight `w` with gradient `g`, step `t`)**, simplified form:
```
m  = 0.9   * m + 0.1   * g              # running average of the gradient (momentum)
v  = 0.999 * v + 0.001 * g²             # running average of the squared gradient (how noisy / large this weight's gradients are)
m̂ = Nesterov look-ahead blend of m and the current g, bias-corrected
v̂ = v / (1 - 0.999^t)                   # bias correction
w  = w - lr * m̂ / ( sqrt(v̂) + 1e-8 )
```
(PyTorch's NAdam additionally uses a slowly changing momentum schedule, `momentum_decay = 0.004`; the shape of the rule is the same. No weight decay is used.)
What this means in practice:
- A weight moves in the direction opposite to its (smoothed) gradient: gradient positive -> the weight goes down, negative -> up.
- Dividing by `sqrt(v̂)` makes the step size roughly **`lr` per step for every weight, no matter how big or small its gradient is**. A weight with a tiny but consistent gradient moves as fast as one with a huge gradient. At the peak learning rate 5e-4 a weight moves at most about 5e-4 per step; the weights are typically around ±0.2, so changing a weight significantly takes hundreds to thousands of steps.
- `lr` is the base 5e-4 times the schedule from section 6 (warm-up for 500 steps, then cosine decay down to 5%).

One full training step is therefore: forward (section 3) -> loss -> **backward (this section, reverse order)** -> clip -> NAdam update. Repeat 55,836 times for 3 epochs.

### 9.8 Memory side effect
To run backward, autograd must keep the forward-pass intermediates (every `(E x 144)` concatenation, every `(E x 48)` activation, every attention tensor of all 10 blocks) in GPU memory until `backward()` finishes. That is why the batch size is limited by the edge tensors (~150,000 edges x 10 blocks) rather than by the 270k parameters themselves. `evaluate()` runs under `no_grad()`, which skips this bookkeeping.

### 9.9 A curious detail the gradient explains: the +21.33 bias on the move head
In section 3c, the last layer of the `next_move` head had a bias of **+21.33**, added to every edge. Softmax is shift-invariant (adding the same number to every score of a position changes nothing),
so the correct gradient for that bias is exactly zero: for each position the `next_move` gradients `p_k - 1[target]` sum to 0. I verified it: the gradient on that bias was 2.8e-8 on a real batch, and the per-position sums were ~3e-9, i.e. just floating-point rounding.
Why did the bias nevertheless reach 21? Most likely this: NAdam divides by `sqrt(v̂)`, so even a gradient that is pure rounding noise gets scaled up to steps of roughly `lr` size, and if that noise has any consistent sign it makes the bias drift slowly in one direction
(this checkpoint was trained for 312,117 steps, far more than 55,836). It is harmless for move choice, because the shift cancels out, but it is a good illustration of what Adam-type optimizers do with tiny gradients. (This explanation is my inference from the measurements, not something I tested directly.)
