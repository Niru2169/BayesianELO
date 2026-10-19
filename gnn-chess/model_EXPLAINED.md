# model.py explained

`model.py` defines the whole neural network (`ChessGNN`), the 11 output heads, and the loss that trains them.
It is the only file with learned weights (269,925 numbers). It sits in the middle of the pipeline:

```
 board ──> graph_builder.py (features.py, graph_utils.py) ──> graph dict + legal_in bit
                                                                    │
                                                                    ▼
                                     model.py  ChessGNN.forward(graph, legal_in) ──> dict of 11 outputs
                                                                    │
                       train.py: compute_loss(outputs, targets, graph) ──> one number ──> backprop
                       plays.py / search.py: read outputs['next_move'] and outputs['final_result']
```

---------------------------------------------------------------------------------------------------

## 1. Imports and who they come from

| import | from | what it gives the model |
|--------|------|--------------------------|
| `GLOBAL_FEATURES` (8), `NODE_FEATURES` (12) | `features.py` | input widths |
| `concat_*`, `global_average_pooling`, `scatter_and_aggregate`, `scatter_and_softmax` | `layers.py` | all graph plumbing (see `layers_EXPLAINED.md`) |
| `_binary_crossentropy`, `_binary_crossentropy_per_position`, `_categorical_crossentropy`, `_sigmoid_mae`, `_softmax_over_edges` | `../ours_gpu_extras/targets_and_loss.py` | the five loss formulas (the `sys.path.append` line makes that folder importable) |

## 2. Constants (the "hyper-parameters" of the architecture)

| name | value | where it is used |
|------|-------|------------------|
| `EMBED_DIM` | 48 | width of every node / edge / global vector, and of LayerNorm |
| `N_BLOCKS` | 10 | number of `FullBlock`s |
| `LN_EPS` | 1e-3 | LayerNorm epsilon (Keras-style; PyTorch default would be 1e-5) |
| `LEAKY_SLOPE` | 0.3 | leaky ReLU slope (Keras-style default; PyTorch default is 0.01) |
| `EDGE_HIDDEN` | 32 | width of the first edge-update layer |
| `GAT_HIDDEN` | 48 | width of the hidden layer inside the attention-score MLP |
| `GAT_HEADS` | 24 | number of attention heads |
| `GAT_HEAD_OUT` | 2 | numbers each head outputs (24 x 2 = 48 = `EMBED_DIM`) |
| `HEAD_HIDDEN_1` / `HEAD_HIDDEN_2` | 72 / 32 | hidden widths of each output head MLP |

## 3. Tiny helpers

- `leaky(x)` = `leaky_relu(x, 0.3)`: `x` if `x >= 0` else `0.3*x`. The only nonlinearity in the network (besides the softmax in attention).
- `layer_norm()` = `LayerNorm(48, eps=1e-3)`: for each vector separately, subtract its mean over the 48 numbers, divide by its std, then
  multiply by a learned per-number scale (init 1) and add a learned shift (init 0). 96 parameters. Keeps every vector at a stable size
  so 10 stacked residual blocks don't blow up.

---------------------------------------------------------------------------------------------------

## 4. `MultiHeadGAT` - attention over each square's neighbours (10,488 parameters)

A graph attention layer. "For each from-square, look at all squares it is connected to; decide how much each matters (24 different
ways at once); take a weighted mix."

Parameters: `att1` Linear(96→48) · `att2` Linear(48→24) · `proj` Linear(96→48, **no bias**) · `bias` (48 free numbers).

`forward(v, receivers, senders)` where `v` is the node tensor `(V,48)` (in practice `v_mean`, see FullBlock):
```
e   = [ v[receiver] | v[sender] ]                      (E, 96)     concat_edges_ends_only
score = leaky( att2( leaky( att1(e) ) ) )              (E, 24)     small MLP: how interesting is this neighbour, per head
alpha = scatter_and_softmax(score, receivers)          (E, 24)     for each receiver and head, weights over its neighbours sum to 1
val   = proj(e).view(E, 24, 2)                         (E, 24, 2)  what each neighbour says, per head (2 numbers per head)
out   = scatter-sum over receivers of alpha[...,None] * val   (V, 24, 2)
return leaky( out.flatten(1) + bias )                  (V, 48)
```
Notes
- Query and key are the same concatenation `[receiver | sender]`, so the score depends on *both* squares (it is "additive attention" via a small MLP, not a dot product).
- The value also contains the receiver's own half, so a square also "hears" itself in every message (and it always has a self-loop neighbour).
- The sum over incoming edges gives each of the 64 squares a 48-number result. 24 heads x 2 = 48.
- Attention is **local**: only the squares connected by the piece-mobility graph. Information from further away arrives after multiple blocks.

---------------------------------------------------------------------------------------------------

## 5. `InputBlock` - raw features -> first embeddings (11,952 parameters)

Parameters: `v1` 12→48, `v2` 48→48, `e0` 12→48, `e1` 97→48, `u0` 12→48, `u1` 56→48, plus three LayerNorms.

`forward(graph, legal_moves)`; inputs: `node_features (V,12)`, `global_features (P,8)`, `receiver/sender`, `legal_moves (E,)` (the 0/1 legal bit).

```
NODE   v = LN_v( leaky( v2( leaky( v1(node_features) ) ) ) )                        (V,48)

EDGE   e = leaky( e0(node_features) )                                              (V,48)   embed every square
           = [ e[receiver] | e[sender] | legal_bit ]                               (E,97)
           = LN_e( leaky( e1(.) ) )                                                (E,48)

GLOBAL u = leaky( u0(node_features) )                                              (V,48)
           = mean over the 64 squares of each position                             (P,48)
           = [ pooled | 8 global features ]                                        (P,56)
           = LN_u( leaky( u1(.) ) )                                                (P,48)
```
Key points
- The edge has **no features of its own**. Its whole input is "my two end squares + am I legal". `e0` is a separate square embedding from `v1` (different weights), so the edge branch learns its own view of squares.
- The global branch does not see the board layout, only the average of 64 per-square embeddings plus the 8 hand-made globals (turn, castling, clocks).
- The legal bit enters the network *only here*, in the edge branch.
- Returns the triple `(v, e, u)`.

---------------------------------------------------------------------------------------------------

## 6. `FullBlock` - the repeated message-passing block (19,304 parameters each)

Parameters: `e1` Linear(144→32), `e2` Linear(80→48), `gat` (MultiHeadGAT), and 3 LayerNorms.

`forward(v_prev, e_prev, u_prev, graph)`:

```
 (1) edge update  — nodes and global flow INTO the edge
     cat1  = [ e_prev | v_prev[receiver] | v_prev[sender] ]       (E,144)
     e1    = leaky( Linear144→32(cat1) )                          (E,32)
     cat2  = [ u_prev[position of edge] | e1 ]                    (E,80)
     e_new = leaky( Linear80→48(cat2) )                           (E,48)

 (2) edge → node  — edges flow INTO the node
     v_mean = mean of e_new over all edges whose receiver is this square          (V,48)
     v_gat  = MultiHeadGAT(v_mean, receivers, senders)                            (V,48)

 (3) node → global  — nodes flow INTO the global
     u_new = mean over the 64 squares of v_gat                                    (P,48)

 (4) residual connections + LayerNorm, per stream
     v = LN_v( v_prev + v_gat )     e = LN_e( e_prev + e_new )     u = LN_u( u_prev + u_new )
```
Observations
- The **edge is the hub**: it is the only place where node, edge and global information are combined in one vector (step 1).
  The node never reads `u` directly, and the global never reads edges directly; they're connected through the edge/the mean of the edges.
- Nodes are updated **only from `v_mean`**, not from `v_prev` directly; `v_prev` reaches the new node vector only via the residual `v_prev + v_gat`
  and via step 1 (it appears inside the edge input).
- The residual ("`prev +`") makes each block a refinement of the previous, and is what lets 10 blocks train stably.
- The block uses **different weights in each of the 10 copies** (`ModuleList` of 10 independent `FullBlock`s).
- `u_prev` is never used by the global update except in the residual and in step 1.

---------------------------------------------------------------------------------------------------

## 7. `MLPHead` - the output heads (5,897 parameters, or 5,963 for 3 outputs)

```
x (48) → Linear 48→72 → leaky → Linear 72→32 → leaky → Linear 32→out
```
`out = 1` for ten heads, `out = 3` for `final_result`. The final layer has no activation: the output is a raw **logit**.
(Sigmoid / softmax is applied inside the loss functions, or later by `search.py`.)

## 8. `LOSSES` - which head uses which loss

```python
'next_piece_moved'   binary_crossentropy              'checked_in_1'   binary_crossentropy
'next_piece_captured'binary_crossentropy              'checked_in_3'   binary_crossentropy
'legal_moves'        binary_crossentropy_per_position 'checking_in_1'  binary_crossentropy
'legal_captures'     binary_crossentropy_per_position 'checking_in_3'  binary_crossentropy
'plies_till_end'     sigmoid_mae                      'final_result'   categorical_crossentropy
'next_move'          softmax_over_edges
```
This dict does double duty: `ChessGNN.__init__` builds one `MLPHead` per key (so the key is the head's name), and `compute_loss` loops the same keys.

## 9. `ChessGNN` - the whole network

```python
self.inp    = InputBlock()
self.blocks = ModuleList([FullBlock() for _ in range(10)])
self.heads  = ModuleDict({name: MLPHead(3 if name=='final_result' else 1) for name in LOSSES})
```
`forward(graph, legal_moves)`:
```
v, e, u = inp(graph, legal_moves)
for block in blocks: v, e, u = block(v, e, u, graph)       # 10 times
outputs = {
  'next_piece_moved'   : head(v)  → (V,)      # one per square
  'next_piece_captured': head(v)  → (V,)
  'next_move'          : head(e)  → (E,)      # one per edge  ← the one used to choose a move
  'legal_moves'        : head(e)  → (E,)
  'legal_captures'     : head(e)  → (E,)
  'checked_in_1' / 'checked_in_3' / 'checking_in_1' / 'checking_in_3' / 'plies_till_end': head(u) → (P,)
  'final_result'       : head(u)  → (P,3)
}
```
`.squeeze(-1)` removes the trailing "1" so each output is a flat vector.

Shapes for a training batch of P = 256 positions: V = 16,384 nodes, E ≈ 150,000 edges (about 600 per position; 608 in the demo position), P = 256 globals.
All 11 heads share the **same trunk** (input block + 10 blocks); only the last small MLP differs. This is "multi-task learning": the extra heads (checks, captures,
game result …) force the trunk to learn chess concepts that also help pick the move.

Subtle consequence: `next_move` reads `e` from block 10, which was built from block 9's `v` and `u`. Block 10's own `v` and `u` outputs feed only the node and global heads.

## 10. `compute_loss(outputs, targets, graph)`

```python
for head, kind in LOSSES.items():
    losses[head] = LOSS_FUNCTIONS[kind](head, outputs, targets, graph)
total = sum(losses.values())
return total, losses
```
`total` is the **plain unweighted sum** of 11 scalar losses; that's what `loss.backward()` differentiates. `losses` is returned for logging (`train.py` prints `losses['next_move']`).
`targets` is the dict built by `make_targets` (see `train_EXPLAINED.md`), `graph` supplies `edge_position`, `n_positions`, `edges_per_position`.

The five loss kinds (from `targets_and_loss.py`):

| kind | formula | applied to |
|------|---------|-----------|
| `binary_crossentropy` | `BCEwithLogits(output, target)`, mean over every element | per-square flags (`next_piece_*`, V values) and per-position flags (checks, P values) |
| `binary_crossentropy_per_position` | BCE per edge, **sum** over the position's edges, divide by that position's edge count, mean over positions | `legal_moves`, `legal_captures` (stops positions with many edges from dominating) |
| `sigmoid_mae` | `mean(| sigmoid(output) − target |)` | `plies_till_end` (target is `exp(−0.2·(plies_left−1))`, in (0,1]) |
| `categorical_crossentropy` | `cross_entropy(logits (P,3), class 0/1/2)` | `final_result` (0 white win, 1 draw, 2 black win) |
| `softmax_over_edges` | for each position: `logsumexp(scores of ALL its edges) − score[played-move edge]`, averaged over positions whose played edge exists in the graph | `next_move` |

`softmax_over_edges` is the key one: a softmax-cross-entropy "classifier" whose classes are the edges of that position. Illegal edges are included in the
denominator during training; they get pushed down mostly because the legal bit is an input. At play time the illegal edges are simply removed (`-inf`).

---------------------------------------------------------------------------------------------------

## 11. Parameter count summary

| part | parameters |
|------|-----------|
| InputBlock | 11,952 |
| 10 x FullBlock (e1 4,640 + e2 3,888 + GAT 10,488 + 3 LN 288) | 193,040 |
| 11 heads (10 x 5,897 + 1 x 5,963) | 64,933 |
| **total** | **269,925** |

Initialisation is PyTorch's default (`nn.Linear`: Kaiming-uniform; LayerNorm: scale 1, shift 0; GAT `bias`: 0). There is no custom init code.

## 12. Where the data comes from and goes to (one training step, with this file highlighted)

```
data.batch(indices) ─> graph_builder.build ─> graph {node_features, global_features, receiver, sender, edge_position, edge_key, ...}
                                         └─> make_targets ─> targets, legal_in
                                              │
                           ┌──────────────────┘
                           ▼
                  [model.py] ChessGNN(graph, legal_in)  ─ InputBlock ─ 10 x FullBlock ─ 11 x MLPHead ─> outputs
                  [model.py] compute_loss(outputs, targets, graph) ─> total loss
                           ▼
                  train.py: backward → clip → NAdam step → LR schedule step
```
