# layers.py explained

`layers.py` has **no learned weights**. It is the toolbox of "graph plumbing" operations that `model.py` (and the loss code and
the search code) use to move numbers between the three streams: nodes (squares), edges (moves) and globals (whole position).
Everything here is pure tensor arithmetic: gather, scatter, average, softmax.

If you understand this file, `model.py` becomes a short list of "plumbing call, then Linear layer, repeat".

---------------------------------------------------------------------------------------------------

## 0. The data layout these functions assume

Many positions are merged into ONE big graph (a "disjoint batch"). For a batch of `P` positions:

| tensor | shape | meaning |
|--------|-------|---------|
| node tensor `v` | `(P*64, F)` | row `position*64 + square` is that square |
| edge tensor `e` | `(E, F)` | one row per directed edge, E = total edges of all positions |
| global tensor `u` | `(P, F)` | one row per position |
| `receivers` | `(E,)` int | node row index of the edge's **receiver** (= the move's from-square) |
| `senders` | `(E,)` int | node row index of the edge's **sender** (= the move's to-square) |
| `edge_position` | `(E,)` int | which position each edge belongs to (0..P-1) |

(`graph_builder.py` creates `receivers`, `senders`, `edge_position`; see `graph_utils.board_to_coo`.)
Because node indices include `position*64`, edges of one position can never point to nodes of another position.
So one giant tensor op processes the whole batch at once, and the positions never mix.

**Direction convention**: for the move a1→b1, `receiver = a1`, `sender = b1`. So "aggregate over the edges whose receiver is
a1" means "collect everything about every square a1 is connected to". The attention in `model.py` is therefore computed
*per from-square*, over its possible destination squares.

---------------------------------------------------------------------------------------------------

## 1. Scatter aggregators (many rows -> one row per group)

Common signature: `values (N, ...)`, `group_index (N,)`, `n_groups`. Row `i` of `values` belongs to group `group_index[i]`.
The result has shape `(n_groups, ...)`.

Toy example used below (verified by running the code):
```
values      = [[1,2],[3,4],[5,6],[7,8]]
group_index = [0,   1,    0,    0  ]        n_groups = 3   (group 2 gets no rows)
```

### `_scatter_sum`
```python
values.new_zeros((n_groups,) + values.shape[1:]).index_add_(0, group_index, values)
```
Make a zero tensor with one row per group, then add each value row into the row named by its group.
`sum -> [[13,16],[3,4],[0,0]]` (group 0 = rows 0,2,3 = 1+5+7, 2+6+8).

### `_scatter_mean`
Sum, divided by the count of rows in each group. Count is computed by scattering a column of ones, with `clamp(min=1)` so an empty group
gives `0/1 = 0` instead of NaN. `mean -> [[4.33,5.33],[3,4],[0,0]]`.
This is what `FullBlock` uses to turn edge vectors into node vectors ("average of all edges pointing into this square").

### `_scatter_max`
Start with `-inf` everywhere and `scatter_reduce_(..., 'amax', include_self=True)`. `group_index` is reshaped/expanded to the shape
of `values` because `scatter_reduce_` needs an index of the same shape. `max -> [[7,8],[3,4],[-inf,-inf]]`.
Empty groups stay `-inf`. Used for the numerically-stable softmax and by the `next_move` loss.

### `AGGREGATORS` / `scatter_and_aggregate(values, group_index, n_groups, agg_method)`
A dictionary lookup: `'sum' | 'mean' | 'max'` -> the function above. Every caller goes through this one entry point.

Where it is used:
| call | file | purpose |
|------|------|---------|
| `'mean'` over `receivers` | `model.py FullBlock` | edge vectors -> node vectors |
| `'sum'` over `receivers` | `model.py MultiHeadGAT` | attention-weighted sum of neighbours |
| `'sum'` over `edge_position` | `targets_and_loss.py` | per-position BCE sum for `legal_moves/legal_captures` |
| `'max'`,`'sum'` over `edge_position` | `targets_and_loss.py` | log-sum-exp for the `next_move` loss |

---------------------------------------------------------------------------------------------------

## 2. `scatter_and_softmax(scores, group_index, n_groups)`

A softmax where the "set of things competing" is a **group of rows**, not a row of columns. Used for attention:
all the edges that share one receiver compete for 100% of the attention.

```python
shifted = exp(scores - max_of_group[group_index])      # subtract each group's max (stability)
return shifted / sum_of_group[group_index]             # divide by each group's total
```
`_scatter_max(...).index_select(0, group_index)` copies each group's max back onto every row of the group (a "gather").

Toy check: rows 0,2,3 are group 0 with scores 1,3,5, row 1 is alone in group 1:
`softmax -> [0.016, 1.000, 0.117, 0.867]`. Rows 0,2,3 sum to 1.0; the lone row in group 1 gets exactly 1.0.

If `scores` has several columns (the attention has 24, one per head) each column is its own independent softmax, because every operation
is applied per column. In `model.py`, `scores` is `(E, 24)` so 24 independent attention distributions are computed at once.

Also used by `search.py`: the move probabilities are a softmax over the legal moves of each board, grouped by `board_of_move`.

---------------------------------------------------------------------------------------------------

## 3. Concatenation helpers (gather vectors onto edges)

All use `index_select(0, idx)`, i.e. "for every edge pick the row of the node (or global) it refers to".

### `concat_edges_ends_only(node_feats, receivers, senders)`  -> `(E, 2F)`
`[ node[receiver] | node[sender] ]`. Toy: for the edge 0->1 with node rows `[0,1]` and `[2,3]` the result is `[0,1,2,3]`.
Used by: the input block (edge branch), the GAT attention input, the GAT value projection.

### `concat_edge_with_ends(node_feats, edge_feats, receivers, senders)` -> `(E, F_e + 2F)`
`[ edge | node[receiver] | node[sender] ]`. In the model: 48 + 48 + 48 = **144** numbers.
Used at the start of every `FullBlock` edge update. This is the moment where edge, from-square and to-square information meet.

### `concat_globals_with_anything(global_feats, items, position_of_item)` -> `(N, F_u + F_items)`
`[ global[position of this item] | item ]`. Used with `items = e1` (edges, 32 wide) so the result is 48 + 32 = **80**.
Each edge receives a copy of *its own position's* global vector. (`position_of_item = edge_position`.)
This is how the global vector flows INTO the edge stream.

---------------------------------------------------------------------------------------------------

## 4. `global_average_pooling(node_feats, n_positions)` -> `(P, F)`

```python
node_feats.view(n_positions, 64, F).mean(1)
```
Reshape the flat `(P*64, F)` node tensor into `(P, 64, F)` and average over the 64 squares. This works only because every position has
exactly 64 nodes in a fixed order (that is why nodes exist for empty squares too). It is how information flows from nodes INTO the global vector.
It is called (a) in the input block on `leaky(u0(node_features))`, (b) in every full block on `v_gat`.
Note: it is a plain mean over all 64 squares, empty or not.

---------------------------------------------------------------------------------------------------

## 5. How the pieces connect in one `FullBlock` (what each helper does in the data flow)

```
edge:  concat_edge_with_ends(v, e, recv, send)        (E,144)   <- gather from nodes
       Linear 144->32, leaky
       concat_globals_with_anything(u, e1, edge_pos)  (E,80)    <- gather from globals
       Linear 80->48, leaky                                      = e_new
node:  scatter_and_aggregate(e_new, recv, 'mean')     (V,48)    <- scatter from edges to nodes
       MultiHeadGAT:
          concat_edges_ends_only(v_mean, recv, send)  (E,96)    <- gather from nodes
          Linear -> 24 scores per edge
          scatter_and_softmax(scores, recv)           (E,24)    <- normalise per receiver
          Linear -> values (E,24,2)
          scatter_and_aggregate(att*values, recv, 'sum') (V,24,2)  <- scatter from edges to nodes
global: global_average_pooling(v_gat, P)              (P,48)    <- pool from nodes
```
Pattern: **gather** (index_select) pushes information from nodes/globals onto edges; **scatter** (index_add / scatter_reduce) pulls information
from edges back to nodes; **pooling** pulls nodes to the global. The three streams stay in sync only through these six functions.

---------------------------------------------------------------------------------------------------

## 6. Things worth knowing
- No parameters, no state, no gradients of their own: gradients flow through them (they are differentiable gather/scatter ops) into the Linear layers around them.
- `_scatter_max` is only used for numerical stability / loss computation, not for values that the model learns from.
- `scatter_and_softmax` on `(E, 24)` relies on broadcasting: `max_of_group` and `sum_of_group` are `(V, 24)` and are gathered back to `(E, 24)`.
- Empty groups: mean gives 0, sum gives 0, max gives `-inf`. In a real batch every node has at least a self-loop edge, so no node group is ever empty for the model.
