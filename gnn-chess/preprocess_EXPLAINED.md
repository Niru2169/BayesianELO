# preprocess.py explained

`preprocess.py` is step 1 of the whole pipeline. It turns a giant PGN text file of chess games (`games.pgn`, 2.36 GB) into a few hundred
compact numpy files ("shards") that `train.py` can load straight onto the GPU. Nothing in this file touches the neural network.

```
games.pgn ──[preprocess.py + ours_gpu_extras/labels.py + chessutil.py]──> data/shard_000000.npz … shard_000188.npz
                                                                              │
                                                       ours_gpu_extras/data.py (load_shards, GPUData)  ──> train.py
```

Actual output of the run that produced `ours_gpu/data/`: **189 shards, 4,814,133 positions** (about 25,000 per shard, ~13 positions kept per game, ~28 legal moves per position on average).

Command:
```
python preprocess.py games.pgn --out data --sample-rate 0.1
```

---------------------------------------------------------------------------------------------------

## 1. What one "training example" is

Each saved example is **one chess position taken from a real game** plus the answers (labels) for every head of the model:

| stored field | dtype | meaning | used by head / input |
|--------------|-------|---------|----------------------|
| `board` | uint8 x 64 | piece code per square (0 empty, 1-6 white PNBRQK, 7-12 black) | node features |
| `ep` | int8 | en-passant target square if a legal ep capture exists, else -1 | node feature 11 |
| `castle` | uint8 | 4 bits: WK=1, WQ=2, BK=4, BQ=8 | global features |
| `turn` | uint8 | 1 white to move, 0 black | global features |
| `half` | uint8 | halfmove clock (capped at 255) | global feature |
| `rep` | uint8 | how many times this position occurred earlier in the game (capped 255) | global feature |
| `next_from`, `next_to` | uint8 | the move that was **actually played** in the game | `next_move` target |
| `moved` | int64 bitmask | which pieces move in the next 10 plies | `next_piece_moved` target |
| `captured` | int64 bitmask | which pieces get captured in the next 10 plies | `next_piece_captured` target |
| `chk` | uint8 bit-flags | check events in the near future (table in section 5) | the 4 check heads |
| `ptte` | int16 | plies from this position to the end of the game | `plies_till_end` target |
| `result` | uint8 | 0 = white won, 1 = draw, 2 = black won | `final_result` target |
| `legal_len`, `legal_data` | uint8, uint16 | all legal moves in this position (ragged list, see 6) | `legal_moves`, `legal_captures` targets and the model's legal-bit input |

Remember: the board, ep, castle, turn, half, rep become the **inputs** (`features.py`); everything else is a **label**.

---------------------------------------------------------------------------------------------------

## 2. Constants

| name | value | meaning |
|------|-------|---------|
| `RESULTS` | `{'1-0':0, '1/2-1/2':1, '0-1':2}` | PGN result string -> class id. Games with any other result (`*`, missing) are dropped |
| `GAMES_PER_SHARD` | 2000 | a shard file is written every 2000 games read |
| `OPENING_DROP_MAX` | 0.98 | maximum fraction of positions thrown away at ply 0 |
| `OPENING_DROP_SCALE` | 0.075 | how quickly the opening discount fades with the ply number |

## 3. `keep_prob(ply, sample_rate)` and `sample_plies(n_moves, sample_rate, rng)` - which positions of a game are kept

Almost every game starts with the same opening moves, and keeping all of them would make the data repetitive. So early plies are down-weighted:

```
opening_drop = max(0, 0.98 - (0.075 * ply)^2)
keep_prob    = sample_rate * (1 - opening_drop)
```
`ply` is 0-based: ply 0 = the starting position (white to move), ply 1 = after white's first move, etc. With `sample_rate = 0.1` (verified by running it):

| ply | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 13 | 14 and later |
|-----|---|---|---|---|---|---|----|----|----|--------------|
| keep probability | 0.2% | 0.26% | 0.42% | 1.1% | 2.2% | 3.8% | 5.8% | 8.3% | 9.7% | 10% |

So from ply 14 on, each position is kept with probability 10%; the first ~12 plies are kept much less. `sample_plies` flips an independent weighted coin for each ply `0 … n_moves-1`
(`rng.random() < keep_prob(ply)`) and returns the set of kept plies. The last position of the game (after the final move) is never sampled because there is no move to predict from it.
The random generator is `np.random.default_rng(0)`, so **re-running gives the same shards**.

---------------------------------------------------------------------------------------------------

## 4. `process_game(game, sample_rate, rng)` - one game -> a list of rows

```python
result = RESULTS.get(game.headers.get('Result', ''))       # None => skip
moves  = list(game.mainline_moves())                       # the sequence of Move objects
if n_moves < 2: skip
board  = game.board()                                       # starting position (respects a FEN header)
sampled_plies = sample_plies(...)
move_infos, states = record_game(board, moves, sampled_plies)
return build_rows(move_infos, states, result)
```
Two phases (both in `ours_gpu_extras/labels.py`):

### Phase A: `record_game` - replay the game once, move by move
For every ply `p` with the board *before* move `p`:
1. If `p` is sampled: store a `PositionState` snapshot: `board_array`, `ep_square`, `castle_bits`, `turn`, `halfmove_clock`, `repetition`, `legal_codes` (all legal moves as integer codes `from*64 + to (+4096 if capture)`).
2. **Always** (for every ply, sampled or not): store a `MoveInfo(from, to, captured_square, rook_from, rook_to, gives_check)` for the move about to be played.
   - `captured_square`: the to-square for a normal capture; for en passant it's the square *behind* the target (`to-8` for white, `to+8` for black); -1 if no capture.
   - `rook_from/rook_to`: for castling the rook's squares (kingside: h-file -> next to king; queenside: a-file); -1 otherwise.
   - `gives_check`: `board.gives_check(move)`.
3. `board.push(move)` and update the repetition counter: `seen[pos_key]` counts how many times each position was reached; `repetition` = how many times the *new* position occurred before.
   (`pos_key` = piece bitboards + white-occupancy + castling rights + side to move; en passant and clocks are ignored.)

The move_infos are kept for **every** ply because the labels of a sampled position look ahead up to 10 plies.

### Phase B: `build_rows` - make labels for each sampled position
For each sampled ply `p` (in order), produce one `Row`:
```
next_from, next_to = move_infos[p].from_square, .to_square          # the move actually played
moved, captured    = pieces_moved_and_captured(move_infos, p)       # 10-ply lookahead bitmasks
chk                = check_flags(move_infos, p)
ptte               = n_moves - p                                     # plies left, counting this move
result             = game result
legal              = state.legal
```

---------------------------------------------------------------------------------------------------

## 5. The label tricks in detail

### 5a. `moved` and `captured` - "which piece moves/gets captured in the next 10 plies?"
Goal: a flag **per square of the current position** (64 flags), saying "the piece that is on this square right now will move (or be captured) within the next 10 plies".
The difficulty: a piece may move before it moves "again", so squares change. `pieces_moved_and_captured` follows each piece with a dictionary `current_to_original`
(where is each piece now -> where was it at the start):

```
for ply in start … min(start+10, end):
    original = current_to_original.pop(info.from_square, info.from_square)     # which original piece is moving
    moved.append(original)
    if info.captured_square >= 0:
        captured.append( current_to_original.pop(info.captured_square, info.captured_square) )
    current_to_original[info.to_square] = original                              # it now stands on the to-square
    if castling: the rook is tracked the same way (rook_from -> rook_to) and also appended to moved
```
Example: knight on g1 plays g1→f3 (ply 0), later f3→e5 (ply 4), and is captured on e5 (ply 7).
`moved` gets g1 twice (same bit), `captured` gets g1. The flags are then raised on **g1**, the square where that knight stood at the sampled position.
Both sides' moves count (10 plies = 5 white + 5 black moves, starting with the move at this ply). A piece first appearing from a promotion is tracked from its pawn's origin.
A pawn that is captured *en passant* is credited correctly through `captured_square`.

Each set is turned into one integer by `squares_to_number`: `sum(2**square)`. So square 0 is bit 0, square 63 is bit 63. Since bit 63 doesn't fit a signed 64-bit integer,
`rows_to_arrays` stores it as `uint64` then `.view(np.int64)` (same bits, signed). Later `targets_and_loss._packed_to_square_flags` unpacks bits 0-62 with `(packed // 2^i) % 2` and bit 63 from the sign (`packed < 0`).

### 5b. `chk` - check flags
`CHECK_WINDOWS = [(0,1), (0,3), (0,5), (1,1), (1,3), (1,5)]` = `(first_offset, number_of_own_plies)`. For window `(offset, n)` look at plies `p+offset, p+offset+2, …` (n of them, skipping the opponent's moves) and set the bit if **any** of those moves gives check.
Plies beyond the end of the game are ignored.

| bit | window | meaning | head |
|-----|--------|---------|------|
| 0 (value 1)  | (0,1) | the side to move gives check with the move it's about to play | `checking_in_1` |
| 1 (value 2)  | (0,3) | ... gives check within its next 3 moves | `checking_in_3` |
| 2 (value 4)  | (0,5) | ... within 5 moves | *unused by any head* |
| 3 (value 8)  | (1,1) | the opponent gives check on its very next move | `checked_in_1` |
| 4 (value 16) | (1,3) | ... within its next 3 moves | `checked_in_3` |
| 5 (value 32) | (1,5) | ... within 5 moves | *unused* |

Windows are nested, so a stored value like `56 = 8+16+32` means "the opponent checks me within 1, 3 and 5 moves"; `4` means "I give a check, but only on my 5th move from now, not within 3".
(`train` extracts the bits with `(chk // 2**flag) % 2`.)

### 5c. `ptte` and `result`
`ptte = n_moves - ply` (number of plies left, including the one played from this position). At training time `features.plies_till_end_transform` maps it to `exp(-0.2*(ptte-1))`, so "1 ply left" = 1.0 and long games trend to 0.
`result` is the same class for every position of one game (0 white wins / 1 draw / 2 black wins), i.e. the labels from the game outcome, not from an engine.

---------------------------------------------------------------------------------------------------

## 6. `rows_to_arrays` and `save_shard` - from Python objects to compressed-friendly arrays

`rows_to_arrays(rows)` stacks the list of `Row` tuples into numpy arrays with small dtypes (table in section 1):
- Scalar fields -> one array each (`column(field, dtype)`).
- `board` -> `np.stack` into `(n, 64) uint8`.
- The **legal moves list has a different length per position**, so it's stored in a "ragged array" form:
  `legal_len[i]` = number of legal moves of position i (uint8; max chess legal moves is 218 < 256) and `legal_data` = all codes of all positions glued one after another (uint16; max code 4096+4095 = 8191).
  Position i's moves are `legal_data[ start_i : start_i + legal_len[i] ]` where `start_i = sum(legal_len[:i])` (`GPUData.first_move_of_position` precomputes those starts).

`save_shard(rows, out_dir, shard_id)` writes `shard_{id:06d}.npz` via `np.savez` (uncompressed) and returns `False` if there were no rows (nothing written).

---------------------------------------------------------------------------------------------------

## 7. `main()` - the streaming loop

```
rng = default_rng(0); rows = []; games_in_shard = 0; shard_id = 0
open games.pgn (utf-8, errors='replace')
loop:
    game = chess.pgn.read_game(file)        # parses one game at a time → the 2.3 GB file never sits in memory
    if game is None: break
    game_rows = process_game(...)           # None if skipped
    rows.extend(game_rows); games_in_shard += 1      # NOTE: counts every game read, even skipped ones
    if games_in_shard == 2000:
        if save_shard(rows, out, shard_id): shard_id += 1; n_positions += len(rows)
        rows, games_in_shard = [], 0
after the loop: save the remaining rows as the last (smaller) shard
print("done: N positions in K shards")
```
Memory stays small because only ≤ 2000 games' worth of rows (~25k positions) are alive at once.

---------------------------------------------------------------------------------------------------

## 8. How the shards are consumed later (so you can see the whole flow)

1. `data.py: load_data` globs `data/shard_*.npz`, sorts them, and uses **all but the last 2** shards as training data (`4,764,749` positions here) and the **last 2** as validation (`49,384`).
2. `load_shards` concatenates every field across shards; `GPUData` converts `legal_data` to int32 and copies **everything** to the GPU.
3. Each training step, `GPUData.batch(indices)` gathers 256 positions' fields, expands the ragged legal moves back into `(legal_position, legal_from, legal_to, legal_is_capture)` lists (`decode_move` undoes `from*64+to+4096*capture`).
4. `GraphBuilder.build` makes node/global features + edges from `board, ep, castle, turn, half, rep`; `make_targets` builds all 11 targets from `next_from/next_to, moved, captured, chk, ptte, result, legal*`.

## 9. Observations
- The validation set is simply the last two shards (the last ~4000 games in the PGN), not a random split. If the PGN is sorted (by date/rating/tournament) validation is not representative.
- Because positions come from real games, the model learns to imitate the players in `games.pgn` (the label for `next_move` is the human/engine move, not the best move).
- `half`, `rep` and the future-looking labels are computed from the game history, which the model cannot see in a single position; `rep` and `half` are given to it as numbers, the rest is what it must predict.
- Bits 2 and 5 of `chk` (the 5-move windows) are computed and stored but no head trains on them.
- Games are filtered only by result and length; there is no rating filter or time-control filter in this file.
