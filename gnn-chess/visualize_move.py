
import argparse
import os
import sys

import chess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, '..', 'ours_gpu_extras'))

from chessutil import decode_move, legal_codes, repetition_count          # noqa: E402
from graph_builder import GraphBuilder, find_move_edges                    # noqa: E402
from layers import (concat_edge_with_ends, concat_edges_ends_only,         # noqa: E402
                    concat_globals_with_anything, global_average_pooling,
                    scatter_and_aggregate, scatter_and_softmax)
from model import GAT_HEADS, GAT_HEAD_OUT, leaky                           # noqa: E402
from plays import load_model                                               # noqa: E402

# ----------------------------------------------------------------------------------------------
# THE hardcoded position (Italian-style middlegame, white to move).  Change it here or pass --fen.
# ----------------------------------------------------------------------------------------------
FEN = 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7'

NODE_NAMES = ['rook', 'knight', 'bishop', 'queen', 'king', 'pawn',
              'is_black', 'is_white', 'sq_colour', 'rank', 'file', 'en_passant']
GLOBAL_NAMES = ['white_to_move', 'black_to_move', 'castle_WK', 'castle_WQ',
                'castle_BK', 'castle_BQ', 'halfmove*0.04', 'repetition*0.5']

COLOR = True


# ==============================================================================================
# tiny drawing toolkit
# ==============================================================================================
def _rgb(t):
    """t in [-1, 1] -> (r, g, b); grey at 0, blue negative, red positive."""
    base = (38, 38, 46)
    end = (255, 112, 67) if t >= 0 else (66, 133, 244)
    t = min(1.0, abs(t)) ** 0.7
    return tuple(int(b + (e - b) * t) for b, e in zip(base, end))


def cell(x, m):
    """One character that encodes a value."""
    t = 0.0 if m == 0 else max(-1.0, min(1.0, x / m))
    if COLOR:
        r, g, b = _rgb(t)
        return f'\x1b[38;2;{r};{g};{b}m█\x1b[0m'
    level = min(3, int(abs(t) * 4))
    return (['.', '+', '*', '#'] if t >= 0 else ['.', '-', '=', '%'])[level]


def bg(text, t, lo=(38, 38, 46), hi=(255, 140, 60)):
    """Text on a background whose colour goes lo -> hi for t in [0, 1]."""
    if not COLOR:
        return text
    t = max(0.0, min(1.0, t)) ** 0.6
    r, g, b = (int(a + (c - a) * t) for a, c in zip(lo, hi))
    return f'\x1b[48;2;{r};{g};{b}m\x1b[38;2;255;255;255m{text}\x1b[0m'


def vals(t):
    return [float(v) for v in t.detach().flatten().cpu()]


def strip(t, m=None):
    v = vals(t)
    m = max(abs(x) for x in v) if m is None else m
    return ''.join(cell(x, m) for x in v)


def banner(title):
    print()
    print('=' * 100)
    print(f'  {title}')
    print('=' * 100)


def sub(title):
    print(f'\n--- {title} ' + '-' * max(0, 94 - len(title)))


def stage(label, t, note=''):
    """One line: label | shape | colour strip | min / mean / max."""
    v = vals(t)
    shape = 'x'.join(str(s) for s in t.shape)
    print(f'  {label:<34}{shape:>7}  {strip(t)}  min {min(v):+.3f}  mean {sum(v) / len(v):+.3f}  '
          f'max {max(v):+.3f} {note}')


def sq(i):
    return chess.square_name(int(i))


def piece_char(board, s):
    p = board.piece_at(int(s))
    return p.symbol() if p else '.'


# ==============================================================================================
# the arithmetic microscope:  y = leaky_relu( W . x + b )  for one input vector
# ==============================================================================================
def show_matmul(title, lin, x, names=None, act=True, neurons=2, full=False, table_max_in=32):
    W, b = lin.weight.detach(), lin.bias.detach() if lin.bias is not None else None
    out_dim, in_dim = W.shape
    pre = W @ x + (b if b is not None else 0.0)
    post = leaky(pre) if act else pre
    fn = 'leaky_relu(W . x + b)' if act else 'W . x + b'
    sub(f'{title}:  y = {fn}      W is {out_dim} x {in_dim}  ({out_dim * in_dim} weights + {out_dim} biases)')
    print(f'  input  x ({in_dim:>3})  {strip(x)}')

    shown = range(out_dim) if full else range(min(neurons, out_dim))
    for j in shown:
        prod = W[j] * x
        s = float(prod.sum())
        bj = float(b[j]) if b is not None else 0.0
        print(f'\n  neuron {j}:')
        if in_dim <= table_max_in:
            print(f'    {"i":>3} {"input":<14}{"x_i":>9} {"w_i":>9} {"x_i*w_i":>10} {"running sum":>12}')
            run = 0.0
            for i in range(in_dim):
                run += float(prod[i])
                nm = names[i] if names else ''
                print(f'    {i:>3} {nm:<14}{float(x[i]):>+9.4f} {float(W[j, i]):>+9.4f} '
                      f'{float(prod[i]):>+10.4f} {run:>+12.4f}')
        else:
            print(f'    x      {strip(x)}')
            print(f'    w_{j:<4} {strip(W[j])}')
            print(f'    x*w    {strip(prod)}')
            top = torch.topk(prod.abs(), min(5, in_dim)).indices.tolist()
            print('    biggest products: ' + ',  '.join(
                f'[{i}] {float(x[i]):+.3f}*{float(W[j, i]):+.3f}={float(prod[i]):+.3f}' for i in top))
        lk = float(leaky(torch.tensor(s + bj)))
        print(f'    sum of products = {s:+.4f}   + bias {bj:+.4f}   = {s + bj:+.4f}'
              + (f'   -> leaky_relu = {lk:+.4f}' if act else ''))
    if out_dim > 1:
        print(f'\n  pre-activation  W.x+b ({out_dim:>3})  {strip(pre)}')
        if act:
            print(f'  after leaky_relu      ({out_dim:>3})  {strip(post)}')
    return post


# ==============================================================================================
# forward pass re-written step by step so every intermediate can be looked at
# (identical maths to model.py; checked against model(...) at the end of the run)
# ==============================================================================================
@torch.no_grad()
def trace(model, graph, legal_in):
    recv, send, epos, n_pos = graph['receiver'], graph['sender'], graph['edge_position'], graph['n_positions']
    n_nodes = graph['node_features'].shape[0]
    pos, glob = graph['node_features'], graph['global_features']
    T = {'input': {}, 'blocks': []}

    # ---- input block ----
    I, d = model.inp, T['input']
    d['v1_pre'] = I.v1(pos); d['v1'] = leaky(d['v1_pre'])
    d['v2_pre'] = I.v2(d['v1']); d['v2'] = leaky(d['v2_pre'])
    d['v'] = I.ln_v(d['v2'])
    d['e0_pre'] = I.e0(pos); d['e0'] = leaky(d['e0_pre'])
    d['e_cat'] = torch.cat([concat_edges_ends_only(d['e0'], recv, send), legal_in[:, None]], -1)
    d['e1_pre'] = I.e1(d['e_cat']); d['e1'] = leaky(d['e1_pre'])
    d['e'] = I.ln_e(d['e1'])
    d['u0_pre'] = I.u0(pos); d['u0'] = leaky(d['u0_pre'])
    d['u_pool'] = global_average_pooling(d['u0'], n_pos)
    d['u_cat'] = torch.cat([d['u_pool'], glob], -1)
    d['u1_pre'] = I.u1(d['u_cat']); d['u1'] = leaky(d['u1_pre'])
    d['u'] = I.ln_u(d['u1'])
    v, e, u = d['v'], d['e'], d['u']

    # ---- 10 full blocks ----
    for blk in model.blocks:
        b = {'v_prev': v, 'e_prev': e, 'u_prev': u}
        b['e_cat1'] = concat_edge_with_ends(v, e, recv, send)
        b['e1_pre'] = blk.e1(b['e_cat1']); b['e1'] = leaky(b['e1_pre'])
        b['e_cat2'] = concat_globals_with_anything(u, b['e1'], epos)
        b['e2_pre'] = blk.e2(b['e_cat2']); b['e_new'] = leaky(b['e2_pre'])
        b['v_mean'] = scatter_and_aggregate(b['e_new'], recv, n_nodes, 'mean')

        g = blk.gat
        b['g_cat'] = concat_edges_ends_only(b['v_mean'], recv, send)
        b['att1_pre'] = g.att1(b['g_cat']); b['att1'] = leaky(b['att1_pre'])
        b['att2_pre'] = g.att2(b['att1']); b['att_score'] = leaky(b['att2_pre'])
        b['att_w'] = scatter_and_softmax(b['att_score'], recv, n_nodes)
        b['proj'] = g.proj(b['g_cat']).view(-1, GAT_HEADS, GAT_HEAD_OUT)
        b['weighted'] = b['att_w'][..., None] * b['proj']
        b['gat_sum'] = scatter_and_aggregate(b['weighted'], recv, n_nodes, 'sum')
        b['v_gat'] = leaky(b['gat_sum'].flatten(1) + g.bias)
        b['u_new'] = global_average_pooling(b['v_gat'], n_pos)

        b['v_res'], b['e_res'], b['u_res'] = v + b['v_gat'], e + b['e_new'], u + b['u_new']
        v, e, u = blk.ln_v(b['v_res']), blk.ln_e(b['e_res']), blk.ln_u(b['u_res'])
        b['v'], b['e'], b['u'] = v, e, u
        T['blocks'].append(b)

    T['v'], T['e'], T['u'] = v, e, u
    return T


# ==============================================================================================
# sections
# ==============================================================================================
def section_position(board, graph, codes):
    banner('1. THE POSITION AND THE GRAPH THE MODEL BUILDS FROM IT')
    print(f'\n  FEN: {board.fen()}')
    print(f'  {"White" if board.turn else "Black"} to move, {len(codes)} legal moves\n')
    print('  (white pieces = bright white capitals, black pieces = orange lowercase)')
    for rank in range(7, -1, -1):
        row = ''
        for file in range(8):
            ch = piece_char(board, chess.square(file, rank))
            light = (rank + file) % 2 == 1
            if COLOR:
                back = (110, 100, 90) if light else (55, 55, 65)
                front = (255, 255, 255) if ch.isupper() else (255, 150, 40)
                row += (f'\x1b[48;2;{back[0]};{back[1]};{back[2]}m\x1b[38;2;{front[0]};{front[1]};{front[2]}m'
                        f'\x1b[1m {ch if ch != "." else " "} \x1b[0m')
            else:
                row += f' {ch} '
        print(f'   {rank + 1} {row}')
    print('     ' + ' '.join(f'{c} ' for c in 'abcdefgh') + '\n')

    n_edges = graph['receiver'].numel()
    print(f'  Graph: 64 nodes (one per square), {n_edges} directed edges '
          f'(self-loops + squares an occupying piece could reach, made symmetric).')
    print(f'  Edge (receiver=from-square, sender=to-square).  The model gives every edge one score; '
          f'only the legal ones are used.')


def section_inputs(board, graph):
    banner('2. THE RAW NUMBERS THE MODEL SEES')
    nf, gf = graph['node_features'], graph['global_features'][0]
    print('\n  Node features, one row per square (12 numbers each).  "." = 0')
    print('  ' + ' ' * 8 + ''.join(f'{n[:6]:>7}' for n in NODE_NAMES))
    for s in range(64):
        row = nf[s]
        txt = ''.join(f'{float(x):>7.2f}' if abs(float(x)) > 1e-9 else f'{".":>7}' for x in row)
        print(f'  {sq(s)} ({piece_char(board, s)})  {txt}')
    print('\n  Global features (8 numbers):')
    for n, x in zip(GLOBAL_NAMES, gf):
        print(f'    {n:<16}{float(x):>+8.3f}')


def section_params(model, dump_path):
    banner('3. EVERY PARAMETER OF THE MODEL')
    total = 0
    print(f'\n  Detailed rows for the input block, block 1 and the move head; the other 9 blocks have the')
    print(f'  same shapes as block 1 and get one summary line each.  Every number is in the dump file.')
    print(f'\n  {"name":<44}{"shape":<14}{"count":>8}{"mean":>9}{"std":>8}{"max|w|":>8}')
    per_block = {}
    for name, p in model.named_parameters():
        total += p.numel()
        parts = name.split('.')
        if parts[0] == 'blocks' and parts[1] != '0':
            per_block[int(parts[1])] = per_block.get(int(parts[1]), 0) + p.numel()
            continue
        if parts[0] == 'heads' and parts[1] != 'next_move':
            continue
        pd = p.detach()
        shape = 'x'.join(str(s) for s in p.shape)
        print(f'  {name:<44}{shape:<14}{p.numel():>8}{float(pd.mean()):>+9.3f}'
              f'{float(pd.std()) if p.numel() > 1 else 0:>8.3f}{float(pd.abs().max()):>8.3f}')
    for i, n in sorted(per_block.items()):
        print(f'  blocks.{i}.*  (same shapes as blocks.0){"":<18}{n:>8}')
    print('  heads.<the 10 other heads>.*  (same shapes: 48->72->32->1, final_result ends in 3)')
    print(f'\n  total parameters: {total:,}')
    if dump_path:
        with open(dump_path, 'w', encoding='utf-8') as f:
            for name, p in model.named_parameters():
                a = p.detach().cpu().numpy()
                f.write(f'### {name}   shape={tuple(a.shape)}\n')
                np.savetxt(f, a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1), fmt='%.5f')
                f.write('\n')
        print(f'  -> every single weight written to {dump_path}  ({os.path.getsize(dump_path) / 1e6:.1f} MB)')


def section_input_block(model, board, graph, T, fs, ei):
    banner('4. INPUT BLOCK  (12 raw numbers per square -> 48-number embeddings)')
    nf = graph['node_features']
    print(f'\n  Example: square {sq(fs)} ({piece_char(board, fs)}), the from-square of the chosen move.')
    show_matmul('node layer 1 (v1)', model.inp.v1, nf[fs], names=NODE_NAMES, act=True, neurons=2)
    d = T['input']
    sub('what happens to that square next')
    stage('v1  after leaky   (12 -> 48)', d['v1'][fs])
    stage('v2  after leaky   (48 -> 48)', d['v2'][fs])
    stage('v   after LayerNorm', d['v'][fs], '  <- initial node embedding')
    sub('the chosen edge and the global vector go through their own little MLPs')
    stage('edge: [e0[from] | e0[to] | legal?]', d['e_cat'][ei], '  (97 numbers in)')
    stage('edge: e1 after leaky (97 -> 48)', d['e1'][ei])
    stage('e   after LayerNorm', d['e'][ei], '  <- initial edge embedding')
    stage('global: [pool(u0) | 8 globals]', d['u_cat'][0], '  (56 numbers in)')
    stage('u   after LayerNorm', d['u'][0], '  <- initial global embedding')


def section_residual_stream(T, fs, ei):
    banner('5. THE RESIDUAL STREAM: the chosen edge / square / global vector after every block')
    for title, key, idx in [('chosen EDGE embedding e   (this is what gets scored)', 'e', ei),
                            ('from-square NODE embedding v', 'v', fs),
                            ('GLOBAL embedding u', 'u', 0)]:
        sub(title)
        rows = [('input block', T['input'][key][idx])] + \
               [(f'block {i + 1}', b[key][idx]) for i, b in enumerate(T['blocks'])]
        m = max(float(r.abs().max()) for _, r in rows)
        print(f'  {"":<12}{"".join(str(i % 10) for i in range(48))}   (colour scale shared: +/-{m:.2f})')
        prev = None
        for name, r in rows:
            change = '' if prev is None else f'  |change| {float((r - prev).norm()):.2f}'
            print(f'  {name:<12}{strip(r, m)}   norm {float(r.norm()):.2f}{change}')
            prev = r


def section_block(model, board, graph, T, k, fs, ei):
    b, blk = T['blocks'][k], model.blocks[k]
    to = int(graph['sender'][ei]) % 64
    banner(f'6. BLOCK {k + 1} OPENED UP  (edge {sq(fs)}->{sq(to)}, node {sq(fs)})')

    sub('a) edge update: two small layers  [edge | node_from | node_to]  ->  32  ->  [global | .]  ->  48')
    stage('concat [e_prev | v_from | v_to]', b['e_cat1'][ei], '(144 numbers)')
    stage('e1: Linear 144->32 + leaky', b['e1'][ei])
    stage('concat [u | e1]', b['e_cat2'][ei], '(80 numbers)')
    stage('e2: Linear 80->48 + leaky  = e_new', b['e_new'][ei])

    sub('b) every square averages the e_new of all edges that point INTO it')
    n_in = int((graph['receiver'] == fs).sum())
    stage(f'v_mean[{sq(fs)}]  (mean of {n_in} edges)', b['v_mean'][fs])

    sub('c) 24-head graph attention over the neighbours of the from-square')
    nbr = (graph['receiver'] == fs).nonzero().flatten()
    senders = [int(graph['sender'][i]) % 64 for i in nbr]
    labels = [f'{sq(s)}{"*" if s == fs else ""}' for s in senders]
    print(f'  neighbours of {sq(fs)} ({len(nbr)}): ' + ' '.join(f'{i}:{l}' for i, l in enumerate(labels)) +
          '     (* = itself)')
    att, score = b['att_w'][nbr], b['att_score'][nbr]                 # (n_nbr, 24)
    print(f'\n  attention weight each head gives each neighbour (softmax over neighbours, rows sum to 1):')
    print(f'  {"head":<6}{"".join(str(i % 10) for i in range(len(nbr)))}   strongest neighbour')
    for h in range(GAT_HEADS):
        top = int(att[:, h].argmax())
        print(f'  {h:<6}{strip(att[:, h], float(att.max()))}   {labels[top]:<5} {float(att[top, h]):.2f}')
    h0 = 0
    print(f'\n  head {h0} in numbers:   weight = exp(score - max) / sum')
    print(f'    {"nbr":<6}{"raw score":>11}{"softmax":>10}')
    for i, l in enumerate(labels):
        print(f'    {l:<6}{float(score[i, h0]):>+11.4f}{float(att[i, h0]):>10.4f}')
    print(f'    {"sum":<6}{"":>11}{float(att[:, h0].sum()):>10.4f}')
    stage(f'attention-mixed output at {sq(fs)}', b['v_gat'][fs], '(24 heads x 2 = 48)')

    sub('d) residual connection + LayerNorm')
    stage('edge: e_prev + e_new', b['e_res'][ei])
    stage('edge: LayerNorm  -> e', b['e'][ei])
    stage(f'node {sq(fs)}: v_prev + v_gat', b['v_res'][fs])
    stage(f'node {sq(fs)}: LayerNorm -> v', b['v'][fs])
    stage('global: u_prev + mean(v_gat)', b['u_res'][0])
    stage('global: LayerNorm -> u', b['u'][0])


def section_head(model, e_final, ei, neurons, full):
    banner('7. MOVE-SCORING HEAD: edge embedding (48) -> 72 -> 32 -> 1 number')
    head = model.heads['next_move']
    sub('the weight matrices as pictures (row = output neuron, column = input; colour scale per matrix)')
    for nm, lin in [('a  (48 -> 72)', head.a), ('b  (72 -> 32)', head.b), ('c  (32 -> 1)', head.c)]:
        W = lin.weight.detach()
        m = float(W.abs().max())
        print(f'\n  head.{nm}   max|w| = {m:.3f}')
        for r in range(W.shape[0]):
            print(f'  {r:>3} {strip(W[r], m)}  b={float(lin.bias[r]):+.3f}')

    x = e_final[ei]
    h1 = show_matmul('head layer a', head.a, x, act=True, neurons=neurons, full=full)
    h2 = show_matmul('head layer b', head.b, h1, act=True, neurons=neurons, full=full)
    logit = show_matmul('head layer c  (no activation: this IS the move score)', head.c, h2,
                        act=False, neurons=1, full=True)
    print(f'\n  => logit for this move = {float(logit[0]):+.4f}')
    print(f'  (the final bias {float(head.c.bias[0]):+.4f} is added to EVERY move, so it shifts all logits equally '
          f'and cancels out in softmax;')
    print(f'   only the "sum of products" part, {float(logit[0]) - float(head.c.bias[0]):+.4f} here, '
          f'makes moves different from each other.)')
    return float(logit[0])


def section_softmax(board, codes, labels, all_edge_logits, edge_idx, is_edge, sel):
    banner('8. THE OUTPUT: logits (before softmax) -> probabilities (after softmax)')
    logits = all_edge_logits[edge_idx].clone()
    n_edges = all_edge_logits.numel()
    print(f'\n  The network produced {n_edges} edge scores. Only {len(codes)} of those edges are legal moves;')
    print(f'  the rest are masked out (set to -inf) so they get probability 0.')

    m = float(logits.max())
    ex = torch.exp(logits - m)
    z = float(ex.sum())
    probs = ex / z

    print(f'\n  softmax:   p_i = exp(logit_i - {m:+.4f}) / {z:.4f}      '
          f'(subtracting the max is only for numerical safety)\n')
    order = torch.argsort(logits, descending=True).tolist()
    print(f'  {"rank":>4}  {"move":<14}{"logit":>9}{"logit-max":>11}{"exp()":>9}{"prob":>9}   bar')
    for rank, i in enumerate(order, 1):
        mark = ' <== explained above' if i == sel else ''
        bar = bg(' ' * int(round(float(probs[i]) * 60)), 1.0) if COLOR else '#' * int(round(float(probs[i]) * 60))
        print(f'  {rank:>4}  {labels[i]:<14}{float(logits[i]):>+9.4f}{float(logits[i]) - m:>+11.4f}'
              f'{float(ex[i]):>9.4f}{float(probs[i]) * 100:>8.2f}%   {bar}{mark}')
    print(f'  {"":>4}  {"sum":<14}{"":>9}{"":>11}{z:>9.4f}{float(probs.sum()) * 100:>8.2f}%')

    np.set_printoptions(precision=4, suppress=True, linewidth=110, threshold=100000)
    print('\n  Arrays in legal-move order (same order as the "moves" line right below):')
    print('  moves   :', ' '.join(l.split()[0] for l in labels))
    print('  BEFORE softmax (logits):')
    print('   ', np.array2string(logits.numpy(), separator=', ', prefix='    '))
    print('  AFTER softmax (probabilities):')
    print('   ', np.array2string(probs.numpy(), separator=', ', prefix='    '))

    ent = -float((probs * torch.log(probs.clamp_min(1e-12))).sum())
    print(f'\n  entropy of the distribution: {ent:.3f} nats (uniform over {len(codes)} moves would be '
          f'{np.log(len(codes)):.3f})')
    print('  Note: plays.py picks argmax of the raw logits.  softmax never changes the order, so it is the same move.')

    # ---- board heat maps ----
    from_p, to_p = np.zeros(64), np.zeros(64)
    for i, code in enumerate(codes):
        f, t, _ = decode_move(code)
        from_p[f] += float(probs[i])
        to_p[t] += float(probs[i])
    for title, arr in [('probability that the move STARTS on each square', from_p),
                       ('probability that the move ENDS on each square', to_p)]:
        sub(title + '  (percent)')
        top = arr.max() if arr.max() > 0 else 1.0
        for rank in range(7, -1, -1):
            row = ''.join(bg(f'{arr[chess.square(file, rank)] * 100:5.1f}', arr[chess.square(file, rank)] / top)
                          if arr[chess.square(file, rank)] > 0 else bg('    .', 0.0)
                          for file in range(8))
            print(f'   {rank + 1} {row}')
        print('       ' + '     '.join(c for c in 'abcdefgh'))
    return probs


def section_other_heads(outputs):
    sub('bonus: two of the other 10 heads that are trained alongside the move head')
    wdl = torch.softmax(outputs['final_result'][0], -1)
    print(f'  final_result softmax  white wins {float(wdl[0]) * 100:5.1f}%   draw {float(wdl[1]) * 100:5.1f}%   '
          f'black wins {float(wdl[2]) * 100:5.1f}%   (logits {[round(float(x), 3) for x in outputs["final_result"][0]]})')
    print(f'  checked_in_1 (sigmoid) {float(torch.sigmoid(outputs["checked_in_1"][0])) * 100:5.1f}%   '
          f'checking_in_1 (sigmoid) {float(torch.sigmoid(outputs["checking_in_1"][0])) * 100:5.1f}%')


# ==============================================================================================
def main():
    global COLOR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=os.path.join(HERE, 'chessgnn.pt'))
    ap.add_argument('--fen', default=FEN)
    ap.add_argument('--move', default=None, help='explain this move (UCI, e.g. d3d4) instead of the top one')
    ap.add_argument('--block', type=int, default=10, help='which of the 10 blocks to open up in section 6')
    ap.add_argument('--neurons', type=int, default=2, help='neurons shown with full arithmetic per layer')
    ap.add_argument('--full-mults', action='store_true', help='show arithmetic for every neuron of the head')
    ap.add_argument('--dump-weights', default=os.path.join(HERE, 'weights_dump.txt'))
    ap.add_argument('--no-dump', action='store_true')
    ap.add_argument('--no-color', action='store_true')
    args = ap.parse_args()

    COLOR = not args.no_color
    torch.set_grad_enabled(False)                       # we only look, we never train
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    if os.name == 'nt':
        os.system('')                                   # switches on ANSI colours in the Windows console

    device = torch.device('cpu')
    model = load_model(args.model, device)
    gb = GraphBuilder(device)
    board = chess.Board(args.fen)

    codes = legal_codes(board)
    codes_t = torch.tensor(codes, dtype=torch.long)
    from_t, to_t, _ = decode_move(codes_t)
    moves = []
    for c in codes:
        f, t, _ = decode_move(c)
        mv = chess.Move(int(f), int(t))
        moves.append(mv if board.is_legal(mv) else chess.Move(int(f), int(t), promotion=chess.QUEEN))
    labels = [f'{board.san(m):<7}({m.uci()})' for m in moves]

    graph = gb.build_from_boards([board], [repetition_count(board)])
    edge_idx, is_edge, legal_in = find_move_edges(graph, 0, from_t, to_t)

    with torch.no_grad():
        outputs = model(graph, legal_in)
        T = trace(model, graph, legal_in)
    all_logits = outputs['next_move']

    # which move do we explain?
    logits_legal = all_logits[edge_idx]
    if args.move:
        uci = [m.uci() for m in moves]
        if args.move not in uci:
            sys.exit(f'{args.move} is not legal here. Legal: {" ".join(uci)}')
        sel = uci.index(args.move)
    else:
        sel = int(logits_legal.argmax())
    ei = int(edge_idx[sel])
    fs = int(from_t[sel])
    print(f'\nExplaining the score of  {labels[sel]}   (graph edge #{ei} of {all_logits.numel()}).')
    print('Colour legend: blue = negative, red = positive, brighter = bigger |value| (scaled within each row).')

    section_position(board, graph, codes)
    section_inputs(board, graph)
    section_params(model, None if args.no_dump else args.dump_weights)
    section_input_block(model, board, graph, T, fs, ei)
    section_residual_stream(T, fs, ei)
    section_block(model, board, graph, T, max(1, min(10, args.block)) - 1, fs, ei)
    my_logit = section_head(model, T['e'], ei, args.neurons, args.full_mults)
    section_softmax(board, codes, labels, all_logits, edge_idx, is_edge, sel)
    section_other_heads(outputs)

    banner('CHECK: this script re-implements the forward pass by hand')
    diff_edges = float((model.heads['next_move'](T['e']).squeeze(-1) - all_logits).abs().max())
    print(f'\n  hand-traced logit for the chosen move : {my_logit:+.6f}')
    print(f'  model(...) logit for the chosen move  : {float(all_logits[ei]):+.6f}')
    print(f'  max |difference| over all {all_logits.numel()} edges     : {diff_edges:.2e}   (should be ~0)')


if __name__ == '__main__':
    main()
