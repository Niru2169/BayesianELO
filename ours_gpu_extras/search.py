from collections import Counter
import math
import os
import sys

import chess
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn-chess'))
from chessutil import decode_move, legal_codes, pos_key
from graph_builder import find_move_edges
from layers import scatter_and_softmax

IMPOSSIBLY_LOW_SCORE = -1e9


def _sample_subgraph_graph(graph: dict, drop_rate: float = 0.10) -> dict:
    """Sample a subgraph by randomly masking non-self mobility edges.

    Self-loops are always kept; each non-self edge is dropped with probability
    ``drop_rate`` (ScalableAlphaZero-style subgraph sampling for evaluation
    ensembles).
    """
    if drop_rate <= 0.0:
        return graph
    receivers, senders = graph['receiver'], graph['sender']
    is_self_loop = receivers == senders
    keep_mask = is_self_loop | (torch.rand_like(receivers, dtype=torch.float32) > drop_rate)

    sampled = dict(graph)
    sampled['receiver'] = receivers[keep_mask]
    sampled['sender'] = senders[keep_mask]
    sampled['edge_position'] = graph['edge_position'][keep_mask]
    sampled['edge_key'] = graph['edge_key'][keep_mask]
    if 'edge_geo_features' in graph:
        sampled['edge_geo_features'] = graph['edge_geo_features'][keep_mask]
    sampled['edges_per_position'] = torch.bincount(
        sampled['edge_position'], minlength=graph['n_positions']
    ).clamp_(min=1)
    return sampled


@torch.no_grad()
def evaluate_batch(model, graph_builder, boards: list, repetition_counts: Counter,
                   n_subgraphs: int = 1, subgraph_drop: float = 0.10) -> list:
    device = graph_builder.device
    n_boards = len(boards)
    codes_per_board = [legal_codes(board) for board in boards]
    moves_per_board = [len(codes) for codes in codes_per_board]

    all_codes = torch.tensor([code for codes in codes_per_board for code in codes],
                             device=device, dtype=torch.long)
    board_of_move = torch.repeat_interleave(
        torch.arange(n_boards, device=device), torch.tensor(moves_per_board, device=device))
    from_square, to_square, _ = decode_move(all_codes)

    repetitions = [repetition_counts.get(pos_key(board), 0) for board in boards]
    graph = graph_builder.build_from_boards(boards, repetitions)
    edge_index, is_edge, legal_in = find_move_edges(graph, board_of_move, from_square, to_square)
    outputs = model(graph, legal_in)

    logits = outputs['next_move'][edge_index].float()
    logits = torch.where(is_edge, logits, torch.full_like(logits, IMPOSSIBLY_LOW_SCORE))
    move_probs = scatter_and_softmax(logits[:, None], board_of_move, n_boards).squeeze(1)

    result_probs = torch.softmax(outputs['final_result'].float(), -1)

    # Subgraph value ensemble: average the outcome head over the full graph and
    # n_subgraphs - 1 randomly edge-masked subgraphs to reduce leaf variance.
    if n_subgraphs > 1:
        for _ in range(n_subgraphs - 1):
            sub_g = _sample_subgraph_graph(graph, drop_rate=subgraph_drop)
            _, _, sub_legal_in = find_move_edges(sub_g, board_of_move, from_square, to_square)
            sub_out = model(sub_g, sub_legal_in)
            result_probs = result_probs + torch.softmax(sub_out['final_result'].float(), -1)
        result_probs = result_probs / float(n_subgraphs)

    white_values = (result_probs[:, 0] - result_probs[:, 2]).cpu().numpy()

    codes_flat, probs_flat = all_codes.cpu().numpy(), move_probs.cpu().numpy()
    results = []
    start = 0
    for i in range(n_boards):
        end = start + moves_per_board[i]
        move_codes = codes_flat[start:end].tolist()
        probs = probs_flat[start:end].tolist()
        results.append((list(zip(move_codes, probs)), float(white_values[i])))
        start = end
    return results


def code_to_move(board: chess.Board, code: int) -> chess.Move:
    from_square, to_square, _ = decode_move(code)
    move = chess.Move(from_square, to_square)
    if board.is_legal(move):
        return move
    return chess.Move(from_square, to_square, promotion=chess.QUEEN)


class Node:
    def __init__(self, board: chess.Board, move, path_prob: float, policy_prob: float,
                 depth: int, move_code: int | None = None):
        self.board = board
        self.move = move
        self.move_code = move_code
        self.path_prob = path_prob
        self.policy_prob = policy_prob
        self.depth = depth
        self.children = []
        self.value = 0.0
        self.net_value = 0.0
        self.move_probs = []
        self.is_terminal = False
        # MCTS statistics (white-perspective value sum & visit count)
        self.visits = 0
        self.value_sum = 0.0
        self.virtual_losses = 0
        self.expanded = False

    def q_value(self, parent_turn_is_white: bool) -> float:
        total_visits = self.visits + self.virtual_losses
        if total_visits == 0:
            return 0.0
        sign = 1.0 if parent_turn_is_white else -1.0
        # Virtual losses penalize in-flight branches by -1.0 from the parent's perspective
        return (sign * self.value_sum - self.virtual_losses) / total_visits


def terminal_value(board: chess.Board):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
        return 0.0
    return None


def count_earlier_positions(board: chess.Board) -> Counter:
    history = Counter()
    if board.move_stack:
        replay = chess.Board(board.root().fen())
    else:
        replay = board.copy()
    history[pos_key(replay)] += 1
    for move in board.move_stack:
        replay.push(move)
        history[pos_key(replay)] += 1
    history[pos_key(board)] -= 1
    return history


# =====================================================================
# 1. MINIMAX SEARCH & GUMBEL / SOFT-MINIMAX DISTILLATION TARGETS
# =====================================================================
def evaluate_nodes(model, graph_builder, nodes: list, history: Counter,
                   n_subgraphs: int = 1) -> list:
    live_nodes = []
    for node in nodes:
        value = terminal_value(node.board)
        if value is not None:
            node.is_terminal, node.value, node.net_value = True, value, value
        else:
            live_nodes.append(node)

    if live_nodes:
        results = evaluate_batch(model, graph_builder, [node.board for node in live_nodes],
                                 history, n_subgraphs=n_subgraphs)
        for node, (move_probs, net_value) in zip(live_nodes, results):
            node.move_probs, node.net_value = move_probs, net_value
    return live_nodes


def expand(node: Node, min_prob: float) -> list:
    new_children = []
    for code, prob in node.move_probs:
        path_prob = node.path_prob * prob
        if node.depth > 0 and path_prob < min_prob:
            continue
        move = code_to_move(node.board, code)
        child_board = node.board.copy()
        child_board.push(move)
        child = Node(child_board, move, path_prob, prob, node.depth + 1, move_code=code)
        node.children.append(child)
        new_children.append(child)
    return new_children


def build_tree(model, graph_builder, board: chess.Board, history: Counter,
               max_depth: int, min_prob: float, n_subgraphs: int = 1) -> Node:
    root = Node(board.copy(), None, 1.0, 1.0, 0)
    frontier = [root]
    while frontier:
        live_nodes = evaluate_nodes(model, graph_builder, frontier, history, n_subgraphs=n_subgraphs)
        next_frontier = []
        for node in live_nodes:
            if node.depth < max_depth:
                next_frontier.extend(expand(node, min_prob))
        frontier = next_frontier
    return root


def back_up_values(root: Node):
    nodes_in_visit_order = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes_in_visit_order.append(node)
        stack.extend(node.children)

    for node in reversed(nodes_in_visit_order):
        if node.is_terminal:
            continue
        if not node.children:
            node.value = node.net_value
            continue
        child_values = [child.value for child in node.children]
        white_to_move = node.board.turn == chess.WHITE
        node.value = max(child_values) if white_to_move else min(child_values)


def pick_root_move(root: Node, board: chess.Board, policy_weight: float) -> chess.Move:
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    best_child, best_score = None, IMPOSSIBLY_LOW_SCORE
    for child in root.children:
        score = sign * child.value + policy_weight * child.policy_prob
        if score > best_score:
            best_child, best_score = child, score
    return best_child.move


def gumbel_minimax_targets(root: Node, c_scale: float = 2.0) -> tuple[dict[int, float], float]:
    """Gumbel / soft-minimax improved policy over legal root moves plus the backed-up value.

    exp pi = softmax(log pi_prior + c_scale * Q_minimax)  (Danihelka et al.).
    """
    if not root.children:
        return {}, float(root.net_value)

    sign = 1.0 if root.board.turn == chess.WHITE else -1.0
    logits = []
    codes = []
    for child in root.children:
        prior_logit = math.log(max(child.policy_prob, 1e-8))
        completed_q = sign * child.value
        logits.append(prior_logit + c_scale * completed_q)
        codes.append(child.move_code)

    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).tolist()
    return dict(zip(codes, probs)), float(root.value)


# =====================================================================
# 2. LIGHTWEIGHT VIRTUAL-LOSS BATCHED PUCT MCTS
# =====================================================================
def _select_puct_child(node: Node, c_puct: float) -> Node:
    parent_white = node.board.turn == chess.WHITE
    sqrt_total = math.sqrt(max(1, node.visits + node.virtual_losses))
    best_child, best_score = None, IMPOSSIBLY_LOW_SCORE
    for child in node.children:
        q = child.q_value(parent_white)
        u = c_puct * child.policy_prob * sqrt_total / (1 + child.visits + child.virtual_losses)
        score = q + u
        if score > best_score:
            best_child, best_score = child, score
    return best_child


def _expand_mcts_node(node: Node):
    if node.expanded or node.is_terminal:
        return
    for code, prob in node.move_probs:
        move = code_to_move(node.board, code)
        child_board = node.board.copy()
        child_board.push(move)
        child = Node(child_board, move, node.path_prob * prob, prob, node.depth + 1, move_code=code)
        t_val = terminal_value(child_board)
        if t_val is not None:
            child.is_terminal = True
            child.value = child.net_value = t_val
        node.children.append(child)
    node.expanded = True


def mcts_search_tree(model, graph_builder, board: chess.Board, history: Counter,
                     simulations: int = 64, batch_size: int = 16,
                     c_puct: float = 1.5, n_subgraphs: int = 1) -> Node:
    """Batched PUCT MCTS with virtual loss.

    Each wave selects ``batch_size`` leaves (virtual loss keeps them on distinct
    branches), evaluates them in one GPU forward pass, expands them, and
    backpropagates the true values with the virtual losses removed.
    """
    root = Node(board.copy(), None, 1.0, 1.0, 0)
    evaluate_nodes(model, graph_builder, [root], history, n_subgraphs=n_subgraphs)
    root.visits = 1
    root.value_sum = root.net_value
    _expand_mcts_node(root)

    if not root.children:
        return root

    n_waves = max(1, math.ceil(simulations / batch_size))
    for _ in range(n_waves):
        paths = []
        leaves_to_eval = []
        leaf_map = {}

        for _ in range(batch_size):
            curr = root
            path = [curr]
            curr.virtual_losses += 1
            while curr.expanded and not curr.is_terminal and curr.children:
                curr = _select_puct_child(curr, c_puct)
                curr.virtual_losses += 1
                path.append(curr)
            paths.append(path)

            if not curr.is_terminal and not curr.expanded and id(curr) not in leaf_map:
                leaf_map[id(curr)] = curr
                leaves_to_eval.append(curr)

        if leaves_to_eval:
            evaluate_nodes(model, graph_builder, leaves_to_eval, history, n_subgraphs=n_subgraphs)
            for leaf in leaves_to_eval:
                _expand_mcts_node(leaf)

        # Backpropagate the real values and remove the virtual losses
        for path in paths:
            leaf = path[-1]
            val = leaf.value if leaf.is_terminal else leaf.net_value
            for node in path:
                node.virtual_losses -= 1
                node.visits += 1
                node.value_sum += val

    return root


def mcts_targets(root: Node, temperature: float = 1.0) -> tuple[dict[int, float], float]:
    """MCTS visit-count policy distribution pi(s, a) and the root Q-value."""
    if not root.children:
        return {}, float(root.net_value)

    visits = torch.tensor([max(child.visits, 0) for child in root.children], dtype=torch.float32)
    if visits.sum() <= 0:
        probs = [child.policy_prob for child in root.children]
    elif temperature <= 1e-3:
        probs = F.one_hot(visits.argmax(), num_classes=len(root.children)).float().tolist()
    else:
        scaled = visits.pow(1.0 / temperature)
        probs = (scaled / scaled.sum()).tolist()

    codes = [child.move_code for child in root.children]
    root_q = root.value_sum / max(1, root.visits)
    return dict(zip(codes, probs)), float(root_q)


def pick_mcts_move(root: Node) -> chess.Move:
    best_child = max(
        root.children,
        key=lambda c: (c.visits, c.q_value(root.board.turn == chess.WHITE), c.policy_prob)
    )
    return best_child.move


# =====================================================================
# 3. UNIFIED ENTRYPOINTS FOR GAMEPLAY AND SEARCH DISTILLATION
# =====================================================================
def search_move(model, graph_builder, board: chess.Board, max_depth: int = 4,
                min_prob: float = 0.005, policy_weight: float = 2.0,
                mode: str = 'minimax', simulations: int = 64,
                batch_size: int = 16, n_subgraphs: int = 1) -> chess.Move:
    history = count_earlier_positions(board)
    if mode == 'mcts':
        root = mcts_search_tree(model, graph_builder, board, history,
                                simulations=simulations, batch_size=batch_size,
                                n_subgraphs=n_subgraphs)
        return pick_mcts_move(root)

    root = build_tree(model, graph_builder, board, history, max_depth, min_prob,
                      n_subgraphs=n_subgraphs)
    back_up_values(root)
    return pick_root_move(root, board, policy_weight)


def distill_search_step(model, optimizer, graph_builder, boards: list,
                        mode: str = 'both', max_depth: int = 3,
                        min_prob: float = 0.01, simulations: int = 48,
                        mcts_batch: int = 16, n_subgraphs: int = 1) -> float:
    """Distill shallow-to-deep policy (exp pi) and value (exp v) into the model.

    ``mode``:
      - 'minimax': Gumbel / soft-minimax backed-up policy & value targets.
      - 'mcts':    virtual-loss batched PUCT MCTS visit distribution & root Q.
      - 'both':    average the two policy and value targets.
    """
    model.eval()
    target_policies = []
    target_values = []
    repetitions = []

    for board in boards:
        history = count_earlier_positions(board)
        repetitions.append(history.get(pos_key(board), 0))

        mm_pi, mm_v = {}, 0.0
        mc_pi, mc_v = {}, 0.0

        if mode in ('minimax', 'both'):
            mm_root = build_tree(model, graph_builder, board, history,
                                 max_depth=max_depth, min_prob=min_prob, n_subgraphs=n_subgraphs)
            back_up_values(mm_root)
            mm_pi, mm_v = gumbel_minimax_targets(mm_root)

        if mode in ('mcts', 'both'):
            mc_root = mcts_search_tree(model, graph_builder, board, history,
                                       simulations=simulations, batch_size=mcts_batch,
                                       n_subgraphs=n_subgraphs)
            mc_pi, mc_v = mcts_targets(mc_root, temperature=1.0)

        if mode == 'minimax':
            pi_dict, val = mm_pi, mm_v
        elif mode == 'mcts':
            pi_dict, val = mc_pi, mc_v
        else:
            all_k = set(mm_pi.keys()) | set(mc_pi.keys())
            pi_dict = {k: 0.5 * (mm_pi.get(k, 0.0) + mc_pi.get(k, 0.0)) for k in all_k}
            val = 0.5 * (mm_v + mc_v)

        target_policies.append(pi_dict)
        target_values.append(val)

    model.train()
    device = graph_builder.device
    n_boards = len(boards)
    codes_per_board = [legal_codes(b) for b in boards]
    moves_per_board = [len(codes) for codes in codes_per_board]
    all_codes_list = [c for codes in codes_per_board for c in codes]
    all_codes = torch.tensor(all_codes_list, device=device, dtype=torch.long)
    board_of_move = torch.repeat_interleave(
        torch.arange(n_boards, device=device), torch.tensor(moves_per_board, device=device))
    from_sq, to_sq, _ = decode_move(all_codes)

    graph = graph_builder.build_from_boards(boards, repetitions)
    edge_index, is_edge, legal_in = find_move_edges(graph, board_of_move, from_sq, to_sq)
    outputs = model(graph, legal_in)

    # 1. Value distillation: MSE on the expected White outcome in [-1, 1]
    result_probs = torch.softmax(outputs['final_result'].float(), dim=-1)
    pred_white_value = result_probs[:, 0] - result_probs[:, 2]
    tgt_white_value = torch.tensor(target_values, device=device, dtype=torch.float32)
    value_loss = F.mse_loss(pred_white_value, tgt_white_value)

    # 2. Policy distillation: cross-entropy over the legal move edges
    logits = outputs['next_move'][edge_index].float()
    logits = torch.where(is_edge, logits, torch.full_like(logits, IMPOSSIBLY_LOW_SCORE))
    pred_move_probs = scatter_and_softmax(logits[:, None], board_of_move, n_boards).squeeze(1).clamp(min=1e-8)

    flat_target_probs = []
    for b_idx, codes in enumerate(codes_per_board):
        pi_map = target_policies[b_idx]
        probs = [pi_map.get(c, 0.0) for c in codes]
        total_p = sum(probs)
        if total_p > 0:
            probs = [p / total_p for p in probs]
        else:
            probs = [1.0 / max(1, len(codes)) for _ in codes]
        flat_target_probs.extend(probs)

    tgt_move_probs = torch.tensor(flat_target_probs, device=device, dtype=torch.float32)
    policy_loss = -(tgt_move_probs * torch.log(pred_move_probs)).sum() / max(1, n_boards)

    loss = value_loss + policy_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())
