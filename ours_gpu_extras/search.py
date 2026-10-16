from collections import Counter

import os
import sys

import chess
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gnn-chess'))
from chessutil import decode_move, legal_codes, pos_key
from graph_builder import find_move_edges
from layers import scatter_and_softmax

IMPOSSIBLY_LOW_SCORE = -1e9


@torch.no_grad()
def evaluate_batch(model, graph_builder, boards: list, repetition_counts: Counter) -> list:
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
    def __init__(self, board: chess.Board, move, path_prob: float, policy_prob: float, depth: int):
        self.board = board
        self.move = move
        self.path_prob = path_prob
        self.policy_prob = policy_prob
        self.depth = depth
        self.children = []
        self.value = 0.0
        self.net_value = 0.0
        self.move_probs = []
        self.is_terminal = False


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


def evaluate_nodes(model, graph_builder, nodes: list, history: Counter) -> list:
    live_nodes = []
    for node in nodes:
        value = terminal_value(node.board)
        if value is not None:
            node.is_terminal, node.value, node.net_value = True, value, value
        else:
            live_nodes.append(node)

    if live_nodes:
        results = evaluate_batch(model, graph_builder, [node.board for node in live_nodes], history)
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
        child = Node(child_board, move, path_prob, prob, node.depth + 1)
        node.children.append(child)
        new_children.append(child)
    return new_children


def build_tree(model, graph_builder, board: chess.Board, history: Counter, max_depth: int, min_prob: float) -> Node:
    root = Node(board.copy(), None, 1.0, 1.0, 0)
    frontier = [root]
    while frontier:
        live_nodes = evaluate_nodes(model, graph_builder, frontier, history)
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


def search_move(model, graph_builder, board: chess.Board, max_depth: int = 4,
                min_prob: float = 0.005, policy_weight: float = 2.0) -> chess.Move:
    history = count_earlier_positions(board)
    root = build_tree(model, graph_builder, board, history, max_depth, min_prob)
    back_up_values(root)
    return pick_root_move(root, board, policy_weight)
