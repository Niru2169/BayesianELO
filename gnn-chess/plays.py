import argparse
import os
import sys

import chess
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ours_gpu_extras'))
from chessutil import decode_move, legal_codes, repetition_count
from graph_builder import GraphBuilder, find_move_edges
from search import search_move
from model import ChessGNN


def load_model(path: str, device) -> ChessGNN:
    model = ChessGNN().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()


@torch.no_grad()
def policy_and_result(model, graph_builder, board: chess.Board):
    """Legal-move policy plus the predicted game result for ``board``.

    Returns a ``(moves, probs, result_probs)`` triple. ``moves`` and ``probs``
    are aligned by index, where ``probs`` is a softmax over the legal-move
    logits (illegal moves are masked out). ``result_probs`` is the model's
    outcome head as ``{'white': p, 'draw': p, 'black': p}``.
    """
    device = graph_builder.device
    codes = torch.tensor(legal_codes(board), device=device, dtype=torch.long)
    from_square, to_square, _ = decode_move(codes)

    graph = graph_builder.build_from_boards([board], [repetition_count(board)])
    edge_index, is_edge, legal_in = find_move_edges(graph, 0, from_square, to_square)
    outputs = model(graph, legal_in)

    scores = outputs['next_move'][edge_index]
    scores = scores.masked_fill(~is_edge, float('-inf'))
    probs = torch.softmax(scores, dim=0)

    legal_idx = [idx for idx in range(int(scores.shape[0])) if bool(is_edge[idx])]
    moves = []
    for idx in legal_idx:
        move = chess.Move(int(from_square[idx]), int(to_square[idx]))
        if move not in board.legal_moves:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        moves.append(move)

    outcome = torch.softmax(outputs['final_result'][0].float(), dim=0)
    result_probs = {
        'white': float(outcome[0]),
        'draw': float(outcome[1]),
        'black': float(outcome[2]),
    }
    return moves, probs[legal_idx], result_probs


@torch.no_grad()
def move_probabilities(model, graph_builder, board: chess.Board):
    """Policy distribution over the legal moves of ``board``.

    Returns a ``(moves, probs)`` pair aligned by index, where ``probs`` is a
    softmax over the legal-move logits (illegal moves are masked out).
    """
    moves, probs, _ = policy_and_result(model, graph_builder, board)
    return moves, probs


@torch.no_grad()
def predict(model, graph_builder, board: chess.Board, top_p: float = 0.9):
    """Nucleus move candidates plus the predicted game result.

    Returns ``(candidates, result_probs)``. ``candidates`` is the smallest set of
    highest-probability moves whose cumulative probability reaches ``top_p`` (see
    :func:`nucleus_moves`), as ``(move, probability)`` pairs. ``result_probs`` is
    the outcome head from :func:`policy_and_result`.
    """
    moves, probs, result_probs = policy_and_result(model, graph_builder, board)
    if top_p is None or top_p >= 1.0:
        candidates = [(move, float(p)) for move, p in zip(moves, probs)]
    else:
        candidates = nucleus_moves(moves, probs, top_p)
    return candidates, result_probs


def nucleus_moves(moves, probs, top_p: float):
    """Moves contributing to the top ``top_p`` fraction of the probability mass.

    Rather than always taking a fixed number of top moves (top-n), this takes the
    smallest set of highest-probability moves whose cumulative probability reaches
    ``top_p`` (for example 0.9 selects the moves that make up the top 90%).

    Returns a list of ``(move, probability)`` pairs in descending order.
    """
    order = torch.argsort(probs, descending=True)
    cumulative = torch.cumsum(probs[order], dim=0)

    threshold = torch.tensor(float(top_p), device=probs.device, dtype=probs.dtype)
    count = int(torch.searchsorted(cumulative, threshold).item()) + 1
    count = max(1, min(count, len(moves)))

    return [(moves[i], float(probs[i])) for i in order[:count].tolist()]


@torch.no_grad()
def model_move(model, graph_builder, board: chess.Board, top_p: float = 0.9,
               sample: bool = True, generator: torch.Generator | None = None) -> chess.Move:
    """Pick the model's move using top-p (nucleus) selection.

    ``top_p`` is the probability mass covered by the candidate moves. A move is
    then sampled from those candidates (proportional to their probabilities) when
    ``sample`` is true, otherwise the most probable candidate is returned.
    """
    moves, probs = move_probabilities(model, graph_builder, board)
    if not moves:
        raise ValueError('No legal moves in the current position.')

    if top_p is None or top_p >= 1.0:
        candidates = [(move, float(p)) for move, p in zip(moves, probs)]
    else:
        candidates = nucleus_moves(moves, probs, top_p)

    if sample and len(candidates) > 1:
        weights = torch.tensor([p for _, p in candidates], device=probs.device)
        total = float(weights.sum())
        if total <= 0.0:
            return candidates[0][0]
        pick = int(torch.multinomial(weights / total, 1, generator=generator).item())
        return candidates[pick][0]

    return max(candidates, key=lambda item: item[1])[0]


def ask_human_move(board: chess.Board):
    text = input('\nYour move (UCI, e.g. e2e4): ').strip()
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        print('Could not parse that move.')
        return None
    if move not in board.legal_moves:
        print('Illegal move.')
        return None
    return move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--model-plays', choices=['white', 'black'], default='black')
    parser.add_argument('--search', action='store_true', help='use shallow minimax on top of the policy')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='probability mass (0-1] covered by the candidate moves (nucleus selection)')
    parser.add_argument('--greedy', action='store_true',
                        help='take the most probable candidate instead of sampling from the nucleus')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed for reproducible sampling')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, graph_builder = load_model(args.model_path, device), GraphBuilder(device)
    board = chess.Board()
    model_color = chess.WHITE if args.model_plays == 'white' else chess.BLACK

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    print(board)
    while not board.is_game_over():
        if board.turn == model_color:
            if args.search:
                move = search_move(model, graph_builder, board, max_depth=args.depth)
            else:
                move = model_move(model, graph_builder, board, top_p=args.top_p,
                                  sample=not args.greedy, generator=generator)
            print(f'\nModel plays {move.uci()}')
        else:
            move = ask_human_move(board)
            if move is None:
                continue
        board.push(move)
        print()
        print(board)
    print('\nGame over:', board.result())


if __name__ == '__main__':
    main()
