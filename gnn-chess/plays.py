import argparse
import os
import sys

import chess
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ours_gpu_extras'))
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
def model_move(model, graph_builder, board: chess.Board) -> chess.Move:
    device = graph_builder.device
    codes = torch.tensor(legal_codes(board), device=device, dtype=torch.long)
    from_square, to_square, _ = decode_move(codes)

    graph = graph_builder.build_from_boards([board], [repetition_count(board)])
    edge_index, is_edge, legal_in = find_move_edges(graph, 0, from_square, to_square)
    outputs = model(graph, legal_in)

    scores = outputs['next_move'][edge_index]
    scores[~is_edge] = float('-inf')
    best = int(scores.argmax())
    from_best, to_best = int(from_square[best]), int(to_square[best])

    move = chess.Move(from_best, to_best)
    if move not in board.legal_moves:
        move = chess.Move(from_best, to_best, promotion=chess.QUEEN)
    return move


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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, graph_builder = load_model(args.model_path, device), GraphBuilder(device)
    board = chess.Board()
    model_color = chess.WHITE if args.model_plays == 'white' else chess.BLACK

    print(board)
    while not board.is_game_over():
        if board.turn == model_color:
            if args.search:
                move = search_move(model, graph_builder, board, max_depth=args.depth)
            else:
                move = model_move(model, graph_builder, board)
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
