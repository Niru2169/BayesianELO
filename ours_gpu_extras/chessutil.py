import chess
import numpy as np

CAPTURE_CODE_OFFSET = 4096
MAX_COUNT = 255


def encode_move(from_square: int, to_square: int, is_capture: bool) -> int:
    code = from_square * 64 + to_square
    if is_capture:
        code += CAPTURE_CODE_OFFSET
    return code


def decode_move(code):
    to_square = code % 64
    from_square = (code // 64) % 64
    is_capture = code // CAPTURE_CODE_OFFSET
    return from_square, to_square, is_capture


def board_array(board: chess.Board) -> np.ndarray:
    squares = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        piece_code = piece.piece_type
        if piece.color == chess.BLACK:
            piece_code += 6
        squares[square] = piece_code
    return squares


def castle_bits(board: chess.Board) -> int:
    total = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        total += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        total += 2
    if board.has_kingside_castling_rights(chess.BLACK):
        total += 4
    if board.has_queenside_castling_rights(chess.BLACK):
        total += 8
    return total


def ep_square(board: chess.Board) -> int:
    if board.ep_square is not None and board.has_legal_en_passant():
        return board.ep_square
    return -1


def pos_key(board: chess.Board) -> tuple:
    return (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
            board.occupied_co[chess.WHITE], board.castling_rights, board.turn)


def legal_codes(board: chess.Board) -> list:
    codes = set()
    for move in board.legal_moves:
        codes.add(encode_move(move.from_square, move.to_square, board.is_capture(move)))
    return sorted(codes)


def repetition_count(board: chess.Board) -> int:
    if not board.move_stack:
        return 0

    replay = chess.Board(board.root().fen())
    seen = {pos_key(replay): 1}
    count = 0
    for move in board.move_stack:
        replay.push(move)
        key = pos_key(replay)
        count = seen.get(key, 0)
        seen[key] = count + 1
    return count
