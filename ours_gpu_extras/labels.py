import os
from typing import NamedTuple

import chess
import numpy as np

from chessutil import MAX_COUNT, board_array, castle_bits, ep_square, pos_key, legal_codes

LOOKAHEAD = 10
CHECK_WINDOWS = [(0, 1), (0, 3), (0, 5), (1, 1), (1, 3), (1, 5)]


class MoveInfo(NamedTuple):
    from_square: int
    to_square: int
    captured_square: int
    rook_from: int
    rook_to: int
    gives_check: bool


class PositionState(NamedTuple):
    board: np.ndarray
    ep: int
    castle: int
    turn: int
    half: int
    rep: int
    legal: list


class Row(NamedTuple):
    board: np.ndarray
    ep: int
    castle: int
    turn: int
    half: int
    rep: int
    next_from: int
    next_to: int
    moved: int
    captured: int
    chk: int
    ptte: int
    result: int
    legal: list


def squares_to_number(squares) -> int:
    return sum(2 ** square for square in set(squares))


def captured_square(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        if board.turn == chess.WHITE:
            return move.to_square - 8
        return move.to_square + 8
    if board.is_capture(move):
        return move.to_square
    return -1


def castling_rook_squares(board: chess.Board, move: chess.Move):
    if not board.is_castling(move):
        return -1, -1

    back_rank_start = 0 if board.turn == chess.WHITE else 56
    is_kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
    if is_kingside:
        return back_rank_start + 7, move.from_square + 1
    return back_rank_start, move.from_square - 1


def pieces_moved_and_captured(move_infos: list, start_ply: int):
    current_to_original = {}
    moved_squares = []
    captured_squares = []
    end_ply = min(start_ply + LOOKAHEAD, len(move_infos))

    for ply in range(start_ply, end_ply):
        info = move_infos[ply]

        original = current_to_original.pop(info.from_square, info.from_square)
        moved_squares.append(original)

        if info.captured_square >= 0:
            captured_original = current_to_original.pop(info.captured_square, info.captured_square)
            captured_squares.append(captured_original)

        current_to_original[info.to_square] = original

        if info.rook_from >= 0:
            rook_original = current_to_original.pop(info.rook_from, info.rook_from)
            moved_squares.append(rook_original)
            current_to_original[info.rook_to] = rook_original

    return squares_to_number(moved_squares), squares_to_number(captured_squares)


def check_flags(move_infos: list, start_ply: int) -> int:
    flags = 0
    for flag, (first_offset, n_own_plies) in enumerate(CHECK_WINDOWS):
        plies_to_look_at = [start_ply + first_offset + 2 * i for i in range(n_own_plies)]
        plies_to_look_at = [ply for ply in plies_to_look_at if ply < len(move_infos)]
        if any(move_infos[ply].gives_check for ply in plies_to_look_at):
            flags += 2 ** flag
    return flags


def record_game(board: chess.Board, moves: list, sampled_plies: set):
    move_infos = []
    states = {}
    seen = {pos_key(board): 1}
    repetition = 0

    for ply, move in enumerate(moves):
        if ply in sampled_plies:
            states[ply] = PositionState(
                board=board_array(board), ep=ep_square(board), castle=castle_bits(board),
                turn=int(board.turn), half=min(board.halfmove_clock, MAX_COUNT),
                rep=min(repetition, MAX_COUNT), legal=legal_codes(board))

        rook_from, rook_to = castling_rook_squares(board, move)
        move_infos.append(MoveInfo(move.from_square, move.to_square, captured_square(board, move),
                                   rook_from, rook_to, board.gives_check(move)))

        board.push(move)
        key = pos_key(board)
        repetition = seen.get(key, 0)
        seen[key] = repetition + 1

    return move_infos, states


def build_rows(move_infos: list, states: dict, result: int) -> list:
    n_moves = len(move_infos)
    rows = []
    for ply in sorted(states):
        moved, captured = pieces_moved_and_captured(move_infos, ply)
        state = states[ply]
        rows.append(Row(
            board=state.board, ep=state.ep, castle=state.castle, turn=state.turn,
            half=state.half, rep=state.rep,
            next_from=move_infos[ply].from_square, next_to=move_infos[ply].to_square,
            moved=moved, captured=captured, chk=check_flags(move_infos, ply),
            ptte=n_moves - ply, result=result, legal=state.legal))
    return rows


def rows_to_arrays(rows: list):
    if not rows:
        return None

    legal_len = np.array([len(row.legal) for row in rows], dtype=np.uint8)
    all_legal_codes = (code for row in rows for code in row.legal)
    legal_data = np.fromiter(all_legal_codes, dtype=np.uint16, count=int(legal_len.sum()))

    def column(field, dtype):
        return np.array([getattr(row, field) for row in rows], dtype=dtype)

    return dict(
        board=np.stack([row.board for row in rows]),
        ep=column('ep', np.int8),
        castle=column('castle', np.uint8),
        turn=column('turn', np.uint8),
        half=column('half', np.uint8),
        rep=column('rep', np.uint8),
        next_from=column('next_from', np.uint8),
        next_to=column('next_to', np.uint8),
        moved=column('moved', np.uint64).view(np.int64),
        captured=column('captured', np.uint64).view(np.int64),
        chk=column('chk', np.uint8),
        ptte=column('ptte', np.int16),
        result=column('result', np.uint8),
        legal_len=legal_len, legal_data=legal_data,
    )


def save_shard(rows: list, out_dir: str, shard_id: int) -> bool:
    arrays = rows_to_arrays(rows)
    if arrays is None:
        return False
    np.savez(os.path.join(out_dir, f'shard_{shard_id:06d}.npz'), **arrays)
    return True
