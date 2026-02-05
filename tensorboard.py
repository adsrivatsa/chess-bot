import functools
from typing import Dict

import chess
import numpy as np
import torch

from device import device


class TensorBoard(chess.Board):
    def to_tensor(self) -> torch.Tensor:
        """
        Converts the board to a 18x8x8 tensor.
        Channels 0-11: Pieces
        Channel 12: En Passant
        Channel 13: Turn
        Channels 14-17: Castling Rights
        """
        tensor = torch.zeros(18, 8, 8)

        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        # --- Channels 0-11: Piece Locations ---
        for i in range(8):
            for j in range(8):
                square = chess.square(j, i)
                piece = self.piece_at(square)

                if piece:
                    channel = piece_map[piece.piece_type]
                    if piece.color == chess.BLACK:
                        channel += 6
                    tensor[channel, i, j] = 1

        # --- Channel 12: En Passant Target ---
        if self.ep_square is not None:
            # divmod returns (rank, file) from the square index
            rank, file = divmod(self.ep_square, 8)
            tensor[12, rank, file] = 1

        # --- Channel 13: Turn ---
        # We fill the whole plane with 1s so the Conv layer sees it everywhere
        if self.turn == chess.WHITE:
            tensor[13, :, :] = 1

        # --- Channels 14-17: Castling Rights ---
        # 14: White Kingside
        if self.has_kingside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1

        # 15: White Queenside
        if self.has_queenside_castling_rights(chess.WHITE):
            tensor[15, :, :] = 1

        # 16: Black Kingside
        if self.has_kingside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1

        # 17: Black Queenside
        if self.has_queenside_castling_rights(chess.BLACK):
            tensor[17, :, :] = 1

        return tensor.to(dtype=torch.float32, device=device)

    #
    # @classmethod
    # def from_tensor(cls, tensor: torch.Tensor, turn: Literal['w', 'b'] = 'w') -> 'TensorBoard':
    #     """
    #     Creates a TensorBoard instance from a 12x8x8 tensor.
    #     """
    #     board = cls(None)
    #
    #     # Reverse map: Channel index -> Piece Type
    #     # Channels 0-5 are White, 6-11 are Black
    #     channel_to_piece = {
    #         0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP,
    #         3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING
    #     }
    #
    #     # Get indices where pieces exist (values > 0)
    #     channels, rows, cols = torch.nonzero(tensor, as_tuple=True)
    #
    #     for c, r, file_idx in zip(channels, rows, cols):
    #         color = chess.BLACK if c >= 6 else chess.WHITE
    #         piece_type = channel_to_piece[c.item() % 6]
    #
    #         piece = chess.Piece(piece_type, color)
    #
    #         # Set piece on board
    #         # r is rank (0-7), file_idx is file (0-7)
    #         square = chess.square(file_idx.item(), r.item())
    #         board.set_piece_at(square, piece)
    #
    #     # Default behavior: It's White's turn.
    #     if turn == 'b':
    #         board.turn = chess.BLACK
    #     else:
    #         board.turn = chess.WHITE
    #
    #     return board

    @classmethod
    @functools.cache
    def move_to_index(cls) -> Dict[str, int]:
        """
        Creates a dictionary mapping every possible chess move (in UCI format)
        to a unique integer index.

        Returns:
            dict: {'a1a2': 0, 'a1a3': 1, ..., 'h7h8q': N}
        """
        moves = []

        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue

                # 1. Always add the base move (e.g. "e2e4", "a7a8")
                # Needed for non-pawn pieces moving to back ranks.
                moves.append(chess.Move(from_sq, to_sq).uci())

                # 2. Check for Valid Promotion Geometries
                # Calculate ranks (0-7)
                from_rank = from_sq // 8
                to_rank = to_sq // 8

                # White Pawn Promotion: Rank 6 -> Rank 7
                # Black Pawn Promotion: Rank 1 -> Rank 0
                is_promoting_move = (from_rank == 6 and to_rank == 7) or (
                    from_rank == 1 and to_rank == 0
                )

                if is_promoting_move:
                    # Add the 4 promotion variants
                    for promo_piece in [
                        chess.QUEEN,
                        chess.ROOK,
                        chess.BISHOP,
                        chess.KNIGHT,
                    ]:
                        moves.append(
                            chess.Move(from_sq, to_sq, promotion=promo_piece).uci()
                        )

        moves.sort()

        return {uci: idx for idx, uci in enumerate(moves)}

    @classmethod
    @functools.cache
    def index_to_move(cls) -> Dict[int, str]:
        move_to_index = cls.move_to_index()
        return {idx: uci for uci, idx in move_to_index.items()}
