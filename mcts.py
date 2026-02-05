from typing import List

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from device import device
from tensorboard import TensorBoard


class Node:
    def __init__(
        self,
        board: TensorBoard,
        parent: "Node" = None,
        action: chess.Move = None,
        prior: float = 0,
    ):
        self.parent = parent
        self.action = action
        self.board = board

        self.visits = 0
        self.value_sum = 0
        self.prior = prior

        self.children: List[Node] = []

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def expand(self, policy_preds: np.ndarray):
        legal_moves = list(self.board.legal_moves)
        move_to_index = TensorBoard.move_to_index()

        for move in legal_moves:
            uci_str = move.uci()
            move_idx = move_to_index.get(uci_str)
            prior = float(policy_preds[move_idx])

            board = self.board.copy()
            board.push(move)

            self.children.append(Node(board, parent=self, action=move, prior=prior))

    def select_best_child(self, c_puct=1.0):
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_total_visits = np.sqrt(self.visits)

        for child in self.children:
            # Q-value: The average value of this node (Win rate).
            # Range: [-1, 1]. If we've never visited, it's 0.
            if child.visits == 0:
                q_value = 0
            else:
                q_value = child.value_sum / child.visits

            # Prior (P): What the Neural Network originally guessed for this move.
            prior = child.prior

            # U-value (Exploration Bonus):
            # High if the Network liked it (prior) AND we haven't visited it much (visit_count).
            u_value = c_puct * prior * sqrt_total_visits / (1 + child.visits)

            # --- The Formula ---
            # PUCT = Q + U
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = child.action
                best_child = child

        return best_action, best_child

    def backpropagate(self, value: float):
        self.visits += 1
        self.value_sum += value

        if self.parent is not None:
            self.parent.backpropagate(-value)


def search(model: nn.Module, root: Node, max_sims: int, temperature: float = 0):
    model.eval()
    with torch.no_grad():
        for _ in range(max_sims):
            node = root

            while node.is_expanded() and not node.board.is_game_over():
                _, node = node.select_best_child()

            if node.board.is_game_over():
                outcome = node.board.outcome()
                if outcome.winner == chess.WHITE:
                    reward = 1.0
                elif outcome.winner == chess.BLACK:
                    reward = -1.0
                else:
                    reward = 0.0

                turn_sign = 1 if node.board.turn == chess.WHITE else -1
                value = reward * turn_sign
                node.backpropagate(value)
                continue

            tensor = node.board.to_tensor()
            tensor = tensor.unsqueeze(0)
            policy_logits, value = model(tensor)
            policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            value = value.item()

            node.expand(policy_probs)
            node.backpropagate(value)

    action_space_size = len(TensorBoard.move_to_index())
    pi = np.zeros(action_space_size)

    action_map = TensorBoard.move_to_index()

    for child in root.children:
        move_idx = action_map[child.action.uci()]
        pi[move_idx] = child.visits

    if temperature == 0:
        # Hard Argmax for competitive play
        best_idx = np.argmax(pi)
        pi = np.zeros_like(pi)
        pi[best_idx] = 1.0
    else:
        # Soften or sharpen distribution
        pi = pi ** (1.0 / temperature)
        pi = pi / np.sum(pi)

    return pi
