from dataclasses import dataclass
from datetime import datetime
from typing import List

import chessboard_image as cbi
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

import checkpoint
import mcts
import wandb
from cnn import ChessCNN
from device import device
from replay_buffer import ReplayBuffer
from tensorboard import TensorBoard


@dataclass
class Config:
    save_folder: str = "./models"
    resume_path: str = ""

    max_sims: int = 100
    temperature: float = 1.0
    c_puct: float = 0.01

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 100
    replay_buffer_cap: int = 100000
    train_steps: int = 5


args = Config()


def board_video(fens: List[str]) -> np.ndarray:
    board_video = []

    for fen in fens:
        rendered_board = cbi.generate_pil(fen, size=400)
        rendered_board = np.array(rendered_board)
        rendered_board = rendered_board.transpose(2, 0, 1)
        board_video.append(rendered_board)

    return np.stack(board_video)


def selfplay(model: nn.Module):
    board = TensorBoard()
    root = mcts.Node(board)

    fens = []
    game_history = []

    steps = 0
    while not board.is_game_over():
        steps += 1

        fens.append(board.fen())

        pi = mcts.search(model, root, args.max_sims, args.temperature)

        game_history.append((board.to_tensor(), pi, board.turn))

        best_move_idx = np.random.choice(len(pi), p=pi)
        move_uci = TensorBoard.index_to_move().get(best_move_idx)

        chosen_child = None
        for child in root.children:
            if child.action.uci() == move_uci:
                chosen_child = child
                break

        assert chosen_child is not None, (
            f"Selected move {move_uci} was not in children!"
        )

        root = chosen_child
        root.parent = None
        board = root.board

    outcome = board.outcome()
    winner = outcome.winner

    history = []
    for board, pi, turn in game_history:
        if winner is None:
            z = 0.0
        elif winner == turn:
            z = 1.0
        else:
            z = -1.0
        history.append((board, pi, z))

    return history, board_video(fens), steps


def train(
    model: nn.Module,
    optimizer: optim.AdamW,
    replay_buffer: ReplayBuffer,
    run: wandb.Run,
    start_epoch: int = 0,
):
    for e in range(start_epoch, args.epochs):
        metrics = {
            "epoch": e,
            "loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "buffer_size": len(replay_buffer),
        }

        history, board_video, steps = selfplay(model)
        replay_buffer.save_game(history)

        model.train()
        for _ in range(args.train_steps):
            boards, policies, values = replay_buffer.sample_batch(args.batch_size)
            values = values.view(-1, 1)
            pred_policies, pred_values = model(boards)
            pred_policies = F.log_softmax(pred_policies, dim=1)

            # value loss
            value_loss = F.mse_loss(pred_values, values)

            # policy loss
            policy_loss = F.kl_div(pred_policies, policies, reduction="batchmean")

            loss = value_loss + policy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["loss"] += loss.item()
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()

        metrics["loss"] /= args.train_steps
        metrics["policy_loss"] /= args.train_steps
        metrics["value_loss"] /= args.train_steps
        metrics["selfplay"] = wandb.Video(board_video, fps=1, format="gif")
        metrics["steps"] = steps
        run.log(data=metrics)

        checkpoint.save(args.resume_path, e, run.id, model, optimizer, replay_buffer)


def main():
    args.resume_path = f"{args.save_folder}/latest_checkpoint.pt"

    action_space = len(TensorBoard.move_to_index())
    model = ChessCNN(channels=18, action_space=action_space)
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    replay_buffer = ReplayBuffer(args.replay_buffer_cap)

    start_epoch, wandb_id = checkpoint.load(
        args.resume_path, model, optimizer, replay_buffer, device
    )

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(
        entity="papaya147-ml",
        project="AlphaZero-MCTS-Chess",
        config=args.__dict__,
        name=f"AlphaZero-MCTS_Chess_sims={args.max_sims}_bs={args.batch_size}_{date_str}",
        job_type="train",
        id=wandb_id,
        resume="allow",
    )

    train(model, optimizer, replay_buffer, run, start_epoch)


if __name__ == "__main__":
    main()
