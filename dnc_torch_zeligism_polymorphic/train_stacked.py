# train_stacked.py

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from dnc_torch_zeligism_polymorphic.configuration import (
    controller_config,
    memory_config,
    training_config,
)
from dnc_torch_zeligism_polymorphic.stacked_controller import StackedController
from repeat_copy import RepeatCopy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Stacked DNC on the Repeat-Copy task")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of stacked layers")
    parser.add_argument("--use_rms_norm", action="store_true", help="Use RMS normalization")
    parser.add_argument(
        "--batch_size", type=int, default=training_config["batch_size"], help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=training_config["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=training_config["num_examples"],
        help="Number of examples to train on",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=training_config["checkpoint"], help="Checkpoint interval"
    )
    parser.add_argument(
        "--seed", type=int, default=training_config["random_seed"], help="Random seed"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/stacked", help="Directory for TensorBoard logs"
    )
    return parser.parse_args()


def train(model: nn.Module, task: RepeatCopy, args: argparse.Namespace) -> None:
    """Train the model on the repeat-copy task."""
    # Set up optimizer and loss function
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Set up TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Training loop
    start_time = time.time()
    for i, data in enumerate(task.generate(args.num_examples)):
        # Generate a new example
        inputs, true_outputs = data

        # Reset model state
        if hasattr(model, "detach_state"):
            model.detach_state()

        # Forward pass
        pred_outputs = model(inputs)

        # Compute loss
        # loss = criterion(pred_outputs, true_outputs)
        loss = task.loss(pred_outputs, true_outputs)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if (i + 1) % args.checkpoint == 0:
            elapsed_time = time.time() - start_time
            examples_per_second = args.checkpoint / elapsed_time
            print(
                f"[{i+1}/{args.num_examples}] Loss = {loss.item():.3f}, Examples/sec = {examples_per_second:.2f}"
            )
            writer.add_scalar("Loss/train", loss.item(), i + 1)
            start_time = time.time()

            # Save model checkpoint
            torch.save(
                {
                    "epoch": i + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                f"{args.log_dir}/checkpoint_{i+1}.pt",
            )

    # Close TensorBoard writer
    writer.close()


def main():
    args = parse_arguments()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Update batch size in memory_config
    memory_config["batch_size"] = args.batch_size

    # Create task
    # task = RepeatCopy(batch_size=args.batch_size)
    task = RepeatCopy()

    # Create model
    model = StackedController(
        num_layers=args.num_layers,
        input_size=task.input_size,
        output_size=task.output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        use_rms_norm=args.use_rms_norm,
    )

    # Print model summary
    print(f"Model: Stacked Controller with {args.num_layers} layers")
    print(f"Input size: {task.input_size}, Output size: {task.output_size}")
    print(f"Using RMS normalization: {args.use_rms_norm}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train the model
    train(model, task, args)


if __name__ == "__main__":
    main()
