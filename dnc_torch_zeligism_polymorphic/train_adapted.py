"""
Training script for the adapted DNC implementation.
Modeled after the original train.py from dnc_torch_zeligism.
"""

import argparse
import sys
import time

import numpy as np
import torch
from torch import Tensor, nn

# from torch.utils.tensorboard.writer import SummaryWriter
from dnc_torch_zeligism.training_configs import *
from dnc_torch_zeligism_polymorphic.configuration import (
    controller_config,
    memory_config,
    training_config,
)
from dnc_torch_zeligism_polymorphic.dnc_adapted import DNC_Adapted
from dnc_torch_zeligism_polymorphic.repeat_copy import RepeatCopy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a DNC on the Repeat-Copy task")
    parser.add_argument(
        "--num_memories", type=int, default=1, help="Number of parallel memory units"
    )
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
        "--checkpoint",
        type=int,
        default=training_config["checkpoint"],
        help="How often to report progress",
    )
    parser.add_argument(
        "--seed", type=int, default=training_config["random_seed"], help="Random seed"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/adapted", help="Directory for TensorBoard logs"
    )
    return parser.parse_args()


def train(
    model: nn.Module,
    task: RepeatCopy,
    args: argparse.Namespace,
):
    """Train the adapted DNC model on the given dataset.

    Args:
        model: The DNC_Adapted model to train.
        dataset: The dataset to train on.
        num_examples: Number of examples to train on.
        checkpoint: How often to report progress.

    Returns:
        None

    """
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("optimizer: ", optimizer)
    model.init_state()

    accuracy_list = []
    loss_list = []
    iteration_list = []

    # Track training time
    start_time = time.time()

    print("Model (train_adapted.py): ", model)
    # sys.exit()

    print(f"{args.num_examples=}")
    start_time = time.time()
    for i, data in enumerate(task.generate(args.num_examples)):
        # Zero gradients
        optimizer.zero_grad()

        t_start = time.time()

        # Unpack input/output
        inputs, true_outputs = data

        # Forward pass
        pred_outputs = model(inputs.clone())

        # Compute loss
        loss: Tensor = task.loss(pred_outputs, true_outputs)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        # print(f"Time taken for fwd+backward pass: {time.time() - t_start}")

        # Print report at checkpoints
        if (i + 1) % args.checkpoint == 0:
            # Create a fresh model state for evaluation
            # model.init_state()
            # with torch.no_grad():
            # eval_outputs = model(inputs.clone())

            accuracy = task.report(data, pred_outputs.data)
            elapsed_time = time.time() - start_time
            examples_per_second = args.checkpoint / elapsed_time
            # print(f"[{i + 1}/{num_examples}] Loss = {loss.item():.3f}", flush=True)
            print(
                f"[{i+1}/{args.num_examples}] Loss = {loss.item():.3f}, Examples/sec = {examples_per_second:.2f}"
            )
            print(f"   Accuracy: {100*accuracy}%")
            print(f"Time elapsed = {time.time() - start_time:.2f}s")
            start_time = time.time()

            accuracy_list.append(accuracy)
            loss_list.append(loss.item())
            iteration_list.append(i + 1)  # i+1 starts at 1

            # Optional: print memory usage
            # if torch.cuda.is_available():
            #     print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            #     print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("\nIterations, Losses, Accuracies")
    for it, loss, acc in zip(iteration_list, loss_list, accuracy_list, strict=True):
        print(f"{it:05d}, {loss:.4f}, {acc:.3f}")


def main():
    """Main function to set up and run the training."""
    args = parse_arguments()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed or torch.initial_seed())
    np.random.seed(args.seed)

    # Enable anomaly detection to help identify the source of the error
    # torch.autograd.set_detect_anomaly(True)

    print("Initializing dataset...")
    # Initialize dataset (RepeatCopy with default parameters)
    task = RepeatCopy()

    print("Initializing DNC_Adapted model...")
    print(f"{task.input_size=}")
    print(f"{task.output_size=}")
    print(f"{controller_config=}")
    print(f"{memory_config=}")

    # Initialize the adapted DNC model
    model = DNC_Adapted(
        input_size=task.input_size,
        output_size=task.output_size,
        controller_config=controller_config,
        memory_config=memory_config,
    )

    print("Starting training...")
    # Train the model
    train(model, task, args)

    print("Training completed!")


if __name__ == "__main__":
    main()
