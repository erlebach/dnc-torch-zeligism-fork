# train_parallel.py

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore

from dnc_torch_zeligism_polymorphic.configuration import (
    controller_config,
    memory_config,
    training_config,
)
from dnc_torch_zeligism_polymorphic.parallel_controller import ParallelController
from dnc_torch_zeligism_polymorphic.repeat_copy import RepeatCopy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Parallel DNC on the Repeat-Copy task")
    parser.add_argument(
        "--num_memories", type=int, default=3, help="Number of parallel memory units"
    )
    parser.add_argument(
        "--use_projections", action="store_true", help="Use projection layers for each branch"
    )
    parser.add_argument("--use_rms_norm", action="store_true", help="Use RMS normalization")
    parser.add_argument(
        "--combination_method",
        type=str,
        default="concat",
        choices=["concat", "sum", "mean"],
        help="Method to combine outputs from parallel branches",
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
        "--checkpoint", type=int, default=training_config["checkpoint"], help="Checkpoint interval"
    )
    parser.add_argument(
        "--seed", type=int, default=training_config["random_seed"], help="Random seed"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/parallel", help="Directory for TensorBoard logs"
    )
    return parser.parse_args()


def train(
    model: nn.Module,
    task: RepeatCopy,
    args: argparse.Namespace,
) -> None:
    """Train the model on the repeat-copy task.

    Args:
        model: The neural network model to train.
        task: The RepeatCopy task instance.
        args: Command line arguments containing training parameters.

    Returns:
        None

    """
    # Set up optimizer and loss function
    # optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # criterion = nn.BCELoss()

    accuracy_list = []
    loss_list = []
    iteration_list = []

    print("Model (train_parallel.py): ", model)

    # Set up TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Training loop
    start_time = time.time()
    for i, data in enumerate(task.generate(args.num_examples)):
        optimizer.zero_grad()

        t_start = time.time()

        # Unpack input/output
        inputs, true_outputs = data

        # Reset model state
        if hasattr(model, "detach_state"):
            model.detach_state()

        # Forward pass
        pred_outputs = model(inputs.clone())

        # Compute loss
        loss: Tensor = task.loss(pred_outputs, true_outputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Log progress
        if (i + 1) % args.checkpoint == 0:
            # Report on the task
            accuracy = task.report(data, pred_outputs.data)

            elapsed_time = time.time() - start_time
            examples_per_second = args.checkpoint / elapsed_time
            print(
                f"[{i+1}/{args.num_examples}] Loss = {loss.item():.3f}, Examples/sec = {examples_per_second:.2f}"
            )
            writer.add_scalar("Loss/train", loss.item(), i + 1)
            print(f"Time elapsed = {time.time() - start_time}")
            start_time = time.time()

            accuracy_list.append(accuracy)
            loss_list.append(loss.item())
            iteration_list.append(i + 1)  # i+1 starts at 1

            # Save model checkpoint
            # torch.save(
            #     {
            #         "epoch": i + 1,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "loss": loss.item(),
            #     },
            #     f"{args.log_dir}/checkpoint_{i+1}.pt",
            # )

    # print(f"{iteration_list=}")
    # print(f"{loss_list=}")
    # print(f"{accuracy_list=}")
    print("\nIterations, Losses, Accuracies")
    for it, loss, acc in zip(iteration_list, loss_list, accuracy_list, strict=True):
        print(f"{it:05d}, {loss:.4f}, {acc:.3f}")

    # Close TensorBoard writer
    writer.close()


def main():
    args = parse_arguments()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Update batch_size in memory_config
    memory_config["batch_size"] = args.batch_size

    # Create a config dictionary to pass to RepeatCopy
    task_config = {
        "batch_size": args.batch_size,
        # Add other necessary configuration parameters here
    }

    # Create task with config parameter instead of batch_size
    # task = RepeatCopy(config=task_config)
    task = RepeatCopy()

    # Create model
    model = ParallelController(
        num_memories=args.num_memories,
        input_size=task.input_size,
        output_size=task.output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        use_projections=args.use_projections,
        use_rms_norm=args.use_rms_norm,
        combination_method=args.combination_method,
    )

    # Print model summary
    print("=====================================================================")
    print(f"Model: Parallel Controller with {args.num_memories} memory units")
    print(f"Input size: {task.input_size}, Output size: {task.output_size}")
    print(f"Using projections: {args.use_projections}")
    print(f"Using RMS normalization: {args.use_rms_norm}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Number Memories: {args.num_memories}")
    print(f"Combination method: {args.combination_method}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("=====================================================================")

    # Train the model
    train(model, task, args)


if __name__ == "__main__":
    main()
