# train_moe.py

import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from dnc_torch_zeligism_polymorphic.configuration import (
    controller_config,
    memory_config,
    training_config,
)
from dnc_torch_zeligism_polymorphic.parallel_controller_moe import ParallelControllerMoE
from repeat_copy import RepeatCopy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a Mixture of Experts DNC on the Repeat-Copy task"
    )
    parser.add_argument("--num_experts", type=int, default=2, help="Number of expert memory units")
    parser.add_argument(
        "--top_k", type=int, default=1, help="Number of experts to route each input to"
    )
    parser.add_argument(
        "--capacity_factor", type=float, default=1.5, help="Factor to determine expert capacity"
    )
    parser.add_argument(
        "--router_jitter",
        type=float,
        default=0.01,
        help="Noise added to router logits during training",
    )
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
        "--log_dir", type=str, default="logs/moe", help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--load_balancing_coef",
        type=float,
        default=0.01,
        help="Coefficient for load balancing loss",
    )
    return parser.parse_args()


def train(model: ParallelControllerMoE, task: RepeatCopy, args: argparse.Namespace) -> None:
    """Train the model on the repeat-copy task."""
    # Set up optimizer and loss function
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    # criterion = nn.BCELoss()

    # Set up TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Training loop
    start_time = time.time()
    for i, data in enumerate(task.generate(args.num_examples)):
        # Generate a new example
        inputs, true_outputs = data
        # seq_len, seq, target = task.generate_example()

        # Reset model state
        if hasattr(model, "detach_state"):
            model.detach_state()

        # Forward pass
        pred_outputs = model(inputs)

        # Compute loss
        task_loss: Tensor = task.loss(pred_outputs, true_outputs)
        # task_loss = criterion(output, target)

        # Add load balancing loss if available
        if hasattr(model, "load_balancing_loss"):
            loss = task_loss + args.load_balancing_coef * model.load_balancing_loss
        else:
            loss = task_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if (i + 1) % args.checkpoint == 0:
            elapsed_time = time.time() - start_time
            examples_per_second = args.checkpoint / elapsed_time
            print(
                f"[{i+1}/{args.num_examples}] Task Loss = {task_loss.item():.3f}, Total Loss = {loss.item():.3f}, Examples/sec = {examples_per_second:.2f}"
            )
            writer.add_scalar("Loss/task", task_loss.item(), i + 1)
            writer.add_scalar("Loss/total", loss.item(), i + 1)
            if hasattr(model, "load_balancing_loss"):
                writer.add_scalar("Loss/load_balancing", model.load_balancing_loss.item(), i + 1)
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
    model = ParallelControllerMoE(
        num_experts=args.num_experts,
        input_size=task.input_size,
        output_size=task.output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        use_rms_norm=args.use_rms_norm,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        router_jitter=args.router_jitter,
    )

    # Print model summary
    print(f"Model: Mixture of Experts Controller with {args.num_experts} experts")
    print(f"Input size: {task.input_size}, Output size: {task.output_size}")
    print(f"Top-k: {args.top_k}")
    print(f"Capacity factor: {args.capacity_factor}")
    print(f"Router jitter: {args.router_jitter}")
    print(f"Using RMS normalization: {args.use_rms_norm}")
    print(f"Load balancing coefficient: {args.load_balancing_coef}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train the model
    train(model, task, args)


if __name__ == "__main__":
    main()
