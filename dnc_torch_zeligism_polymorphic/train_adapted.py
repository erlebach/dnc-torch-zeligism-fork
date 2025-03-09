"""
Training script for the adapted DNC implementation.
Modeled after the original train.py from dnc_torch_zeligism.
"""

import time
import sys

import torch

from dnc_torch_zeligism_polymorphic.dnc_adapted import DNC_Adapted
from dnc_torch_zeligism.repeat_copy import RepeatCopy
from dnc_torch_zeligism.training_configs import *


def train_adapted(model, dataset, num_examples=NUM_EXAMPLES, checkpoint=CHECKPOINT):
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
    optimizer = torch.optim.Adam(model.parameters())

    # Track training time
    start_time = time.time()
    print("number samples: ", num_examples, flush=True)
    print("checkpoint= ", checkpoint)

    for i, data in enumerate(dataset.generate(num_examples)):
        # print("NEXT TRAINING SAMPLE", flush=True)
        # Zero gradients
        optimizer.zero_grad()

        # Unpack input/output
        inputs, true_outputs = data

        # Reset model state at the beginning of each sequence
        model.init_state()
        # print("exited model.init state", flush=True)

        # Forward pass
        pred_outputs = model(inputs.clone())
        # print("exited model(inputs.clone())", flush=True)

        # Compute loss
        loss = dataset.loss(pred_outputs, true_outputs)
        # print("loss= ", loss.item(), flush=True)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        # print("After optimizer.step", flush=True)

        # Print report at checkpoints
        if (i + 1) % checkpoint == 0:
            # Create a fresh model state for evaluation
            model.init_state()
            with torch.no_grad():
                eval_outputs = model(inputs.clone())

            dataset.report(data, eval_outputs)
            print(f"[{i + 1}/{num_examples}] Loss = {loss.item():.3f}", flush=True)
            print(f"Time elapsed = {time.time() - start_time:.2f}s")

            # Optional: print memory usage
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def main():
    """Main function to set up and run the training."""
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED or torch.initial_seed())

    # Enable anomaly detection to help identify the source of the error
    # torch.autograd.set_detect_anomaly(True)

    print("Initializing dataset...")
    # Initialize dataset (RepeatCopy with default parameters)
    print("Before repeat copy", flush=True)
    dataset = RepeatCopy()
    print("After repeat copy", flush=True)

    # Define controller and memory configurations
    controller_config = {
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    }

    memory_config = {
        "memory_size": MEMORY_SIZE,
        "word_size": WORD_SIZE,
        "num_writes": NUM_WRITES,
        "num_reads": NUM_READS,
        "batch_size": BATCH_SIZE,
    }

    print("Initializing DNC_Adapted model...")
    # Initialize the adapted DNC model
    model = DNC_Adapted(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        controller_config=controller_config,
        memory_config=memory_config,
    )

    print(f"Model created with input_size={dataset.input_size}, output_size={dataset.output_size}")
    print(f"Memory config: {memory_config}")
    print(f"Controller config: {controller_config}")

    print("Starting training...")
    # Train the model
    train_adapted(model, dataset)

    print("Training completed!")


if __name__ == "__main__":
    main()
