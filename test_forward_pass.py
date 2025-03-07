"""Script to test a single forward pass of both DNC implementations with constant weights.

This is a simplified script that initializes both DNC implementations with constant
weights and tests a single forward pass to verify they produce the same outputs.
"""

import inspect
import os
import sys

import numpy as np
import torch

# Import utility functions
from weight_initialization import init_weights_with_constant

# Import both DNC implementations
try:
    from dnc_architecture_graves_2016.controller_config import controller_config
    from dnc_architecture_graves_2016.dnc_module import DNC as NewDNC
    from dnc_architecture_graves_2016.memory_config import memory_config
    from dnc_architecture_graves_2016.repeat_copy import RepeatCopy
    from dnc_architecture_graves_2016.training_config import training_config
    from dnc_torch_zeligism.dnc import DNC as OldDNC

    # Also import the constants from the old code
    from dnc_torch_zeligism.training_configs import BATCH_SIZE as OLD_BATCH_SIZE
except ImportError as e:
    print(f"Error importing DNC implementations: {e}")
    print("Make sure you are running this script from the project root.")
    exit(1)


# Monkey patch the old DNC class to store controller_config
original_old_dnc_init = OldDNC.__init__


def patched_old_dnc_init(
    self, input_size, output_size, controller_config, memory_config, Controller=torch.nn.LSTM
):
    # Store the controller_config as an instance variable
    self.controller_config = controller_config
    # Call the original __init__
    original_old_dnc_init(
        self, input_size, output_size, controller_config, memory_config, Controller
    )


# Apply the monkey patch
OldDNC.__init__ = patched_old_dnc_init


def print_tensor_details(name, tensor):
    """Print details of a tensor for debugging.

    Args:
        name: Name of the tensor
        tensor: The tensor to print details of
    """
    print(
        f"{name}: shape={tensor.shape}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
    )
    print(f"  min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
    print(f"  sample values: {tensor.flatten()[:5].tolist()}")


def main():
    """Run a single forward pass test with constant weights."""
    print("Initializing models with constant weights...")

    # Create dataset for test input
    dataset = RepeatCopy()

    # Create configs for old DNC (without batch_size)
    old_controller_config = {
        "hidden_size": controller_config.get("hidden_size", 64),
        "num_layers": controller_config.get("num_layers", 1),
    }

    old_memory_config = {
        "memory_size": memory_config.get("memory_size", 128),
        "word_size": memory_config.get("word_size", 20),
        "num_writes": memory_config.get("num_writes", 1),
        "num_reads": memory_config.get("num_reads", 1),
    }

    # Make sure BATCH_SIZE in old implementation matches our new one
    print(f"Old batch size: {OLD_BATCH_SIZE}")
    print(f"New batch size: {training_config.get('batch_size', 8)}")

    # Create configs for new DNC (with batch_size)
    new_controller_config = controller_config.copy()

    new_memory_config = memory_config.copy()
    new_memory_config["batch_size"] = training_config.get("batch_size", 8)

    # Create DNCs
    print("Creating old DNC...")
    old_dnc = OldDNC(
        dataset.input_size, dataset.output_size, old_controller_config, old_memory_config
    )

    print("Creating new DNC...")
    new_dnc = NewDNC(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        controller_config=new_controller_config,
        memory_config=new_memory_config,
    )

    # Print parameter counts
    old_params = sum(p.numel() for p in old_dnc.parameters())
    new_params = sum(p.numel() for p in new_dnc.parameters())
    print(f"Old DNC parameters: {old_params}")
    print(f"New DNC parameters: {new_params}")

    # If parameter counts don't match, print more details
    if old_params != new_params:
        print("\nParameter details:")
        print("Old DNC:")
        for name, param in old_dnc.named_parameters():
            print(f"  {name}: {param.shape}, numel={param.numel()}")

        print("\nNew DNC:")
        for name, param in new_dnc.named_parameters():
            print(f"  {name}: {param.shape}, numel={param.numel()}")

    # Set all weights to a constant value for reproducibility
    WEIGHT_CONSTANT = 0.01
    print(f"Setting all weights to {WEIGHT_CONSTANT}...")
    init_weights_with_constant(old_dnc, WEIGHT_CONSTANT)
    init_weights_with_constant(new_dnc, WEIGHT_CONSTANT)

    # Generate a simple test input
    print("Generating test input...")
    inputs, true_outputs = next(dataset.generate(1))

    # Print input details
    print_tensor_details("Input", inputs)

    # Forward pass through both models
    print("\nRunning forward pass on old DNC...")
    old_dnc.eval()
    with torch.no_grad():
        try:
            old_output = old_dnc(inputs)
            print("Old DNC forward pass successful!")
        except Exception as e:
            print(f"Error in old DNC forward pass: {e}")
            import traceback

            traceback.print_exc()
            old_output = None

    print("\nRunning forward pass on new DNC...")
    new_dnc.eval()
    with torch.no_grad():
        try:
            new_output = new_dnc(inputs)
            print("New DNC forward pass successful!")
        except Exception as e:
            print(f"Error in new DNC forward pass: {e}")
            import traceback

            traceback.print_exc()
            new_output = None

    # Compare outputs if both are available
    if old_output is not None and new_output is not None:
        print("\nComparing outputs:")
        print_tensor_details("Old output", old_output)
        print_tensor_details("New output", new_output)

        # Calculate difference
        diff = torch.abs(old_output - new_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\nMax absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        # Check if outputs are close
        are_close = torch.allclose(old_output, new_output, rtol=1e-5, atol=1e-8)
        print(f"Outputs are close: {are_close}")

        # If outputs are not close, print largest differences
        if not are_close:
            _, indices = torch.topk(diff.flatten(), 5)
            print("\nLargest differences:")
            for idx in indices:
                # Convert flat index to multi-dimensional indices
                flat_idx = idx.item()
                shape_tuple = tuple(old_output.shape)
                multi_idx = np.unravel_index(int(flat_idx), shape_tuple)

                # Extract values using the multi-dimensional index
                idx_tuple = tuple(int(i) for i in multi_idx)
                val1 = old_output[idx_tuple].item()
                val2 = new_output[idx_tuple].item()

                print(
                    f"  At {idx_tuple}: old={val1:.6f}, new={val2:.6f}, diff={abs(val1-val2):.6f}"
                )
    else:
        print("Cannot compare outputs because at least one forward pass failed.")


if __name__ == "__main__":
    main()
