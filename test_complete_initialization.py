"""Script to test DNC implementations with complete initialization of all weights and states.

This script ensures that all model parameters, buffers, and states are set to
constant values before comparing the outputs of old and new DNC implementations.
"""

import inspect
import os
import sys

import numpy as np
import torch

# Import utility functions
from complete_weight_initialization import (
    compare_models_completely,
    init_all_states_with_constant,
    init_all_weights_with_constant,
    init_all_with_value_map,
    print_all_tensors,
)

# Import both DNC implementations
try:
    from dnc_architecture_graves_2016.controller_config import controller_config
    # from dnc_architecture_graves_2016.dnc_module import DNC as NewDNC
    from dnc_adapted import DNC_Adapted as NewDNC
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
    """Run a test with complete initialization of all weights and states."""
    print("Initializing models with constant weights and states...")

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

    # Set all weights, buffers, and states to the same constant value
    WEIGHT_CONSTANT = 0.01
    STATE_CONSTANT = 0.0

    print(f"\nSetting all weights and buffers to {WEIGHT_CONSTANT}...")
    init_all_weights_with_constant(old_dnc, WEIGHT_CONSTANT)
    init_all_weights_with_constant(new_dnc, WEIGHT_CONSTANT)

    print(f"\nSetting all state variables to {STATE_CONSTANT}...")
    old_states = init_all_states_with_constant(old_dnc, STATE_CONSTANT)
    new_states = init_all_states_with_constant(new_dnc, STATE_CONSTANT)

    print("\nOld DNC states initialized:")
    for name, shape in old_states.items():
        print(f"  {name}: {shape}")

    print("\nNew DNC states initialized:")
    for name, shape in new_states.items():
        print(f"  {name}: {shape}")

    # Optionally print all tensors for debugging
    # print_all_tensors(old_dnc, "Old DNC")
    # print_all_tensors(new_dnc, "New DNC")

    # Generate a simple test input
    print("\nGenerating test input...")
    inputs, true_outputs = next(dataset.generate(1))
    print_tensor_details("Input", inputs)

    # Forward pass through both models
    print("\nTesting forward pass and comparing models...")
    are_equal, results = compare_models_completely(old_dnc, new_dnc, inputs)

    # Print summary of comparison
    print("\nComparison results:")
    print(f"  All equality tests passed: {results['all_equal']}")
    print(f"  Parameters equality test: {results['parameters_equal']}")
    print(f"  Buffers equality test: {results['buffers_equal']}")
    print(f"  Output equality test: {results['outputs_equal']}")

    # Try a second forward pass to see if differences accumulate
    print("\nRunning a second forward pass...")
    inputs2, _ = next(dataset.generate(1))
    are_equal2, results2 = compare_models_completely(old_dnc, new_dnc, inputs2)

    print("\nSecond comparison results:")
    print(f"  All equality tests passed: {results2['all_equal']}")
    print(f"  Parameters equality test: {results2['parameters_equal']}")
    print(f"  Buffers equality test: {results2['buffers_equal']}")
    print(f"  Output equality test: {results2['outputs_equal']}")


if __name__ == "__main__":
    main()
