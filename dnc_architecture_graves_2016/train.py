import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dnc_architecture_graves_2016.controller_config import controller_config

# Fix imports to use the correct paths
from dnc_architecture_graves_2016.dnc_module import DNC
from dnc_architecture_graves_2016.memory_config import memory_config
from dnc_architecture_graves_2016.model_utils import (
    synchronize_epsilon_values,
    synchronize_memory_states,
)
from dnc_architecture_graves_2016.repeat_copy import RepeatCopy
from dnc_architecture_graves_2016.training_config import training_config
from dnc_torch_zeligism.dnc import DNC as OrigDNC

# Constants from training_config
LEARNING_RATE = training_config.get("learning_rate", 1e-4)
NUM_EXAMPLES = training_config.get("num_examples", 10000)
CHECKPOINT = training_config.get("checkpoint", 1000)

# Combine memory_config with batch_size
memory_config_combined = memory_config.copy()
memory_config_combined["batch_size"] = training_config["batch_size"]


def initialize_weights_to_constants(model, value_base=0.01, increment=0.005):
    """
    Initialize all trainable parameters to small constant values.

    Args:
        model: The model whose parameters to initialize
        value_base: The base value to start with
        increment: The increment between values for different parameters
    """
    value = value_base
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Setting {name} to constant {value:.4f}")
            param.data.fill_(value)
            value += increment

    return model


def initialize_state_to_constants(model, value_base=0.01, increment=0.005):
    """
    Initialize all state variables (non-trainable buffers and state dicts) to constants.

    Args:
        model: The model whose state to initialize
        value_base: The base value to start with
        increment: The increment between values for different states
    """
    value = value_base

    # Initialize DNC state
    if hasattr(model, "state"):
        for key in model.state:
            print(f"Setting state[{key}] to constant {value:.4f}")
            model.state[key].fill_(value)
            value += increment

    # Initialize memory state
    if hasattr(model, "memory") and hasattr(model.memory, "state"):
        for key in model.memory.state:
            print(f"Setting memory.state[{key}] to constant {value:.4f}")
            model.memory.state[key].fill_(value)
            value += increment

    # Initialize interface state if it exists
    if hasattr(model, "interface") and hasattr(model.interface, "state"):
        for key in model.interface.state:
            print(f"Setting interface.state[{key}] to constant {value:.4f}")
            model.interface.state[key].fill_(value)
            value += increment

    # Handle any registered buffers (non-parameter persistent state)
    for name, buffer in model.named_buffers():
        print(f"Setting buffer {name} to constant {value:.4f}")
        buffer.fill_(value)
        value += increment

    return model


def train_comparison(debug_mode=True, max_iterations=5):
    """
    Compare old and new DNC implementations with controlled weight initialization.

    Args:
        debug_mode: Whether to enable comparison mode (default: True)
        max_iterations: Number of training iterations to run (default: 5)
    """
    # Set random seed for reproducibility
    torch.manual_seed(training_config["random_seed"] or 42)

    # Choose dataset and initialize size of data's input and output
    dataset = RepeatCopy()

    # Initialize new DNC implementation
    new_dnc = DNC(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        controller_config=controller_config,
        memory_config=memory_config_combined,
    )

    # Initialize old DNC implementation
    orig_controller_config = {
        "hidden_size": controller_config["hidden_size"],
        "num_layers": controller_config["num_layers"],
    }
    orig_memory_config = {
        "memory_size": memory_config["memory_size"],
        "word_size": memory_config["word_size"],
        "num_writes": memory_config["num_writes"],
        "num_reads": memory_config["num_reads"],
    }

    orig_dnc = OrigDNC(
        dataset.input_size, dataset.output_size, orig_controller_config, orig_memory_config
    )

    # Set all weights to constants
    print("Setting new DNC weights to constants...")
    initialize_weights_to_constants(new_dnc)

    print("Setting original DNC weights to constants...")
    initialize_weights_to_constants(orig_dnc)

    # Initialize states to constants
    print("Setting new DNC state to constants...")
    initialize_state_to_constants(new_dnc)

    print("Setting original DNC state to constants...")
    initialize_state_to_constants(orig_dnc)

    # Initialize optimizers (just for the training loop to work)
    new_optimizer = torch.optim.SGD(new_dnc.parameters(), lr=0.01)
    orig_optimizer = torch.optim.SGD(orig_dnc.parameters(), lr=0.01)

    # Run comparison for a few iterations
    print("\n===== Starting comparison =====")

    for i, data in enumerate(dataset.generate(max_iterations)):
        print(f"\nIteration {i+1}/{max_iterations}")

        # Get data
        inputs, true_outputs = data

        # Forward pass through new DNC
        new_optimizer.zero_grad()
        new_outputs = new_dnc(inputs)
        new_loss = dataset.loss(new_outputs, true_outputs)
        print(f"New DNC - Output first values: {new_outputs[0, 0, :5]}")
        print(f"New DNC - Loss: {new_loss.item():.6f}")

        # Forward pass through original DNC
        orig_optimizer.zero_grad()
        orig_outputs = orig_dnc(inputs)
        orig_loss = dataset.loss(orig_outputs, true_outputs)
        print(f"Orig DNC - Output first values: {orig_outputs[0, 0, :5]}")
        print(f"Orig DNC - Loss: {orig_loss.item():.6f}")

        # Compare outputs
        output_diff = (new_outputs - orig_outputs).abs().mean().item()
        print(f"Output difference (mean abs): {output_diff:.6f}")

        # Don't perform backprop or parameter updates for this test
        # We just want to check if the forward passes match

        if i >= max_iterations - 1:
            break

    print("\n===== Comparison complete =====")


def debug_memory_operations():
    """Debug the memory read operations in both DNC implementations."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dataset
    dataset = RepeatCopy()

    # Initialize configurations
    memory_config_combined = memory_config.copy()
    memory_config_combined["batch_size"] = training_config["batch_size"]

    orig_controller_config = {
        "hidden_size": controller_config["hidden_size"],
        "num_layers": controller_config["num_layers"],
    }
    orig_memory_config = {
        "memory_size": memory_config["memory_size"],
        "word_size": memory_config["word_size"],
        "num_writes": memory_config["num_writes"],
        "num_reads": memory_config["num_reads"],
    }

    # Create both DNC implementations
    new_dnc = DNC(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        controller_config=controller_config,
        memory_config=memory_config_combined,
    )

    orig_dnc = OrigDNC(
        dataset.input_size, dataset.output_size, orig_controller_config, orig_memory_config
    )

    # Set constant value for memory matrices
    constant_value = 0.01

    # For new DNC
    if hasattr(new_dnc.memory, "state") and "memory" in new_dnc.memory.state:
        new_dnc.memory.state["memory"].fill_(constant_value)
        print(f"New DNC memory shape: {new_dnc.memory.state['memory'].shape}")
        print(f"New DNC memory sample: {new_dnc.memory.state['memory'][0, 0, :5]}")

    # For original DNC
    # We need to find the corresponding memory matrix
    for attr_name in dir(orig_dnc):
        attr = getattr(orig_dnc, attr_name)
        if isinstance(attr, torch.Tensor) and "memory" in attr_name.lower():
            attr.fill_(constant_value)
            print(f"Original DNC {attr_name} shape: {attr.shape}")
            print(f"Original DNC {attr_name} sample: {attr[:5]}")

    # Generate sample input
    for data in dataset.generate(1):
        inputs, _ = data
        break

    # Run a single forward pass
    with torch.no_grad():
        # Track memory operations for new DNC
        def hook_memory_read_new(module, input_args, output):
            if isinstance(input_args, dict) and "read_weights" in input_args:
                read_weights = input_args["read_weights"]
                memory = new_dnc.memory.state["memory"]
                print("\nNEW DNC MEMORY READ OPERATION:")
                print(f"Memory shape: {memory.shape}")
                print(f"Read weights shape: {read_weights.shape}")
                print(f"Memory sample: {memory[0, 0, :5]}")
                print(f"Read weights sample: {read_weights[0, 0, :5]}")

                # Manually calculate weighted sum
                batch_size = memory.shape[0]
                memory_size = memory.shape[1]
                word_size = memory.shape[2]
                num_reads = read_weights.shape[1]

                read_words = torch.zeros(batch_size, num_reads, word_size)
                for b in range(batch_size):
                    for r in range(num_reads):
                        for i in range(memory_size):
                            read_words[b, r] += memory[b, i] * read_weights[b, r, i]

                print(f"Manually calculated read words: {read_words[0, 0, :5]}")
                print(f"Output read words: {output[0, 0, :5]}")
            return output

        # Find memory read function in new DNC and attach hook
        if hasattr(new_dnc.memory, "read"):
            new_dnc.memory.read_hook = new_dnc.memory.read.register_forward_hook(
                hook_memory_read_new
            )

        # Track memory operations for original DNC
        def hook_memory_read_orig(module, input_args, output):
            print("\nORIGINAL DNC MEMORY READ OPERATION:")
            print(f"Input args types: {[type(arg) for arg in input_args]}")
            if len(input_args) >= 2:
                memory = input_args[0]
                read_weights = input_args[1]
                print(f"Memory shape: {memory.shape}")
                print(f"Read weights shape: {read_weights.shape}")
                print(f"Memory sample: {memory[0, :5]}")
                print(f"Read weights sample: {read_weights[0, 0, :5]}")

                # Manually calculate weighted sum
                batch_size = memory.shape[0]
                memory_size = memory.shape[1]
                word_size = memory.shape[2]
                num_reads = read_weights.shape[1]

                read_words = torch.zeros(batch_size, num_reads, word_size)
                for b in range(batch_size):
                    for r in range(num_reads):
                        for i in range(memory_size):
                            read_words[b, r] += memory[b, i] * read_weights[b, r, i]

                print(f"Manually calculated read words: {read_words[0, 0, :5]}")
                print(f"Output read words: {output[0, 0, :5]}")
            return output

        # Find memory read function in original DNC and attach hook
        for name, module in orig_dnc.named_modules():
            if "read" in name.lower() and hasattr(module, "forward"):
                module.read_hook = module.register_forward_hook(hook_memory_read_orig)

        # Run forward passes
        print("\n--- New DNC Forward Pass ---")
        new_output = new_dnc(inputs)
        print(f"New DNC output: {new_output[0, 0, :5]}")

        print("\n--- Original DNC Forward Pass ---")
        orig_output = orig_dnc(inputs)
        print(f"Original DNC output: {orig_output[0, 0, :5]}")

        # Remove hooks
        if hasattr(new_dnc.memory, "read_hook"):
            new_dnc.memory.read_hook.remove()

        for name, module in orig_dnc.named_modules():
            if hasattr(module, "read_hook"):
                module.read_hook.remove()


def main():
    # Run the comparison
    train_comparison()

    # Run the debug memory operations
    debug_memory_operations()


if __name__ == "__main__":
    main()
