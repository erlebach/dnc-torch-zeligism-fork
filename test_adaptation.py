"""
Test script to verify that the adapted DNC implementation produces equivalent
outputs to the original implementation.
"""

import sys
import torch
import torch.nn as nn

# Import adapted implementation
from dnc_adapted import DNC_Adapted

# Import original implementation
from dnc_torch_zeligism.dnc import DNC as OriginalDNC
from dnc_torch_zeligism.repeat_copy import RepeatCopy
from dnc_torch_zeligism.training_configs import *


def set_constant_weights(model, value=0.01):
    """Set all parameters of a model to a constant value."""
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)


def forward_pass_original(original_dnc, x_t, batch_size):
    with torch.no_grad():
        # Controller input
        orig_controller_input = torch.cat(
            [x_t.view(batch_size, -1), original_dnc.read_words.view(batch_size, -1)], dim=1
        )
        # Controller
        orig_controller_output, _ = original_dnc.controller(
            orig_controller_input.unsqueeze(0),
            None,  # Original DNC initializes hidden state internally
        )
        # Interface
        orig_interface_vectors = original_dnc.interface_layer(orig_controller_output.squeeze(0))
        # Memory
        for k, v in orig_interface_vectors.items():
            # Add the norm of the array to the print statement
            print(f"orig interface vectors, {k=}, {v.shape=}, {v.norm()=}")
        orig_read_words = original_dnc.memory.update(orig_interface_vectors)  # ### <<<<<a
        print("original_dnc")
        print(type(original_dnc.memory))
        print("orig_read_words")
        # print_orig_memory(original_dnc.memory, adapted_dnc.memory)
        # Output
        orig_pre_output = torch.cat(
            [orig_controller_output.squeeze(0), orig_read_words.view(batch_size, -1)], dim=1
        )
        orig_output = original_dnc.output_layer(orig_pre_output)

        # Return values needed for comparison
        return (
            orig_controller_input,
            orig_controller_output,
            orig_interface_vectors,
            orig_read_words,
            orig_pre_output,
            orig_output,
        )


def forward_pass_adapted(adapted_dnc, x_t, batch_size):
    # Controller input
    adapt_controller_input = torch.cat(
        [x_t.view(batch_size, -1), adapted_dnc.state["read_words"].view(batch_size, -1)],
        dim=1,
    )
    # Controller
    adapt_controller_output, (h_n, c_n) = adapted_dnc.controller(
        adapt_controller_input.unsqueeze(0),
        (adapted_dnc.state["controller_h"], adapted_dnc.state["controller_c"]),
    )
    adapted_dnc.state["controller_h"] = h_n
    adapted_dnc.state["controller_c"] = c_n
    # Interface
    adapt_interface_vectors = adapted_dnc.interface({"output": adapt_controller_output.squeeze(0)})
    # Memory
    for k, v in adapt_interface_vectors.items():
        print(f"adapt interface vectors, {k=}, {v.shape=}, {v.norm()=}")
    # print(f"adapt, {adapt_interface_vectors=}")
    adapt_read_words = adapted_dnc.memory(adapt_interface_vectors)
    adapted_dnc.state["read_words"] = adapt_read_words
    # Output
    adapt_pre_output = torch.cat(
        [adapt_controller_output.squeeze(0), adapt_read_words.view(batch_size, -1)], dim=1
    )
    adapt_output = adapted_dnc.output_layer(adapt_pre_output)

    return (
        adapt_controller_input,
        adapt_controller_output,
        adapt_interface_vectors,
        adapt_read_words,
        adapt_pre_output,
        adapt_output,
    )


def compare_forward_pass(original_dnc, adapted_dnc, inputs):
    """Compare intermediate results between original and adapted DNC."""
    print("+++++ compare_forward_pass")
    # Initialize states
    original_dnc.init_state()
    adapted_dnc.init_state()

    # Get batch size from input tensor
    batch_size = inputs.size(1)

    # Process each step in the sequence
    print(f"{inputs.size(0)=}")  # 15
    for i in range(inputs.size(0)):
        print("+++++++++++++++++++++++++++++++++++++++++++")
        print(f"\nStep {i}:")

        # Get current input
        x_t = inputs[i]
        # Get original DNC forward pass values
        print("******* forward_pass_original  ++++++++++++")
        (
            orig_controller_input,
            orig_controller_output,
            orig_interface_vectors,
            orig_read_words,
            orig_pre_output,
            orig_output,
        ) = forward_pass_original(original_dnc, x_t, batch_size)

        # Adapted DNC
        print("******* forward_pass_adapted  ++++++++++++")
        with torch.no_grad():
            (
                adapt_controller_input,
                adapt_controller_output,
                adapt_interface_vectors,
                adapt_read_words,
                adapt_pre_output,
                adapt_output,
            ) = forward_pass_adapted(adapted_dnc, x_t, batch_size)

        # Compare results
        # controller_output_comparison(orig_controller_
        print("\nController input comparison:")
        print("Original:", orig_controller_input[0, :5])
        print("Adapted:", adapt_controller_input[0, :5])

        print("\nController output comparison:")
        print("Original:", orig_controller_output[0, 0, :5])
        print("Adapted:", adapt_controller_output[0, 0, :5])

        print("\nInterface vectors comparison:")
        print("Original read keys:", orig_interface_vectors["read_keys"][0, 0, :5])
        print("Adapted read keys:", adapt_interface_vectors["read_keys"][0, 0, :5])

        print("\nDebugging memory operations...")
        debug_memory_operations(original_dnc.memory, adapted_dnc.memory, orig_interface_vectors)

        print("\nRead words comparison:")
        print("Original:", orig_read_words[0, 0, :5])
        print("Adapted:", adapt_read_words[0, 0, :5])

        print("\nPre-output comparison:")
        print("Original:", orig_pre_output[0, :5])
        print("Adapted:", adapt_pre_output[0, :5])

        print("\nOutput comparison:")
        print("Original:", orig_output[0, :5])
        print("Adapted:", adapt_output[0, :5])

        # Check if differences appear
        if not torch.allclose(orig_controller_input, adapt_controller_input, rtol=1e-5, atol=1e-8):
            print("\nDifference first appears in controller input!")
            return

        if not torch.allclose(
            orig_controller_output, adapt_controller_output, rtol=1e-5, atol=1e-8
        ):
            print("\nDifference first appears in controller output!")
            return

        if not torch.allclose(
            orig_interface_vectors["read_keys"],
            adapt_interface_vectors["read_keys"],
            rtol=1e-5,
            atol=1e-8,
        ):
            print("\nDifference first appears in interface vectors!")
            return

        if not torch.allclose(orig_read_words, adapt_read_words, rtol=1e-5, atol=1e-8):
            print("\nDifference first appears in read words!")
            return

        if not torch.allclose(orig_pre_output, adapt_pre_output, rtol=1e-5, atol=1e-8):
            print("\nDifference first appears in pre-output!")
            return

        if not torch.allclose(orig_output, adapt_output, rtol=1e-5, atol=1e-8):
            print("\nDifference first appears in final output!")
            return


def print_orig_memory(original_memory, adapted_memory):
    print("\n=== print_orig_memory ===")
    print("Original memory data mean:", original_memory.memory_data.mean().item())
    print("Adapted memory data mean:", adapted_memory.state["memory"].mean().item())
    print(f"{original_memory.memory_data[0][0:2]=}")
    print(f"{adapted_memory.memory_data[0][0:2]=}")


def debug_memory_operations(original_memory, adapted_memory, interface_vectors):
    """Debug memory operations to find the discrepancy."""
    print("\n=== Memory Debugging ===")

    # Clone interface vectors to avoid modifying originals
    orig_vectors = {k: v.clone() for k, v in interface_vectors.items()}
    adapt_vectors = {k: v.clone() for k, v in interface_vectors.items()}

    # Step 1: Check initial memory state
    print("\nInitial Memory State:")
    print("Original memory data mean:", original_memory.memory_data.mean().item())
    print("Adapted memory data mean:", adapted_memory.state["memory"].mean().item())
    print(f"{original_memory.memory_data[0][0:2]=}")
    print(f"{adapted_memory.memory_data[0][0:2]=}")

    sys.exit()  # ERROR ALREADY OCCURS BY THIS POINT

    # Step 2: Check content addressing
    print("\nContent Addressing:")
    original_read_content = original_memory.content_based_address(
        original_memory.memory_data, orig_vectors["read_keys"], orig_vectors["read_strengths"]
    )

    adapted_read_content = adapted_memory.content_based_address(
        adapted_memory.state["memory"], adapt_vectors["read_keys"], adapt_vectors["read_strengths"]
    )

    print("Original read content mean:", original_read_content.mean().item())
    print("Adapted read content mean:", adapted_read_content.mean().item())

    # Step 3: Check read weights
    print("\nRead Weights:")
    # Update the original memory (we need this to get internal states)
    original_read_words = original_memory.update(orig_vectors)

    # Check the read weights
    print("Original read weights mean:", original_memory.read_weights.mean().item())
    print("Adapted read weights mean:", adapted_memory.state["read_weights"].mean().item())

    # Step 4: Check read words calculation
    print("\nRead Words Calculation:")
    print(
        "Original memory data * read weights =",
        torch.matmul(original_memory.read_weights, original_memory.memory_data)[0, 0, :5],
    )
    print(
        "Adapted memory data * read weights =",
        torch.matmul(adapted_memory.state["read_weights"], adapted_memory.state["memory"])[
            0, 0, :5
        ],
    )

    return original_read_words


def verify_memory_initialization(original_memory, adapted_memory):
    """Verify that both memory implementations are initialized identically."""
    print("\n=== Memory Initialization Verification ===")

    # Step 1: Check memory matrix
    print("\nMemory Matrix Comparison:")
    print("Original memory shape:", original_memory.memory_data.shape)
    print("Adapted memory shape:", adapted_memory.state["memory"].shape)
    print("Original memory sample:", original_memory.memory_data[0, 0, :5])
    print("Adapted memory sample:", adapted_memory.state["memory"][0, 0, :5])
    print("Original memory mean:", original_memory.memory_data.mean().item())
    print("Adapted memory mean:", adapted_memory.state["memory"].mean().item())

    # Step 2: Check other components
    print("\nComponent Comparison:")
    print(
        "Usage vectors match:", torch.allclose(original_memory.usage, adapted_memory.state["usage"])
    )
    print(
        "Link matrices match:", torch.allclose(original_memory.link, adapted_memory.state["link"])
    )
    print(
        "Precedence weights match:",
        torch.allclose(
            original_memory.precedence_weights, adapted_memory.state["precedence_weights"]
        ),
    )
    print(
        "Read weights match:",
        torch.allclose(original_memory.read_weights, adapted_memory.state["read_weights"]),
    )
    print(
        "Write weights match:",
        torch.allclose(original_memory.write_weights, adapted_memory.state["write_weights"]),
    )


def debug_memory_initialization(original_memory, adapted_memory):
    """Detailed debugging of memory initialization with side-by-side comparison."""
    print("\n=== Memory Initialization Comparison ===")

    # Configuration comparison
    print("\nConfiguration Comparison:")
    print(f"{'Parameter':<20} | {'Original':<10} | {'Adapted':<10}")
    print("-" * 45)
    print(
        f"{'Batch size':<20} | {original_memory.batch_size:<10} | {adapted_memory.batch_size:<10}"
    )
    print(
        f"{'Memory size':<20} | {original_memory.memory_size:<10} | {adapted_memory.memory_size:<10}"
    )
    print(f"{'Word size':<20} | {original_memory.word_size:<10} | {adapted_memory.word_size:<10}")
    print(f"{'Num reads':<20} | {original_memory.num_reads:<10} | {adapted_memory.num_reads:<10}")
    print(
        f"{'Num writes':<20} | {original_memory.num_writes:<10} | {adapted_memory.num_writes:<10}"
    )

    # Memory tensor comparison
    print("\nMemory Tensor Comparison:")
    print(f"{'Metric':<20} | {'Original':<25} | {'Adapted':<25}")
    print("-" * 65)
    print(
        f"{'Shape':<20} | {str(original_memory.memory_data.shape):<25} | {str(adapted_memory.state['memory'].shape):<25}"
    )
    print(
        f"{'Size':<20} | {original_memory.memory_data.numel():<25} | {adapted_memory.state['memory'].numel():<25}"
    )
    print(
        f"{'Data type':<20} | {str(original_memory.memory_data.dtype):<25} | {str(adapted_memory.state['memory'].dtype):<25}"
    )
    print(
        f"{'Device':<20} | {str(original_memory.memory_data.device):<25} | {str(adapted_memory.state['memory'].device):<25}"
    )

    # Sample values comparison
    print("\nSample Values Comparison:")
    print("Original memory sample:", original_memory.memory_data[0, 0, :5].tolist())
    print("Adapted memory sample:", adapted_memory.state["memory"][0, 0, :5].tolist())

    # Statistical comparison
    print("\nStatistical Comparison:")
    print(f"{'Metric':<20} | {'Original':<25} | {'Adapted':<25}")
    print("-" * 65)
    print(
        f"{'Mean':<20} | {original_memory.memory_data.mean().item():<25.8f} | {adapted_memory.state['memory'].mean().item():<25.8f}"
    )
    print(
        f"{'Std dev':<20} | {original_memory.memory_data.std().item():<25.8f} | {adapted_memory.state['memory'].std().item():<25.8f}"
    )
    print(
        f"{'Min':<20} | {original_memory.memory_data.min().item():<25.8f} | {adapted_memory.state['memory'].min().item():<25.8f}"
    )
    print(
        f"{'Max':<20} | {original_memory.memory_data.max().item():<25.8f} | {adapted_memory.state['memory'].max().item():<25.8f}"
    )

    # Detailed memory content comparison
    print("\nDetailed Memory Content Comparison:")
    print("First memory location (0,0,:):")
    print("Original:", original_memory.memory_data[0, 0, :10].tolist())
    print("Adapted:", adapted_memory.state["memory"][0, 0, :10].tolist())

    print("\nLast memory location (-1,-1,:):")
    print("Original:", original_memory.memory_data[-1, -1, -10:].tolist())
    print("Adapted:", adapted_memory.state["memory"][-1, -1, -10:].tolist())


def compare_initial_states(original_memory, adapted_memory):
    """Compare initial memory states before any updates."""
    print("\n=== Initial State Comparison ===")

    # Print configuration
    print("\nConfiguration:")
    print(f"{'Parameter':<20} | {'Original':<10} | {'Adapted':<10}")
    print("-" * 45)
    print(
        f"{'Batch size':<20} | {original_memory.batch_size:<10} | {adapted_memory.batch_size:<10}"
    )
    print(
        f"{'Memory size':<20} | {original_memory.memory_size:<10} | {adapted_memory.memory_size:<10}"
    )
    print(f"{'Word size':<20} | {original_memory.word_size:<10} | {adapted_memory.word_size:<10}")

    # Print memory data
    print("\nMemory Data Comparison:")
    print("Original memory sample:", original_memory.memory_data[0, 0, :5].tolist())
    print("Adapted memory sample:", adapted_memory.state["memory"][0, 0, :5].tolist())
    print("Original memory mean:", original_memory.memory_data.mean().item())
    print("Adapted memory mean:", adapted_memory.state["memory"].mean().item())

    # Check if memory is exactly zero
    print("\nZero Initialization Check:")
    print("Original is zero:", torch.all(original_memory.memory_data == 0))
    print("Adapted is zero:", torch.all(adapted_memory.state["memory"] == 0))

    # Check shapes
    print("\nShape Comparison:")
    print("Original shape:", original_memory.memory_data.shape)
    print("Adapted shape:", adapted_memory.state["memory"].shape)

    # Check data types
    print("\nData Type Comparison:")
    print("Original dtype:", original_memory.memory_data.dtype)
    print("Adapted dtype:", adapted_memory.state["memory"].dtype)

    # Check devices
    print("\nDevice Comparison:")
    print("Original device:", original_memory.memory_data.device)
    print("Adapted device:", adapted_memory.state["memory"].device)


def main():
    # Create test dataset
    dataset = RepeatCopy()

    # Create configurations
    controller_config = {
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    }

    memory_config = {
        "memory_size": MEMORY_SIZE,
        "word_size": WORD_SIZE,
        "num_writes": NUM_WRITES,
        "num_reads": NUM_READS,
    }

    # Create original DNC
    original_dnc = OriginalDNC(
        dataset.input_size, dataset.output_size, controller_config, memory_config
    )

    # Create adapted DNC
    adapted_dnc = DNC_Adapted(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        controller_config=controller_config,
        memory_config=memory_config,
    )

    # Print parameter counts
    original_param_count = sum(p.numel() for p in original_dnc.parameters())
    adapted_param_count = sum(p.numel() for p in adapted_dnc.parameters())

    print(f"Original DNC parameter count: {original_param_count}")
    print(f"Adapted DNC parameter count: {adapted_param_count}")

    # Set constant weights for deterministic comparison
    print("\nSetting constant weights...")
    set_constant_weights(original_dnc)
    set_constant_weights(adapted_dnc)

    # Add this to check the initial state immediately after initialization
    print("\n=== Checking True Initial Memory State ===")
    original_dnc.init_state()
    adapted_dnc.init_state()
    print("Original initial memory data mean:", original_dnc.memory.memory_data.mean().item())
    print("Adapted initial memory data mean:", adapted_dnc.memory.state["memory"].mean().item())
    print(f"Original first rows: {original_dnc.memory.memory_data[0][0:2]}")
    print(f"Adapted first rows: {adapted_dnc.memory.memory_data[0][0:2]}")

    # Calculate the sum of the norm of all parameters for each model
    original_param_norm = sum(p.norm() for p in original_dnc.parameters())
    adapted_param_norm = sum(p.norm() for p in adapted_dnc.parameters())

    print(f"Original DNC parameter norm: {original_param_norm}")
    print(f"Adapted DNC parameter norm: {adapted_param_norm}")

    # check the parameters that are not trainable for the two models
    for name, param in original_dnc.named_parameters():
        if not param.requires_grad:
            print(f"{name} is not trainable")

    for name, param in adapted_dnc.named_parameters():
        if not param.requires_grad:
            print(f"{name} is not trainable")

    # Generate test input
    print("Generating test input...")
    inputs, _ = next(dataset.generate(1))

    # Run detailed comparison
    print("\nRunning detailed forward pass comparison...")
    compare_forward_pass(original_dnc, adapted_dnc, inputs)
    print(f"Original first rows: {original_dnc.memory.memory_data[0][0:2]}")
    print(f"Adapted first rows: {adapted_dnc.memory.memory_data[0][0:2]}")

    # Compare outputs
    print("\nComparing outputs...")

    print(f"{original_dnc(inputs)[0]=}")
    print(f"{adapted_dnc(inputs)[0]=}")

    are_equal = torch.allclose(original_dnc(inputs), adapted_dnc(inputs), rtol=1e-5, atol=1e-6)
    print(f"Outputs equal: {are_equal}")

    if not are_equal:
        # Calculate difference
        diff = torch.abs(original_dnc(inputs) - adapted_dnc(inputs))
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        # Show some examples
        print("\nSample outputs:")
        print("Original:", original_dnc(inputs)[0, 0, :5].tolist())
        print("Adapted:", adapted_dnc(inputs)[0, 0, :5].tolist())

    # Verify memory initialization
    verify_memory_initialization(original_dnc.memory, adapted_dnc.memory)

    # Detailed memory initialization comparison
    debug_memory_initialization(original_dnc.memory, adapted_dnc.memory)

    # Compare initial states
    compare_initial_states(original_dnc.memory, adapted_dnc.memory)


if __name__ == "__main__":
    main()
