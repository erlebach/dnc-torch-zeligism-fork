import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Fix imports to use the correct paths
from dnc_architecture_graves_2016.dnc_module import DNC
from dnc_architecture_graves_2016.memory_config import memory_config
from dnc_architecture_graves_2016.model_utils import (
    synchronize_epsilon_values,
    synchronize_memory_states,
)
from dnc_architecture_graves_2016.repeat_copy import RepeatCopy
from dnc_architecture_graves_2016.training_config import training_config
from dnc_architecture_graves_2016.training_configs import *  # Import constants for backward compatibility
from dnc_torch_zeligism.dnc import DNC as OrigDNC

# Define controller and memory configurations
controller_config = {
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
}

# Combine memory_config with batch_size
memory_config_combined = memory_config.copy()
memory_config_combined["batch_size"] = BATCH_SIZE


def train(dnc, dataset, debug_mode=False):  # Set default to False to avoid immediate errors
    # Initialize optimizer
    optimizer = torch.optim.Adam(dnc.parameters(), lr=LEARNING_RATE)

    # Track losses for plotting
    losses = []
    orig_losses = []

    # For comparison with original implementation if debug_mode is enabled
    if debug_mode:
        try:
            # Import the original DNC implementation safely
            from dnc_torch_zeligism.dnc import DNC as OrigDNC

            # Create configs for original DNC - WITHOUT batch_size
            orig_controller_config = {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
            }
            orig_memory_config = {
                "memory_size": memory_config["memory_size"],
                "word_size": memory_config["word_size"],
                "num_writes": memory_config["num_writes"],
                "num_reads": memory_config["num_reads"],
                # Don't include batch_size for original implementation
            }

            # Create original DNC
            orig_dnc = OrigDNC(
                dataset.input_size, dataset.output_size, orig_controller_config, orig_memory_config
            )
            print("Original DNC initialized for comparison")
        except Exception as e:
            print(f"Could not initialize original DNC for comparison: {e}")
            debug_mode = False  # Disable debug mode if initialization fails

    # Define input and its true output
    start_time = time.time()
    for i, data in enumerate(dataset.generate(NUM_EXAMPLES)):
        # Zero gradients
        optimizer.zero_grad()

        # Unpack input/output
        inputs, true_outputs = data

        # Do a forward pass, compute loss, then do a backward pass
        pred_outputs = dnc(inputs)
        loss = dataset.loss(pred_outputs, true_outputs)
        loss.backward()

        # Update parameters using the optimizer
        optimizer.step()

        # Save loss for plotting
        losses.append(loss.item())

        # If debug mode, compute loss with original implementation
        if debug_mode and i % 100 == 0:  # Do only every 100 iterations to save time
            try:
                with torch.no_grad():
                    orig_outputs = orig_dnc(inputs)
                    orig_loss = dataset.loss(orig_outputs, true_outputs).item()
                    orig_losses.append(orig_loss)
                    print(f"Original DNC loss: {orig_loss:.3f}, New DNC loss: {loss.item():.3f}")
            except Exception as e:
                print(f"Error comparing with original DNC: {e}")

        # Print report when we reach a checkpoint
        if (i + 1) % CHECKPOINT == 0:
            dataset.report(data, pred_outputs.data)
            print("[%d/%d] Loss = %.3f" % (i + 1, NUM_EXAMPLES, loss.item()), flush=True)
            print("Time elapsed = %ds" % (time.time() - start_time))

            # Print the output and target comparison at checkpoints
            print(f"Pred output sample: {pred_outputs[0, 0, :5]}")
            print(f"True output sample: {true_outputs[0, 0, :5]}")

            # Print memory statistics without assuming specific structure
            try:
                if hasattr(dnc, "memory") and hasattr(dnc.memory, "state"):
                    memory_stats = {
                        "memory_mean": dnc.memory.state["memory"].mean().item(),
                        "memory_std": dnc.memory.state["memory"].std().item(),
                        "usage_mean": dnc.memory.usage.mean().item(),
                    }
                    print(f"Memory stats: {memory_stats}")
            except Exception as e:
                print(f"Could not print memory stats: {e}")

    # Plot losses for comparison
    if debug_mode and len(orig_losses) > 0:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(0, len(losses), 100),
                [losses[i] for i in range(0, len(losses), 100)],
                label="New DNC",
            )
            plt.plot(
                range(0, len(losses), 100)[: len(orig_losses)], orig_losses, label="Original DNC"
            )
            plt.xlabel("Iterations (x100)")
            plt.ylabel("Loss")
            plt.title("Training Loss Comparison")
            plt.legend()
            plt.savefig("loss_comparison.png")
            print("Loss comparison saved to loss_comparison.png")
        except Exception as e:
            print(f"Could not plot loss comparison: {e}")

    # Save losses to file for plotting
    with open("new_dnc_losses.txt", "w") as f:
        for loss_val in losses:
            f.write(f"{loss_val}\n")


def main():
    # Set random seed if given
    torch.manual_seed(RANDOM_SEED or torch.initial_seed())

    # Choose dataset and initialize size of data's input and output
    dataset = RepeatCopy()  # default parameters

    # Initialize DNC with our new implementation
    dnc = DNC(
        input_size=dataset.input_size,
        output_size=dataset.output_size,
        controller_config=controller_config,
        memory_config=memory_config_combined,
    )

    # Use train function with debug_mode=False by default
    train(dnc, dataset)


if __name__ == "__main__":
    main()
