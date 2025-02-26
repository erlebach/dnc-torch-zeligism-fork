import time

import torch
import torch.nn.functional as F

# Fix imports to use the correct paths
from dnc_architecture_graves_2016.dnc_module import DNC
from dnc_architecture_graves_2016.memory_config import memory_config
from dnc_architecture_graves_2016.repeat_copy import RepeatCopy
from dnc_architecture_graves_2016.training_config import training_config
from dnc_architecture_graves_2016.training_configs import *  # Import constants for backward compatibility

# Define controller and memory configurations
controller_config = {
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
}

# Combine memory_config with batch_size
memory_config_combined = memory_config.copy()
memory_config_combined["batch_size"] = BATCH_SIZE


def train(dnc, dataset):
    # Initialize optimizer - use only one optimizer, not two
    optimizer = torch.optim.Adam(dnc.parameters(), lr=LEARNING_RATE)

    # Track losses for plotting
    losses = []

    # Define input and its true output
    start_time = time.time()
    for i, data in enumerate(dataset.generate(NUM_EXAMPLES)):
        # Zero gradients
        optimizer.zero_grad()

        # Unpack input/output and turn them into variables
        inputs, true_outputs = data

        # Do a forward pass, compute loss, then do a backward pass
        pred_outputs = dnc(inputs)
        loss = dataset.loss(pred_outputs, true_outputs)
        loss.backward()

        # Update parameters using the optimizer
        optimizer.step()

        # Save loss for plotting
        losses.append(loss.item())

        # Print report when we reach a checkpoint
        if (i + 1) % CHECKPOINT == 0:
            dataset.report(data, pred_outputs.data)
            # dnc.debug()
            print("[%d/%d] Loss = %.3f" % (i + 1, NUM_EXAMPLES, loss.item()), flush=True)
            print("Time elapsed = %ds" % (time.time() - start_time))

    # Save losses to file for plotting
    with open("o.txt", "w") as f:
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

    train(dnc, dataset)


if __name__ == "__main__":
    main()
