import time

import torch
from dnc import DNC
from repeat_copy import RepeatCopy
from torch import Tensor
from training_configs import config, controller_config, memory_config


def train(dnc: DNC, dataset: RepeatCopy, config: dict) -> None:
    # Initialize optimizer and loss function
    optimizer = torch.optim.SGD(
        dnc.parameters(), lr=config["learning_rate"], momentum=config["momentum"]
    )
    # Adam seems to be faster (maybe)
    optimizer = torch.optim.Adam(dnc.parameters())

    # Define input and its true output
    start_time = time.time()
    for i, data in enumerate(dataset.generate(config["num_examples"])):
        # Zero gradients
        optimizer.zero_grad()

        # Unpack input/output and turn them into variables
        inputs, true_outputs = data

        # Do a forward pass, compute loss, then do a backward pass
        pred_outputs = dnc(inputs)
        loss: Tensor = dataset.loss(pred_outputs, true_outputs)
        loss.backward()

        # Update parameters using the optimizer
        optimizer.step()

        # Print report when we reach a checkpoint
        if (i + 1) % config["checkpoint"] == 0:
            dataset.report(data, pred_outputs.data)
            # dnc.debug()
            print(f"[{i+1}/{config['num_examples']}] Loss = {loss.item()}")
            print(f"Time elapsed = {time.time() - start_time}")


def main(config: dict) -> None:
    # Set random seed if given
    torch.manual_seed(config["random_seed"] or torch.initial_seed())

    # Choose dataset and initialize size of data's input and output
    dataset = RepeatCopy(config=config)  # default parameters

    # Initialize DNC
    dnc = DNC(dataset.input_size, dataset.output_size, controller_config, memory_config)

    train(dnc, dataset, config)


if __name__ == "__main__":
    main(config)
