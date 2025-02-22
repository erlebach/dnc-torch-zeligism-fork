"""My own version of repeat copy."""

from typing import Generator

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from training_configs import config

"""
This is my own version of repeat copy.
"""


class RepeatCopy:
    """A class for generating and processing repeat-copy tasks.

    The repeat-copy task involves memorizing a sequence of bits and repeating it
    a specified number of times. This class generates training examples, calculates
    loss, and provides reporting functionality for this task.

    Attributes:
        num_bits: Number of bits in each sequence.
        min_length: Minimum length of the bit sequence.
        max_length: Maximum length of the bit sequence.
        min_repeats: Minimum number of repetitions.
        max_repeats: Maximum number of repetitions.
        input_size: Size of the input vector.
        output_size: Size of the output vector.
        inputs_lengths: Lengths of input sequences from the last example.

    Methods:
        __init__: Initializes the RepeatCopy task configuration.
        generate: Generates multiple examples using example().
        example: Creates the next repeat-copy example.
        loss: Calculates a refined loss between predicted and true outputs.
        report: Prints a report from data produced by example().

    """

    def __init__(
        self,
        num_bits: int = 4,
        min_length: int = 1,
        max_length: int = 3,
        min_repeats: int = 1,
        max_repeats: int = 3,
        config: dict = config,
    ) -> None:
        """Initialize the RepeatCopy task configuration.

        Args:
            num_bits: Number of bits in each sequence. Must be positive.
            min_length: Minimum length of the bit sequence. Must be between 1 and max_length.
            max_length: Maximum length of the bit sequence. Must be >= min_length.
            min_repeats: Minimum number of repetitions. Must be between 0 and max_repeats.
            max_repeats: Maximum number of repetitions. Must be >= min_repeats.

        Raises:
            AssertionError: If any of the input constraints are violated.

        """

        class NonPositiveBitsError(ValueError):
            """Error for when number of bits is not positive."""

        class InvalidLengthError(ValueError):
            """Error for when sequence length constraints are violated."""

        # Check for obvious errors and save configs
        if num_bits <= 0:
            raise NonPositiveBitsError
        if min_length < 1 or min_length > max_length:
            raise InvalidLengthError
        if min_repeats < 0 or min_repeats > max_repeats:
            raise InvalidLengthError

        self.num_bits = num_bits
        self.min_length = min_length
        self.max_length = max_length
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        self.config = config

        # Input and output sizes are fixed for the given parameters
        self.input_size = num_bits + 2 + max_repeats - min_repeats
        self.output_size = num_bits + 1

        # This variable should hold the inputs' lengths from the last example
        self.inputs_lengths = torch.zeros(1)

    def generate(self, num_examples: int) -> Generator:
        """Generate `num_examples` examples using `example()`."""
        for _ in range(num_examples):
            yield self.example()

    def example(self) -> tuple[Tensor, Tensor]:
        """Fetche/create the next repeat-copy example.

        Also Updates the lengths of the bits in each batch element.

        """
        # Index for the start marker
        start_channel = self.num_bits
        batch_size = config["batch_size"]

        # Get the length of observations and repeats
        bits_lengths = torch.IntTensor(batch_size).random_(self.min_length, self.max_length + 1)
        repeats = torch.IntTensor(batch_size).random_(self.min_repeats, self.max_repeats + 1)

        # Total sequence length is input bits + repeats * bits + channels
        seq_length = torch.max(bits_lengths + repeats * bits_lengths + 3).item()
        # Fill inputs and outputs with zeros
        inputs = torch.zeros(seq_length, batch_size, self.input_size)
        outputs = torch.zeros(seq_length, batch_size, self.output_size)

        for i in range(batch_size):
            # Handy sequence indices to improve readability
            obs_end = bits_lengths[i] + 1
            target_start = bits_lengths[i] + 2
            target_end = target_start + repeats[i] * bits_lengths[i]

            # Create `num_bits` random binary bits of length `obs_length`
            bits = torch.bernoulli(0.5 * torch.ones(bits_lengths[i].item(), self.num_bits))

            # Inputs starts with a marker at `start_channel`
            inputs[0, i, start_channel] = 1
            # Then the observation bits follow (from idx 0 up to start channel)
            inputs[1:obs_end, i, :start_channel] = bits
            # Finally, we activate the appropriate repeat channel
            # (Note that smallest index in repeat channel, which is start_channel + 1,
            # is for min_repeats, not 0 repeats.)
            repeats_active_channel = start_channel + 1 + repeats[i] - self.min_repeats
            inputs[obs_end, i, repeats_active_channel] = 1

            # Fill output up to repeats of bits
            outputs[target_start:target_end, i, 1:] = bits.repeat(repeats[i], 1)
            outputs[target_end, i, 0] = 1

        # Record inputs' lengths of this example
        self.inputs_lengths = bits_lengths + 2

        return inputs, outputs

    def loss(self, pred_outputs: Tensor, true_outputs: Tensor) -> Tensor:
        """Calculate a more refined loss.

        Calculates a more refined loss(or distance if you like)
        between the true outputs and the predicted outputs.

        Here, we use a simple Euclidean distance for each time step indpendently,
        and then sum up the distances for all the relevant time steps.
        The output we get while receiving the input is irrelevant, so there is
        no loss in predicting it wrong (i.e. there isn't a true output).

        """
        inputs_lengths = self.inputs_lengths
        batch_size = self.config["batch_size"]

        # Clean predictions made during input
        for i in range(batch_size):
            pred_outputs[: inputs_lengths[i], i, :] = 0

        # Calculate the accumulated MSE Loss for all time steps
        loss = 0
        for t in range(true_outputs.size()[0]):
            loss += F.mse_loss(pred_outputs[t, ...], true_outputs[t, ...])

        return loss

    def report(self, data, pred_outputs):
        """Print a report from data from example().

        Prints a simple report given the `data` produced by the last call of
        `example()` and the predicted output of the DNC.
        Shows a random input/output example from the batch.
        We show the output rounded to the nearest integer as well.
        Finally, we calculate the number of mispredicted bits (after rounding).

        """
        inputs, true_outputs = data
        inputs_lengths = self.inputs_lengths
        batch_size = self.config["batch_size"]
        # Pick a random batch number
        i = torch.IntTensor(1).random_(1, batch_size).item()

        # Show the true outputs and the (rounded) predictions
        print()
        print("-----------------------------------")
        print(inputs[: inputs_lengths[i], i, :])
        print()
        print("Expected:")
        print(true_outputs[inputs_lengths[i] :, i, :])
        print()
        print("Got:")
        print(pred_outputs[inputs_lengths[i] :, i, :].round().abs())
        print()
        print("Got (without rounding):")
        print(pred_outputs[inputs_lengths[i] :, i, :])
        print()

        # Print the number of mispredicted bits
        bits_miss = (pred_outputs.round() - true_outputs).abs().sum().item()
        bits_total = self.output_size * inputs_lengths.sum().item()
        bits_hits = bits_total - bits_miss
        print("hits =", int(bits_hits), "out of", int(bits_total))
        print("Accuracy = %.2f%%" % (100 * bits_hits / bits_total))
        print()
