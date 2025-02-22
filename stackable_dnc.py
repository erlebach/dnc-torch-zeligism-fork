import torch
import torch.nn as nn

# import torch.nn.functional as F
from beartype import beartype

# from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from dnc import DNC

# from memory import Memory
from training_configs import config, controller_config, memory_config


@beartype
class StackableDNC(nn.Module):
    """A stackable version of the DNC that can be used in sequence.

    Args:
        input_size: Size of the input vector.
        output_size: Size of the output vector.
        controller_config: Configuration for the controller.
        memory_config: Configuration for the memory module.
        config: General configuration parameters.
        controller: Controller class to use (default: nn.LSTM).
        pass_through: Whether to pass the input alongside the output (default: True).

    The shape using Einstein notation where:
        S: sequence length
        B: batch size
        I: input dimension
        O: output dimension

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        controller_config: dict = controller_config,
        memory_config: dict = memory_config,
        config: dict = config,
        controller: type[nn.Module] = nn.LSTM,
        pass_through: bool = True,
    ) -> None:
        super().__init__()

        self.dnc = DNC(
            input_size=input_size,
            output_size=output_size,
            controller_config=controller_config,
            memory_config=memory_config,
            config=config,
            controller=controller,
        )

        self.pass_through = pass_through
        if pass_through:
            self.output_size = output_size + input_size
        else:
            self.output_size = output_size

    def forward(
        self,
        inputs: Float[Tensor, "S B I"],
    ) -> Float[Tensor, "S B O"]:
        """Forward pass through the stackable DNC layer.

        Args:
            inputs: Input tensor of shape (sequence_size, batch_size, input_size)

        Returns:
            Output tensor of shape (sequence_size, batch_size, output_size)
            If pass_through is True, the output includes the original input concatenated

        """
        dnc_output = self.dnc(inputs)

        if self.pass_through:
            # Concatenate the DNC output with the original input
            return torch.cat([dnc_output, inputs], dim=-1)
        return dnc_output


# Example usage:
class MultiLayerDNC(nn.Module):
    """Multiple stacked DNC layers.

    Args:
        input_size: Size of the input vector.
        hidden_sizes: List of hidden layer sizes.
        output_size: Size of the final output vector.
        num_layers: Number of DNC layers to stack.

    """

    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int, **kwargs
    ) -> None:
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(
                StackableDNC(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    pass_through=(i < len(layer_sizes) - 2),  # No pass-through for last layer
                    **kwargs,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Float[Tensor, "S B I"]) -> Float[Tensor, "S B O"]:
        """Forward pass through all DNC layers.

        Args:
            x: Input tensor of shape (sequence_size, batch_size, input_size)

        Returns:
            Output tensor of shape (sequence_size, batch_size, output_size)

        """
        for layer in self.layers:
            x = layer(x)
        return x
