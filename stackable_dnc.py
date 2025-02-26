import torch
import torch.nn as nn

# import torch.nn.functional as F
from beartype import beartype

# from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from dnc import DNC
from repeat_copy import RepeatCopy

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
        self, inputs: Float[Tensor, "batch seq_len input_size"]
    ) -> Float[Tensor, "batch seq_len output_size"]:
        """Forward pass of the StackableDNC.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size + input_size) if pass_through
            else (batch_size, seq_len, output_size)

        """
        # Transpose input to (seq_len, batch_size, input_size) for DNC processing
        inputs_t = inputs.transpose(0, 1)

        # Process through DNC
        dnc_output = self.dnc(inputs_t)

        # Transpose back to (batch_size, seq_len, output_size)
        dnc_output = dnc_output.transpose(0, 1)

        if self.pass_through:
            # Concatenate original input with DNC output along feature dimension
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
            # Calculate the actual input size considering pass_through
            actual_input_size = layer_sizes[i] if i == 0 else layers[-1].output_size

            layers.append(
                StackableDNC(
                    input_size=actual_input_size,
                    output_size=layer_sizes[i + 1],
                    pass_through=(i < len(layer_sizes) - 2),  # No pass-through for last layer
                    **kwargs,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: Float[Tensor, "batch seq_len input_size"]
    ) -> Float[Tensor, "batch seq_len output_size"]:
        """Forward pass through all DNC layers.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, sequence_length, output_size)

        """
        print("INPUT forward")
        for layer in self.layers:
            print("layer: ", layer)
            x = layer(x)
        return x


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Import necessary components
    from training_configs import config, controller_config, memory_config

    # Test parameters
    seq_len = 15
    input_size = 8
    hidden_size = 6
    output_size = 5

    # Test single StackableDNC layer
    print("\nTesting single StackableDNC:")
    stackable_dnc = StackableDNC(
        input_size=input_size,
        output_size=output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        config=config,
    )

    # Create input tensor with shape (batch_size, seq_len, input_size)
    x = torch.randn(config["batch_size"], seq_len, input_size)

    y = stackable_dnc(x)
    print(f"Input shape: {x.shape}")  # Should be (batch_size, 15, 8)
    print(f"Output shape: {y.shape}")  # Should be (batch_size, 15, 13) due to pass_through
    print("StackableDNC works")

    # Test multi-layer DNC
    print("\nTesting MultiLayerDNC:")
    multi_dnc = MultiLayerDNC(
        input_size=input_size,
        hidden_sizes=[hidden_size, 2 * hidden_size, hidden_size],
        output_size=output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        config=config,
    )

    y_multi = multi_dnc(x)
    print(f"Input shape: {x.shape}")  # Should be (batch_size, 15, 8)
    print(f"Output shape: {y_multi.shape}")  # Should be (batch_size, 15, 5)
    print("MultiLayerDNC works")
