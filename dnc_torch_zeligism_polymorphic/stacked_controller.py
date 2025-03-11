# stacked_controller.py

from typing import List

import torch
import torch.nn as nn

from dnc_torch_zeligism_polymorphic.dnc_adapted import DNC_Adapted
from dnc_torch_zeligism_polymorphic.rms_norm_layer import RMSNormLayer


class StackedController(nn.Module):
    """Stacked controller managing multiple DNC_Adapted instances with optional RMS normalization."""

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        output_size: int,
        controller_config: dict,
        memory_config: dict,
        use_rms_norm: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_rms_norm = use_rms_norm
        self.input_size = input_size
        self.output_size = output_size

        # Initialize layers
        self.layers = nn.ModuleList()

        # First layer takes the original input size
        self.layers.append(DNC_Adapted(input_size, output_size, controller_config, memory_config))

        # Subsequent layers take the output size as input
        for _ in range(1, num_layers):
            self.layers.append(
                DNC_Adapted(output_size, output_size, controller_config, memory_config)
            )

        # Initialize RMSNorm layers if needed
        if use_rms_norm:
            self.rms_norms = nn.ModuleList([RMSNormLayer(output_size) for _ in range(num_layers)])

        # Projection layer for the first residual connection (if input_size != output_size)
        self.needs_projection = input_size != output_size
        if self.needs_projection:
            self.projection = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stacked controller.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, input_size).

        Returns:
            Output tensor of shape (sequence_length, batch_size, output_size).
        """
        # Process through the first layer
        residual = x
        x = self.layers[0](x)

        # Apply RMS normalization if enabled
        if self.use_rms_norm:
            x = self.rms_norms[0](x)

        # Add residual connection with projection if needed
        if self.needs_projection:
            x = x + self.projection(residual)
        elif self.input_size == self.output_size:
            x = x + residual

        # Process through subsequent layers
        for i in range(1, self.num_layers):
            residual = x
            x = self.layers[i](x)

            # Apply RMS normalization if enabled
            if self.use_rms_norm:
                x = self.rms_norms[i](x)

            # Add residual connection (dimensions already match)
            x = x + residual

        return x


#----------------------------------------------------------------------
if __name__ == "__main__":
    # Test the StackedController
    stacked_config = {
        "input_size": 10,
        "output_size": 5,
        "num_layers": 3
    }
    controller_config = {"hidden_size": 64, "num_layers": 1}
    memory_config = {
        "memory_size": 128,
        "word_size": 20,
        "num_reads": 4,
        "num_writes": 1,
        "batch_size": 8,
    }

    # Create a stacked controller
    model = StackedController(
        **stacked_config,
        controller_config=controller_config,
        memory_config=memory_config,
        use_rms_norm=True,
    )

    # Generate random input sequence
    seq_length = 5
    batch_size = 8
    input_size = stacked_config['input_size']
    x = torch.randn(seq_length, batch_size, input_size)

    # Forward pass
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
