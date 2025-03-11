# rms_norm_layer.py

import torch
import torch.nn as nn

class RMSNormLayer(nn.Module):
    """RMS Normalization Layer."""

    def __init__(self, normalized_shape: int, epsilon: float = 1e-5):
        """
        Initialize the RMSNorm layer.

        Args:
            normalized_shape: The shape of the input tensor to be normalized.
            epsilon: A small value to avoid division by zero.
        """
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, features).

        Returns:
            Normalized tensor of the same shape as input.
        """
        # Compute the root mean square
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        # Normalize and scale
        return self.scale * x / rms

if __name__ == "__main__":
    # Test the RMSNormLayer
    batch_size = 8
    seq_length = 5
    features = 10

    # Create an RMSNorm layer
    rms_norm = RMSNormLayer(normalized_shape=features)

    # Generate random input
    x = torch.randn(seq_length, batch_size, features)

    # Forward pass
    y = rms_norm(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"Output sample:\n{y[0, 0, :]}")
