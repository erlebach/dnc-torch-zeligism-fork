from typing import Any, Dict

import torch
from torch import Tensor, nn

from dnc.base import BaseController


class LSTMController(BaseController):
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # Batch-first
        self.init_state({"batch_size": 1})  # Initialize state with default batch_size=1

    def init_state(self, config: Dict[str, Any]) -> None:
        """Initialize the state of the LSTM controller.

        Args:
            **config: Configuration parameters for state initialization.
                     Must include 'batch_size'.
        """
        batch_size: int = config["batch_size"]  # Raises KeyError if 'batch_size' is missing
        self.state = {
            "hidden_state": torch.zeros(
                self.lstm.num_layers, batch_size, self.hidden_size
            ),  # (num_layers, batch_size, hidden_size)
            "cell_state": torch.zeros(
                self.lstm.num_layers, batch_size, self.hidden_size
            ),  # (num_layers, batch_size, hidden_size)
        }

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the LSTM controller.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Hidden state tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Get batch size from input tensor
        batch_size: int = x.size(0)

        # I DON'T THINK THIS IS NEEDED. STUDY FURTHER.
        # Reinitialize state if batch size has changed
        if self.state is None or self.state["hidden_state"].size(1) != batch_size:
            self.init_state({"batch_size": batch_size})  # use dict arg for better type inference

        # Use temporary variables for readability
        hidden = self.state["hidden_state"]
        cell = self.state["cell_state"]

        # Update state dictionary directly
        output, (self.state["hidden_state"], self.state["cell_state"]) = self.lstm(
            x, (hidden, cell)
        )

        # Return the output (hidden state for each time step)
        return output


if __name__ == "__main__":
    # Test parameters
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 64

    # Initialize controller
    controller = LSTMController(input_size=input_size, hidden_size=hidden_size)

    # Create random input tensor with shape (batch_size, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)

    # Forward pass through controller
    hidden_state = controller(x)

    # Print results
    print("\nLSTMController Test Results:")
    print(f"Input shape: {x.shape}")
    print(f"Hidden state shape: {hidden_state.shape}")
    print(f"First hidden state sample:\n{hidden_state[0]}")
