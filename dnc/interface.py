from typing import Dict

import torch
from torch import Tensor, nn

from .base import BaseInterface


class DefaultInterface(BaseInterface):
    def __init__(self, hidden_size: int, interface_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.interface_size = interface_size
        self.interface_projection = nn.Linear(hidden_size, interface_size)
        self.init_state()

    def init_state(self) -> None:
        """Initialize the state of the default interface."""
        self.state = {
            "projection": torch.zeros(1, self.interface_size),
        }

    def forward(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Decompose the controller's state into interface vectors."""
        hidden_state = state_dict["hidden_state"]
        interface_vectors = self.interface_projection(hidden_state)

        return {
            "read_keys": interface_vectors[:, : self.hidden_size],
            "write_key": interface_vectors[:, self.hidden_size : 2 * self.hidden_size],
            "erase_vector": interface_vectors[:, 2 * self.hidden_size : 3 * self.hidden_size],
            "write_vector": interface_vectors[:, 3 * self.hidden_size :],
        }


if __name__ == "__main__":
    # Test parameters
    batch_size = 4
    hidden_size = 64
    interface_size = 256

    # Initialize interface
    interface = DefaultInterface(hidden_size=hidden_size, interface_size=interface_size)

    # Create random state dictionary
    state_dict = {"hidden_state": torch.randn(batch_size, hidden_size)}

    # Forward pass through interface
    interface_vectors = interface(state_dict)

    # Print results
    print("\nDefaultInterface Test Results:")
    print(f"State dictionary shape: {state_dict['hidden_state'].shape}")
    print(f"Interface vectors: {interface_vectors}")
