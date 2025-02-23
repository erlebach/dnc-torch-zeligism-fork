from typing import Dict

import torch
from torch import Tensor, nn

from .base import BaseMemory
from .interface import DefaultInterface


class DefaultMemory(BaseMemory):
    def __init__(self, memory_size: int, word_size: int, hidden_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size
        self.memory = nn.Parameter(torch.zeros(memory_size, word_size))
        self.interface: DefaultInterface = DefaultInterface(
            hidden_size=hidden_size, interface_size=256
        )
        self.init_state()

    def init_state(self) -> None:
        """Initialize the state of the default memory."""
        self.state = {
            "memory": torch.zeros(self.memory_size, self.word_size),
        }

    def forward(self, state_dict: Dict[str, Tensor]) -> Tensor:
        """Forward pass through the memory module."""
        interface_vectors = self.interface(state_dict)
        self.state["memory"] = self.state["memory"] + interface_vectors["write_vector"]
        return self.state["memory"]


if __name__ == "__main__":
    # Test parameters
    batch_size = 4
    hidden_size = 64
    memory_size = 100
    word_size = 16

    # Initialize memory and interface
    memory = DefaultMemory(memory_size=memory_size, word_size=word_size, hidden_size=hidden_size)
    interface = DefaultInterface(hidden_size=hidden_size, interface_size=256)
    memory.set_interface(interface)

    # Create random state dictionary
    state_dict = {"hidden_state": torch.randn(batch_size, hidden_size)}

    # Forward pass through memory
    updated_memory = memory(state_dict)

    # Print results
    print("\nDefaultMemory Test Results:")
    print(f"State dictionary shape: {state_dict['hidden_state'].shape}")
    print(f"Updated memory shape: {updated_memory.shape}")
    print(f"First memory sample:\n{updated_memory[0]}")
