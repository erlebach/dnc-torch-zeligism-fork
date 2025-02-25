from typing import Dict

import torch
from torch import Tensor

from beartype import beartype
from dnc.base import BaseMemory


@beartype
class DefaultMemory(BaseMemory):
    def __init__(self, memory_size: int, word_size: int, **kwargs):
        """Initialize DefaultMemory.

        Args:
            memory_size: Number of memory locations.
            word_size: Size of each memory word.
            hidden_size: Size of hidden state from controller.
        """
        super().__init__()
        self.memory_size = memory_size
        self.word_size = word_size

        # Define required shapes for interface outputs
        self.required_shapes = {
            "write_vector": (memory_size, word_size),
            # Add other required vectors here
        }

        # Initialize memory state
        self.state["memory"] = torch.zeros(memory_size, word_size)

    def forward(self, state_dict: Dict[str, Tensor]) -> Tensor:
        """Forward pass through the memory module.

        Args:
            state_dict: Dictionary containing the write vector tensor.

        Returns:
            Updated memory tensor.
        """
        write_vector = state_dict["write_vector"]
        print(f"Write vector shape: {write_vector.shape}")
        print(f"Memory shape: {self.state['memory'].shape}")

        self.state["memory"] = self.state["memory"] + write_vector
        return self.state["memory"]


if __name__ == "__main__":
    # Test parameters, initialize memory
    # Using a dictionary as arguments makes it easier to maintain multiple
    # configuration files.
    mem_config = {"memory_size":100, "word_size":16, "hidden_size":256}
    memory = DefaultMemory(**mem_config)

    # Print memory architecture summary
    print(memory.summary())

    # Create test state dictionary
    state_dict = {"write_vector": torch.randn(mem_config["memory_size"], mem_config["word_size"])}

    # Forward pass through memory
    updated_memory = memory(state_dict)
    print(f"\nUpdated memory shape: {updated_memory.shape}")
