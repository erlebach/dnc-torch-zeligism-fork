from typing import Any, Dict

import torch
import torch.nn.functional as F
from beartype import beartype
from torch import Tensor, nn

from dnc.base import BaseInterface


@beartype
class Interface(BaseInterface):
    """
    Interface module for the DNC architecture based on Graves et al. 2016.

    This class decomposes the controller's output into various interface vectors
    required by the memory module for read/write operations.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int = 128,
        word_size: int = 20,
        num_writes: int = 1,
        num_reads: int = 1,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the interface module.

        Args:
            input_size: Size of the input vector from the controller
            memory_size: Number of memory locations
            word_size: Size of each memory word
            num_writes: Number of write heads
            num_reads: Number of read heads
            batch_size: Batch size for processing
        """
        super().__init__()

        # Store configuration
        self.input_size = input_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_writes = num_writes
        self.num_reads = num_reads
        self.batch_size = batch_size

        # Calculate the number of read modes (backward, forward, content)
        # For each write head, we have a backward and forward mode
        # Plus one content-based addressing mode
        self.num_read_modes = 2 * num_writes + 1

        # Calculate the total size of all interface vectors
        self.interface_size = (
            # Read keys: num_reads * word_size
            num_reads * word_size
            +
            # Read strengths: num_reads
            num_reads
            +
            # Write keys: num_writes * word_size
            num_writes * word_size
            +
            # Write strengths: num_writes
            num_writes
            +
            # Erase vectors: num_writes * word_size
            num_writes * word_size
            +
            # Write vectors: num_writes * word_size
            num_writes * word_size
            +
            # Free gates: num_reads
            num_reads
            +
            # Allocation gates: num_writes
            num_writes
            +
            # Write gates: num_writes
            num_writes
            +
            # Read modes: num_reads * num_read_modes
            num_reads * self.num_read_modes
        )

        # Linear layer to transform controller output to interface vectors
        self.fc = nn.Linear(input_size, self.interface_size)

        # Define the output shapes for validation
        self.output_shapes = {
            "read_keys": (batch_size, num_reads, word_size),
            "read_strengths": (batch_size, num_reads),
            "write_keys": (batch_size, num_writes, word_size),
            "write_strengths": (batch_size, num_writes),
            "erase_vectors": (batch_size, num_writes, word_size),
            "write_vectors": (batch_size, num_writes, word_size),
            "free_gate": (batch_size, num_reads),
            "allocation_gate": (batch_size, num_writes),
            "write_gate": (batch_size, num_writes),
            "read_modes": (batch_size, num_reads, self.num_read_modes),
        }

        # Initialize state
        self.init_state()

    def init_state(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the state of the interface.

        Args:
            config: Optional configuration dictionary
        """
        # The interface doesn't maintain state between time steps
        self.state = {}

    def forward(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Decompose the controller's state into interface vectors.

        Args:
            state_dict: Dictionary containing the controller's state
                        Expected to have a key 'output' with the controller output

        Returns:
            Dictionary of interface vectors for the memory module
        """
        # Get the controller output
        controller_output = state_dict["output"]

        # Transform controller output to interface vectors
        interface_vector = self.fc(controller_output)

        # Split the interface vector into its components
        return self._parse_interface_vector(interface_vector)

    def _parse_interface_vector(self, interface_vector: Tensor) -> Dict[str, Tensor]:
        """
        Parse the flat interface vector into its component vectors.

        Args:
            interface_vector: Flat vector containing all interface components

        Returns:
            Dictionary of parsed interface vectors
        """
        batch_size = interface_vector.size(0)

        # Initialize index for slicing the interface vector
        idx = 0

        # Read keys
        read_keys_size = self.num_reads * self.word_size
        read_keys = interface_vector[:, idx : idx + read_keys_size]
        read_keys = read_keys.view(batch_size, self.num_reads, self.word_size)
        idx += read_keys_size

        # Read strengths
        read_strengths_size = self.num_reads
        read_strengths = interface_vector[:, idx : idx + read_strengths_size]
        read_strengths = F.softplus(read_strengths)
        idx += read_strengths_size

        # Write keys
        write_keys_size = self.num_writes * self.word_size
        write_keys = interface_vector[:, idx : idx + write_keys_size]
        write_keys = write_keys.view(batch_size, self.num_writes, self.word_size)
        idx += write_keys_size

        # Write strengths
        write_strengths_size = self.num_writes
        write_strengths = interface_vector[:, idx : idx + write_strengths_size]
        write_strengths = F.softplus(write_strengths)
        idx += write_strengths_size

        # Erase vectors
        erase_vectors_size = self.num_writes * self.word_size
        erase_vectors = interface_vector[:, idx : idx + erase_vectors_size]
        erase_vectors = erase_vectors.view(batch_size, self.num_writes, self.word_size)
        erase_vectors = torch.sigmoid(erase_vectors)
        idx += erase_vectors_size

        # Write vectors
        write_vectors_size = self.num_writes * self.word_size
        write_vectors = interface_vector[:, idx : idx + write_vectors_size]
        write_vectors = write_vectors.view(batch_size, self.num_writes, self.word_size)
        idx += write_vectors_size

        # Free gate
        free_gate_size = self.num_reads
        free_gate = interface_vector[:, idx : idx + free_gate_size]
        free_gate = torch.sigmoid(free_gate)
        idx += free_gate_size

        # Allocation gate
        allocation_gate_size = self.num_writes
        allocation_gate = interface_vector[:, idx : idx + allocation_gate_size]
        allocation_gate = torch.sigmoid(allocation_gate)
        idx += allocation_gate_size

        # Write gate
        write_gate_size = self.num_writes
        write_gate = interface_vector[:, idx : idx + write_gate_size]
        write_gate = torch.sigmoid(write_gate)
        idx += write_gate_size

        # Read modes
        read_modes_size = self.num_reads * self.num_read_modes
        read_modes = interface_vector[:, idx : idx + read_modes_size]
        read_modes = read_modes.view(batch_size, self.num_reads, self.num_read_modes)
        read_modes = F.softmax(read_modes, dim=2)

        # Return all interface vectors in a dictionary
        return {
            "read_keys": read_keys,
            "read_strengths": read_strengths,
            "write_keys": write_keys,
            "write_strengths": write_strengths,
            "erase_vectors": erase_vectors,
            "write_vectors": write_vectors,
            "free_gate": free_gate,
            "allocation_gate": allocation_gate,
            "write_gate": write_gate,
            "read_modes": read_modes,
        }


if __name__ == "__main__":
    # Test parameters
    batch_size = 4
    hidden_size = 64
    interface_size = 256

    # Initialize interface
    interface = Interface(input_size=hidden_size)

    # Create random state dictionary
    state_dict = {"output": torch.randn(batch_size, hidden_size)}

    # Forward pass through interface
    interface_vectors = interface(state_dict)

    # Print results
    print("\nInterface Test Results:")
    print(f"State dictionary shape: {state_dict['output'].shape}")
    print(f"Interface vectors: {interface_vectors}")

    # Summary
    print(interface.summary())


def test_interface_with_memory():
    """Test the Interface class with the Memory module."""
    import torch

    from dnc_architecture_graves_2016.memory import Memory
    from dnc_architecture_graves_2016.memory_config import memory_config
    from dnc_architecture_graves_2016.training_config import training_config

    # Combine configurations
    config = {**memory_config, **training_config}

    # Create memory module - no need to pass batch_size separately
    memory = Memory(**config)

    # Create interface module
    # Assuming controller output size is 128
    controller_output_size = 128
    interface = Interface(
        input_size=controller_output_size,
        memory_size=config["memory_size"],
        word_size=config["word_size"],
        num_writes=config["num_writes"],
        num_reads=config["num_reads"],
        batch_size=config["batch_size"],
    )

    # Set interface for memory (optional validation)
    memory.set_interface(interface)

    # Create dummy controller output
    controller_output = torch.randn(config["batch_size"], controller_output_size)

    # Process through interface
    interface_vectors = interface({"output": controller_output})

    # Process through memory
    print(f"{interface_vectors=}")
    for name in interface_vectors:
        print(f"vector name: {name}")
    read_words = memory(interface_vectors)  # ERROR

    print(f"Read words shape: {read_words.shape}")
    # Should be (batch_size, num_reads, word_size)

    return read_words


if __name__ == "__main__":
    test_interface_with_memory()
