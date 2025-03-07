"""
Adaptation of the original DNC_InterfaceLayer class from dnc_torch_zeligism to use the polymorphic structure
defined in base.py while preserving the original functionality.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from dnc.base import BaseInterface

@beartype
class DNC_InterfaceLayer_Adapted(BaseInterface):
    """
    Adaptation of the original DNC_InterfaceLayer class to inherit from BaseInterface.
    """

    def __init__(self, input_size, memory_size, word_size, num_writes, num_reads, batch_size=None):
        super().__init__()

        # Initialize state dict
        self.state = {}

        # Store sizes
        self.input_size = input_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_writes = num_writes
        self.num_reads = num_reads
        self.batch_size = batch_size

        # Calculate number of read modes (content + forward/backward for each write head)
        self.num_read_modes = 1 + 2 * num_writes

        # Set output shapes
        self.output_shapes = {
            "read_keys": (num_reads, word_size),
            "read_strengths": (num_reads,),
            "write_keys": (num_writes, word_size),
            "write_strengths": (num_writes,),
            "erase_vectors": (num_writes, word_size),
            "write_vectors": (num_writes, word_size),
            "free_gate": (num_reads,),
            "allocation_gate": (num_writes,),
            "write_gate": (num_writes,),
            "read_modes": (num_reads, self.num_read_modes),
        }

        # Set input shape for summary
        self.input_shape = (None, input_size)

        # Read and write keys and their strengths
        self.read_keys = LinearView(input_size, [num_reads, word_size])
        self.read_strengths = LinearView(input_size, [num_reads])
        self.write_keys = LinearView(input_size, [num_writes, word_size])
        self.write_strengths = LinearView(input_size, [num_writes])

        # Erase and write (i.e. overwrite) vectors
        self.erase_vectors = LinearView(input_size, [num_writes, word_size])
        self.write_vectors = LinearView(input_size, [num_writes, word_size])

        # Free, allocation, and write gates
        self.free_gate = LinearView(input_size, [num_reads])
        self.allocation_gate = LinearView(input_size, [num_writes])
        self.write_gate = LinearView(input_size, [num_writes])

        # Read modes
        self.read_modes = LinearView(input_size, [num_reads, self.num_read_modes])

    def init_state(self) -> None:
        """Initialize interface state."""
        # The original interface doesn't have any state to initialize
        self.state = {}

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the interface.

        Args:
            state_dict: Dictionary containing the controller's state.

        Returns:
            Dictionary of interface vectors.
        """
        # Extract controller output from state_dict
        x = state_dict["output"]

        return {
            "read_keys": self.read_keys(x),
            "read_strengths": self.read_strengths(x),  # debugging value
            #"read_strengths": F.softplus(
                #self.read_strengths(x) + 1    ### Error
            #),  # Add 1 and apply softplus for stability
            "write_keys": self.write_keys(x),   
            "write_strengths": self.write_strengths(x),   # debugging value
            #"write_strengths": F.softplus(
                #self.write_strengths(x) + 1   ### Error
            #),  # Add 1 and apply softplus for stability
            "erase_vectors": torch.sigmoid(self.erase_vectors(x)),
            "write_vectors": torch.sigmoid(self.write_vectors(x)),
            "free_gate": torch.sigmoid(self.free_gate(x)),
            "allocation_gate": torch.sigmoid(self.allocation_gate(x)),
            "write_gate": torch.sigmoid(self.write_gate(x)),
            "read_modes": F.softmax(self.read_modes(x), dim=2),
        }


@beartype
class LinearView(nn.Module):
    """
    Original LinearView implementation (unchanged).

    Similar to linear, except that it outputs a tensor with size `dim`.
    It is assumed that the first dimension is the batch dimension.
    """

    def __init__(self, input_size, output_view):
        super().__init__()
        # Calculate output size (just the product of dims in output_view)
        output_size = 1
        for dim in output_view:
            output_size *= dim
        # Define the layer and the desired view of the output
        self.layer = nn.Linear(input_size, output_size)
        self.output_view = output_view

    def forward(self, x):
        # -1 because we assume batch dimension exists
        return self.layer(x).view(-1, *self.output_view)


if __name__ == "__main__":
    # Test parameters
    batch_size = 4 # W: Constant name "batch_size" doesn't conform to UPPER_CASE naming style
    hidden_size = 64 # W: Constant name "hidden_size" doesn't conform to UPPER_CASE naming style
    interface_size = 256 # W: Constant name "interface_size" doesn't conform to UPPER_CASE naming style

    def __init__(self, input_size, memory_size, word_size, num_writes, num_reads, batch_size=None):

    # Initialize interface
    interface = DNC_InterfaceLayer_Adapted(hidden_size=hidden_size, interface_size=interface_size)

    # Create random state dictionary
    state_dict = {"hidden_state": torch.randn(batch_size, hidden_size)}

    # Forward pass through interface
    interface_vectors = interface(state_dict)

    # Print results
    print("\nDefaultInterface Test Results:")
    print(f"State dictionary shape: {state_dict['hidden_state'].shape}")
    print(f"Interface vectors: {interface_vectors}")

    # Summary
    print(interface.summary())
