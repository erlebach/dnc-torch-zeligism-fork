"""
Adaptation of the original DNC class from dnc_torch_zeligism to use the polymorphic structure
defined in base.py while preserving the original functionality.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from beartype import beartype

from dnc.base import BaseController
from dnc_torch_zeligism.training_configs import BATCH_SIZE
from interface_adapted import DNC_InterfaceLayer_Adapted
from memory_adapted import Memory_Adapted


@beartype
class DNC_Adapted(BaseController):
    """
    Adaptation of the original DNC class to inherit from BaseController.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        controller_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        Controller=nn.LSTM,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set default configurations if not provided
        if controller_config is None:
            controller_config = {"hidden_size": 64, "num_layers": 1}

        if memory_config is None:
            memory_config = {"memory_size": 128, "word_size": 20, "num_writes": 1, "num_reads": 4}

        # Save configurations
        self.controller_config = controller_config

        # Initialize memory
        memory_config_with_batch_size = memory_config.copy()
        if "batch_size" not in memory_config_with_batch_size:
            memory_config_with_batch_size["batch_size"] = BATCH_SIZE

        self.memory = Memory_Adapted(**memory_config_with_batch_size)

        # Calculate controller input size (original input + read vectors)
        self.input_size_raw = input_size
        self.input_size = input_size + self.memory.num_reads * self.memory.word_size
        self.output_size = output_size

        # Initialize controller
        self.controller = Controller(self.input_size, **controller_config)

        # Set controller's hidden size attribute for easy access
        self.controller.hidden_size = controller_config.get("hidden_size", 64)

        # Initialize interface
        self.interface = DNC_InterfaceLayer_Adapted(
            input_size=self.controller.hidden_size,
            memory_size=self.memory.memory_size,
            word_size=self.memory.word_size,
            num_writes=self.memory.num_writes,
            num_reads=self.memory.num_reads,
            batch_size=self.memory.batch_size,
        )

        # Set interface for memory (validates compatibility)
        self.memory.set_interface(self.interface)

        # Define output layer
        pre_output_size = (
            self.controller.hidden_size + self.memory.num_reads * self.memory.word_size
        )
        self.output_layer = nn.Linear(pre_output_size, output_size)

        # Set input/output shapes for summary
        self.input_shape = (None, input_size)
        self.output_shape = (None, output_size)

        # Initialize state
        self.init_state()

    def init_state(self) -> None:
        """Initialize DNC state."""
        # Initialize controller state
        num_layers = getattr(self.controller, "num_layers", 1)
        hidden_size = self.controller_config.get("hidden_size", 64)

        self.state = {
            "controller_h": torch.zeros(num_layers, self.memory.batch_size, hidden_size),
            "controller_c": torch.zeros(num_layers, self.memory.batch_size, hidden_size),
            "read_words": torch.zeros(
                self.memory.batch_size, self.memory.num_reads, self.memory.word_size
            ),
        }

        # Initialize memory and interface states
        self.memory.init_state()
        self.interface.init_state()

    def detach_state(self) -> None:
        """Detach DNC state from computation graph."""
        # Detach controller state
        for key, value in self.state.items():
            if isinstance(value, torch.Tensor):
                self.state[key] = value.detach()

        # Detach memory state
        self.memory.detach_state()

    def debug(self) -> None:
        """Print debug information about DNC state."""
        self.memory.debug()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DNC.

        Args:
            inputs: Input tensor of shape (sequence_length, batch_size, input_size)

        Returns:
            Output tensor of shape (sequence_length, batch_size, output_size)
        """
        # Detach state from previous computation graph
        self.detach_state()

        # Prepare output container
        outputs = []

        # Process each step in the sequence
        for i in range(inputs.size(0)):
            # Get current input
            x_t = inputs[i]

            # Concatenate input with previous read words
            controller_input = torch.cat(
                [
                    x_t.view(self.memory.batch_size, -1),
                    self.state["read_words"].view(self.memory.batch_size, -1),
                ],
                dim=1,
            )

            # Add sequence dimension for controller
            controller_input = controller_input.unsqueeze(0)

            # Run controller
            controller_state = (self.state["controller_h"], self.state["controller_c"])
            controller_output, (h_n, c_n) = self.controller(controller_input, controller_state)
            self.state["controller_h"] = h_n
            self.state["controller_c"] = c_n

            # Remove sequence dimension
            controller_output = controller_output.squeeze(0)

            # Process through interface
            interface_vectors = self.interface({"output": controller_output})

            # Update memory and get read words
            read_words = self.memory(interface_vectors)
            self.state["read_words"] = read_words

            # Prepare final output
            pre_output = torch.cat(
                [controller_output, read_words.view(self.memory.batch_size, -1)], dim=1
            )
            output = self.output_layer(pre_output)

            # Add to outputs
            outputs.append(output)

        # Stack outputs along sequence dimension
        return torch.stack(outputs, dim=0)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the complete state dictionary of the DNC.

        Returns:
            Dictionary containing all state variables
        """
        # Start with controller and read words state
        state_dict = self.state.copy()

        # Add memory state
        for key, value in self.memory.state.items():
            state_dict[f"memory_{key}"] = value

        return state_dict


if __name__ == "__main__":
    """Test the adapted DNC implementation."""
    import sys

    import numpy as np

    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        print("Testing DNC_Adapted implementation...")

        # Define model parameters
        input_size = 10
        output_size = 5
        controller_config = {"hidden_size": 64, "num_layers": 1}
        memory_config = {
            "memory_size": 128,
            "word_size": 20,
            "num_reads": 4,
            "num_writes": 1,
            "batch_size": BATCH_SIZE,
        }

        # Create DNC_Adapted model
        print("Creating DNC_Adapted model...")
        model = DNC_Adapted(
            input_size=input_size,
            output_size=output_size,
            controller_config=controller_config,
            memory_config=memory_config,
        )

        print(f"Model created with input_size={input_size}, output_size={output_size}")
        print(f"Memory config: {memory_config}")
        print(f"Controller config: {controller_config}")

        # Generate random input sequence
        seq_length = 5
        batch_size = BATCH_SIZE
        x = torch.randn(seq_length, batch_size, input_size)

        print(f"Input shape: {x.shape}")

        # Forward pass
        print("Running forward pass...")
        y = model(x)

        print(f"Output shape: {y.shape}")
        print(f"Output sample:\n{y[0, 0, :].detach().numpy()}")

        # Test memory state
        print("\nMemory state:")
        model.debug()

        # Test state dictionary
        print("\nTesting state dictionary...")
        state_dict = model.get_state_dict()
        print(f"State dictionary keys: {list(state_dict.keys())}")

        # Test detach_state
        print("\nTesting state detachment...")
        model.detach_state()
        print("State detached successfully")

        print("\nDNC_Adapted test completed successfully!")

    except ImportError as e:
        print(f"\nERROR: Import error occurred: {e}")
        print("\nThere appears to be an issue with the imported modules.")

    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
