"""
Adaptation of the original DNC class from dnc_torch_zeligism to use the polymorphic structure
defined in base.py while preserving the original functionality.
"""

from typing import Any, Dict, Optional
from beartype import beartype

import torch
import torch.nn as nn

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
