from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from torch import Tensor

from dnc.base import BaseController
from dnc_architecture_graves_2016.interface import Interface
from dnc_architecture_graves_2016.memory import Memory
from dnc_architecture_graves_2016.memory_config import memory_config
from dnc_architecture_graves_2016.training_config import training_config


@beartype
class DNC(BaseController):
    """
    Differentiable Neural Computer (DNC) implementation based on Graves et al. 2016.

    This class implements the full DNC architecture, combining a controller network
    with external memory through an interface layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        controller_config: Dict[str, Any] = None,
        memory_config: Dict[str, Any] = None,
        Controller=nn.LSTM,
        **kwargs,
    ) -> None:
        """
        Initialize the DNC.

        Args:
            input_size: Size of the input vector
            output_size: Size of the output vector
            controller_config: Configuration for the controller
            memory_config: Configuration for the memory
            Controller: Controller class to use (default: nn.LSTM)
        """
        super().__init__(**kwargs)

        # Set default configurations if not provided
        if controller_config is None:
            controller_config = {"hidden_size": 64, "num_layers": 1}

        if memory_config is None:
            from dnc_architecture_graves_2016.memory_config import (
                memory_config as default_memory_config,
            )

            memory_config = default_memory_config.copy()

        # Ensure batch_size is in memory_config
        if "batch_size" not in memory_config and "batch_size" in training_config:
            memory_config["batch_size"] = training_config["batch_size"]

        # Store configuration
        self.memory_config = memory_config
        self.controller_config = controller_config
        self.batch_size = memory_config.get("batch_size", 8)

        # Initialize memory
        self.memory = Memory(**memory_config)

        # Calculate controller input size (input + read vectors)
        self.input_size_raw = input_size
        self.input_size = input_size + (self.memory.num_reads * self.memory.word_size)
        self.output_size = output_size

        # Initialize controller
        self.controller = Controller(self.input_size, **controller_config)

        # Initialize interface
        self.interface = Interface(
            input_size=controller_config.get("hidden_size", 64),
            memory_size=self.memory.memory_size,
            word_size=self.memory.word_size,
            num_writes=self.memory.num_writes,
            num_reads=self.memory.num_reads,
            batch_size=self.batch_size,
        )

        # Set interface for memory (validates compatibility)
        self.memory.set_interface(self.interface)

        # Define output layer
        pre_output_size = controller_config.get("hidden_size", 64) + (
            self.memory.num_reads * self.memory.word_size
        )
        self.output_layer = nn.Linear(pre_output_size, output_size)

        # Set input/output shapes for summary
        self.input_shape = (None, input_size)
        self.output_shape = (None, output_size)

        # Initialize state
        self.init_state()

    def init_state(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the state of the DNC.

        Args:
            config: Optional configuration dictionary
        """
        # Initialize controller state
        if isinstance(self.controller, nn.LSTM):
            num_layers = getattr(self.controller, "num_layers", 1)
            hidden_size = self.controller_config.get("hidden_size", 64)

            self.state = {
                "controller_h": torch.zeros(num_layers, self.batch_size, hidden_size),
                "controller_c": torch.zeros(num_layers, self.batch_size, hidden_size),
                "read_words": torch.zeros(
                    self.batch_size, self.memory.num_reads, self.memory.word_size
                ),
            }
        else:  # RNN or other controller
            num_layers = getattr(self.controller, "num_layers", 1)
            hidden_size = self.controller_config.get("hidden_size", 64)

            self.state = {
                "controller_h": torch.zeros(num_layers, self.batch_size, hidden_size),
                "read_words": torch.zeros(
                    self.batch_size, self.memory.num_reads, self.memory.word_size
                ),
            }

        # Initialize memory state
        self.memory.init_state()

        # Initialize interface state
        self.interface.init_state()

    def detach_state(self) -> None:
        """Detach the state from the computation graph."""
        # Detach controller state
        for key, value in self.state.items():
            if isinstance(value, Tensor):
                self.state[key] = value.detach()

        # Detach memory state
        self.memory.detach_state()

    def debug(self) -> None:
        """Prints helpful information about the DNC for debugging."""
        self.memory.debug()

    def forward(self, inputs: Tensor) -> Tensor:
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
                [x_t.view(self.batch_size, -1), self.state["read_words"].view(self.batch_size, -1)],
                dim=1,
            )

            # Add sequence dimension for controller
            controller_input = controller_input.unsqueeze(0)

            # Run controller
            if isinstance(self.controller, nn.LSTM):
                controller_state = (self.state["controller_h"], self.state["controller_c"])
                controller_output, (h_n, c_n) = self.controller(controller_input, controller_state)
                self.state["controller_h"] = h_n
                self.state["controller_c"] = c_n
            else:  # RNN or other controller
                controller_state = self.state["controller_h"]
                controller_output, h_n = self.controller(controller_input, controller_state)
                self.state["controller_h"] = h_n

            # Remove sequence dimension
            controller_output = controller_output.squeeze(0)

            # Process through interface
            interface_vectors = self.interface({"output": controller_output})

            # Update memory and get read words
            read_words = self.memory(interface_vectors)
            self.state["read_words"] = read_words

            # Prepare final output
            pre_output = torch.cat([controller_output, read_words.view(self.batch_size, -1)], dim=1)
            output = self.output_layer(pre_output)

            # Add to outputs
            outputs.append(output)

        # Stack outputs along sequence dimension
        return torch.stack(outputs, dim=0)

    def get_state_dict(self) -> Dict[str, Tensor]:
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


def test_dnc():
    """Test the DNC implementation."""
    from dnc_architecture_graves_2016.memory_config import memory_config
    from dnc_architecture_graves_2016.training_config import training_config

    # Combine configurations
    config = {**memory_config, **training_config}

    # Controller configuration
    controller_config = {
        "hidden_size": 64,
        "num_layers": 1,
    }

    # Create DNC
    dnc = DNC(
        input_size=10,
        output_size=5,
        controller_config=controller_config,
        memory_config=config,
    )

    # Print architecture summary
    print(dnc.summary())

    # Create dummy input (sequence_length, batch_size, input_size)
    sequence_length = 3
    batch_size = config["batch_size"]
    input_size = 10
    x = torch.randn(sequence_length, batch_size, input_size)

    # Forward pass
    output = dnc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    return output


# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_dnc()
