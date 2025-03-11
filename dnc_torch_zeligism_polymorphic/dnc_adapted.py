"""
Adaptation of the original DNC class from dnc_torch_zeligism to use the polymorphic structure
defined in base.py while preserving the original functionality.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from beartype import beartype

from dnc.base import BaseController

# from dnc_torch_zeligism.training_configs import BATCH_SIZE
from dnc_torch_zeligism_polymorphic.configuration import (
    controller_config,
    memory_config,
    training_config,
)
from dnc_torch_zeligism_polymorphic.interface_adapted import DNC_InterfaceLayer_Adapted
from dnc_torch_zeligism_polymorphic.memory_adapted import Memory_Adapted


@beartype
class DNC_Adapted(BaseController):
    """Adapted Differentiable Neural Computer (DNC) implementation.

    This class implements a DNC that inherits from BaseController, providing a polymorphic
    structure while maintaining the original DNC functionality. The DNC combines a neural
    network controller with an external memory matrix, allowing for complex, memory-based
    computations.

    The architecture consists of:
    - A controller network (typically LSTM) that processes inputs and generates outputs
    - An external memory matrix that can be read from and written to
    - Interface mechanisms for memory access and manipulation

    Attributes:
        input_size_raw: Size of raw input features.
        input_size: Size of controller input (raw input + read vectors).
        output_size: Size of output features.
        controller_config: Configuration dictionary for the controller.
        memory: Memory_Adapted instance for external memory operations.
        controller: Neural network controller (typically LSTM).
        interface: Interface layer for memory operations.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        controller_config: Dict[str, Any],
        memory_config: Dict[str, Any],
        Controller=nn.LSTM,
        **kwargs,
    ):
        """Initialize the DNC_Adapted model.

        Make sure `batch_size` is included in `memory_config`
        """
        super().__init__(**kwargs)

        # Save configurations
        self.controller_config = controller_config

        common_dict = {**memory_config, **training_config}
        self.memory = Memory_Adapted(**common_dict)
        self.memory.print_state("DNC constructor")
        # -----------------------------------------

        # Calculate controller input size (original input + read vectors)
        self.input_size_raw = input_size
        self.input_size = input_size + self.memory.num_reads * self.memory.word_size
        self.output_size = output_size

        # Initialize controller
        self.controller = Controller(self.input_size, **controller_config)

        # Set controller's hidden size attribute for easy access
        self.controller.hidden_size = controller_config.get("hidden_size", 64)

        # Initialize state
        self.init_state()

        # Initialize interface
        self.interface = DNC_InterfaceLayer_Adapted(
            input_size=self.controller.hidden_size,
            memory_size=self.memory.memory_size,
            word_size=self.memory.word_size,
            num_writes=self.memory.num_writes,
            num_reads=self.memory.num_reads,
            batch_size=self.memory.batch_size,
        )

        self.interface.init_state()

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

    def print_state(self, msg: Optional[str] = None):
        """Print the state of the DNC."""
        print(f"\n=== DNC state ({msg})")
        for key, value in self.state.items():
            print(f"{key}: {value.shape=}, norm: {value.norm():.6f}")
        self.memory.print_state(msg)

    def init_state(self) -> None:
        """Initialize DNC state."""
        # Initialize controller state
        print("init_state: model init_state")
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

    def detach_state(self) -> None:
        """Detach DNC state from computation graph."""
        for key, value in self.state.items():
            if isinstance(value, torch.Tensor):
                self.state[key] = value.detach()

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
            # read_words = self.memory.update(interface_vectors)
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


# ======================================================================
if __name__ == "__main__":
    """Test the adapted DNC implementation."""
    import numpy as np

    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # print("Testing DNC_Adapted implementation...")

        # Define model parameters
        input_size = 10
        output_size = 5
        controller_config = {"hidden_size": 64, "num_layers": 1}
        memory_config = {
            "memory_size": 128,
            "word_size": 20,
            "num_reads": 4,
            "num_writes": 1,
            "batch_size": training_config["batch_size"],
        }

        # Create DNC_Adapted model
        # print("Creating DNC_Adapted model...")
        model = DNC_Adapted(
            input_size=input_size,
            output_size=output_size,
            controller_config=controller_config,
            memory_config=memory_config,
        )

        # print(f"Model created with input_size={input_size}, output_size={output_size}")
        # print(f"Memory config: {memory_config}")
        # print(f"Controller config: {controller_config}")

        # Generate random input sequence
        seq_length = 5
        batch_size = training_config["batch_size"]
        x = torch.randn(seq_length, batch_size, input_size)
        # print(f"Input shape: {x.shape}")
        # print(f"Input values: {x}")

        # Forward pass
        # print("Running forward pass...")
        print("==============================================================")
        for i in range(2):
            y = model(x)
            print(f"iteration: {i}")
            print(f"{x.shape=}, {y.shape=}")
            model.print_state(msg=f"Update {i}")
            print("==============================================================")
        quit()

        # print(f"Output shape: {y.shape}")
        # print(f"Output sample:\n{y[0, 0, :].detach().numpy()}")

        # Test memory state
        # print("\nMemory state:")
        model.debug()
        # quit()

        # Test state dictionary
        # print("\nTesting state dictionary...")
        state_dict = model.get_state_dict()
        # print(f"State dictionary keys: {list(state_dict.keys())}")

        # Test detach_state
        # print("\nTesting state detachment...")
        model.detach_state()
        # print("State detached successfully")

        # Store initial weights for comparison
        # print("\nStoring initial weights...")
        initial_weights = {}
        for name, param in model.named_parameters():
            initial_weights[name] = param.clone().detach()

        # Create a simple target and loss function
        # print("\nPerforming backpropagation...")
        target = torch.randn(seq_length, batch_size, output_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Forward pass, compute loss, and backpropagate
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        # print("BEFORE optimizer.step")
        optimizer.step()

        # print(f"Loss after one update: {loss.item()}", flush=True)

        # Check weight changes
        # print("\nChecking weight changes after backpropagation:")
        weight_changes = {}
        for name, param in model.named_parameters():
            weight_change = torch.abs(param - initial_weights[name]).mean().item()
            weight_changes[name] = weight_change
            # print(f"{name}: mean absolute change = {weight_change}")

        # Find largest weight change
        max_change_name = max(weight_changes, key=weight_changes.get)
        # print(
        # f"\nLargest weight change in: {max_change_name} with change of {weight_changes[max_change_name]}"
        # )

        # print("\nDNC_Adapted test completed successfully!")

    except ImportError as e:
        print(f"\nERROR: Import error occurred: {e}")
        print("\nThere appears to be an issue with the imported modules.")

    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
