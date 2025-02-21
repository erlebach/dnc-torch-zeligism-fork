import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from memory import Memory
from torch import Tensor
from training_configs import *


class DNC(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        controller_config: dict,
        memory_config: dict,
        controller=nn.LSTM,  # Class (should be polymorphic)
    ) -> None:
        """Initialize the DNC object."""
        super().__init__()

        # Initialize memory
        self.memory = Memory(**memory_config)

        # First add read vectors' size to controller's input_size
        self.input_size = input_size + self.memory.num_reads * self.memory.word_size
        # Now initialize controller
        # **controller_config are key-values for the controller.
        # Different dictionaries for different controllers.
        self.controller = controller(self.input_size, **controller_config)

        # Initialize state of DNC
        self.init_state()

        # Define interface layers
        self.interface_layer = DNCInterfaceLayer(
            self.controller.hidden_size,
            self.memory.num_writes,
            self.memory.num_reads,
            self.memory.word_size,
        )

        # Define output layer
        self.output_size = output_size
        pre_output_size = (
            self.controller.hidden_size + self.memory.num_reads * self.memory.word_size
        )
        self.output_layer = nn.Linear(pre_output_size, self.output_size)

    def init_state(self) -> None:
        """Initialize the state of the DNC."""
        self.state_dict = {}
        num_layers = self.controller.num_layers
        hidden_size = self.controller.hidden_size
        self.controller_state = (
            torch.zeros(num_layers, BATCH_SIZE, hidden_size),
            torch.zeros(num_layers, BATCH_SIZE, hidden_size),
        )
        # Initialize read_words state
        self.read_words = torch.zeros(BATCH_SIZE, self.memory.num_reads, self.memory.word_size)
        # tuple of states
        self.state_dict["raw_controller_state"] = self.controller_state
        self.state_dict["read_words"] = self.read_words

    def detach_state(self) -> None:
        """Detach the state of the DNC from the graph.

        This helps presserve memory in an RNN
        """
        self.controller_state = (
            self.controller_state[0].detach(),  # Original code had detach() here. WHY?
            self.controller_state[1].detach(),
        )
        self.read_words.detach_()
        # tuple of states
        self.state_dict["raw_controller_state"] = self.controller_state
        self.state_dict["read_words"] = self.read_words
        self.memory.detach_state()

    def debug(self):
        """Print helpful information about the DNC for debugging."""
        self.memory.debug()

    def forward(self, inputs: Tensor) -> Tensor:
        """Make one forward pass one the inputs.

        `inputs` should have dimension:
            (sequence_size, batch_size, input_size)
        `read_words` should have dimension:
            (batch_size, num_reads * word_size)

        Return:
            (sequence_size, batch_size, output_size)

        """
        self.detach_state()

        outputs = []
        print(f"forward, {inputs.shape=}")  # (15,8,8)
        # . inputs.shape: (15, 8, 8)
        for i in range(inputs.size()[0]):
            # We go through the inputs in the sequence one by one.

            # . X_t = input ++ read_vectors/read_words
            controller_input = torch.cat(
                [
                    rearrange(inputs[i], "b ... -> b (...)"),
                    rearrange(self.read_words, "b ... -> b (...)"),
                ],
                dim=1,
            )
            # Add sequence dimension for controller input
            controller_input = rearrange(controller_input, "b f -> 1 b f")
            # Run one step of controller
            controller_output, self.controller_state = self.controller(
                controller_input, self.controller_state
            )
            # Remove sequence dimension from controller output
            controller_output = rearrange(controller_output, "1 b f -> b f")

            """ Compute all the interface tensors by passing
            the controller's output to all the layers, and
            then passing the result as an input to memory. """
            interface = self.interface_layer(controller_output)
            self.read_words = self.memory.update(interface)

            # pre_output.shape = (batch, controller_hidden_size + num_reads * word_size)
            pre_output = torch.cat(
                [
                    controller_output,  # (batch, controller_hidden_size)
                    rearrange(self.read_words, "b ... -> b (...)"),
                ],
                dim=1,
            )
            # output.shape = (batch, output_size)
            output = self.output_layer(pre_output)
            print(f"output.shape={output.shape}")  # (8, 5) = (b, 5)

            outputs.append(output)
            # print(f"{len(outputs)=}, {outputs[0].shape=}, {outputs[-1].shape=}")

        # len(outputs) = 15, each element of size (8, 5)
        print(f"{len(outputs)=}, {outputs[0].shape=}, {outputs[-1].shape=}, {inputs.shape=}")
        return torch.stack(outputs, dim=0)


class DNCInterfaceLayer(nn.Module):
    """Create the interfade layer of the DNC.

    The interface layer of the DNC.
    Simply applies linear layers to the hidden state of the controller.
    Each linear layer is associated with an interface vector,
    as described in the paper. The output is reshaped accordingly in LinearView,
    and activations are applied depending on the type of interface vector.
    """

    def __init__(
        self,
        input_size: int,
        num_writes: int,
        num_reads: int,
        word_size: int,
    ) -> None:
        super().__init__()

        # Read and write keys and their strengths.
        self.read_keys = LinearView(input_size, [num_reads, word_size])
        self.read_strengths = LinearView(input_size, [num_reads])
        self.write_keys = LinearView(input_size, [num_writes, word_size])
        self.write_strengths = LinearView(input_size, [num_writes])
        # Erase and write (i.e. overwrite) vectors.
        self.erase_vectors = LinearView(input_size, [num_writes, word_size])
        self.write_vectors = LinearView(input_size, [num_writes, word_size])
        # Free, allocation, and write gates.
        self.free_gate = LinearView(input_size, [num_reads])
        self.allocation_gate = LinearView(input_size, [num_writes])
        self.write_gate = LinearView(input_size, [num_writes])
        # Read modes (forward + backward for each write head,
        # and one for content-based addressing).
        num_read_modes = 1 + 2 * num_writes
        self.read_modes = LinearView(input_size, [num_reads, num_read_modes])

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass of the DNC interface layer.

        Applies linear transformations and appropriate activations to generate interface vectors.
        These vectors are used by the memory module for read/write operations.

        Args:
            x: Input tensor of shape (batch_size, input_size) containing the hidden state from the controller.

        Notes:
            Any of the output linear layers could be either simplified or complexified.
            For example, the read_strengths could be simplified to a single linear layer,
            or the read_modes could be complexified to include more modes.

        Returns:
            Dictionary containing the following interface vectors:
            - read_keys: Keys for read operations
            - read_strengths: Strengths for read operations
            - write_keys: Keys for write operations
            - write_strengths: Strengths for write operations
            - erase_vectors: Vectors for erasing memory (sigmoid activated)
            - write_vectors: Vectors for writing memory (sigmoid activated)
            - free_gate: Gate for freeing memory (sigmoid activated)
            - allocation_gate: Gate for memory allocation (sigmoid activated)
            - write_gate: Gate for write operations (sigmoid activated)
            - read_modes: Modes for read operations (softmax activated along dim 2)

        """
        return {
            "read_keys": self.read_keys(x),
            "read_strengths": self.read_strengths(x),
            "write_keys": self.write_keys(x),
            "write_strengths": self.write_strengths(x),
            "erase_vectors": torch.sigmoid(self.erase_vectors(x)),
            "write_vectors": torch.sigmoid(self.write_vectors(x)),
            "free_gate": torch.sigmoid(self.free_gate(x)),
            "allocation_gate": torch.sigmoid(self.allocation_gate(x)),
            "write_gate": torch.sigmoid(self.write_gate(x)),
            "read_modes": F.softmax(self.read_modes(x), dim=2),
        }


class LinearView(nn.Module):
    """Output a tensor with size `dim`, similar to linear.

    Similar to linear, except that it outputs a tensor with size `dim`.
    It is assumed that the first dimension is the batch dimension.
    """

    def __init__(self, input_size: int, output_view: list[int]) -> None:
        super().__init__()
        # Calculate output size (just the product of dims in output_view)
        output_size = 1
        for dim in output_view:
            output_size *= dim
        # Define the layer and the desired view of the output
        self.layer = nn.Linear(input_size, output_size)
        self.output_view = output_view

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the LinearView module.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor reshaped according to output_view, with shape
            (batch_size, *output_view).

        """
        # -1 because we assume batch dimension exists
        out1 = self.layer(x).view(-1, *self.output_view)
        # The view operation reshapes the output of the linear layer to match the desired dimensions:
        # - -1 preserves the batch dimension
        # - *self.output_view unpacks the list of desired dimensions (e.g., [4, 8] becomes 4, 8)
        # This results in a tensor with shape (batch_size, *output_view)
        return out1
