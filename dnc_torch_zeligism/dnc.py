import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from dnc_torch_zeligism.memory import Memory
from dnc_torch_zeligism.training_configs import *


@beartype
class DNC(nn.Module):
    def __init__(
        self, input_size, output_size, controller_config, memory_config, Controller=nn.LSTM
    ):
        super().__init__()

        # Initialize memory
        self.memory = Memory(**memory_config)

        # First add read vectors' size to controller's input_size
        self.input_size = input_size + self.memory.num_reads * self.memory.word_size
        # Now initialize controller (LSTM in this case)
        self.controller = Controller(self.input_size, **controller_config)
        self.controller_config = controller_config

        # Initialize state of DNC
        self.init_state()

        # Define interface layers
        self.interface_layer = DNC_InterfaceLayer(
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

    def init_state(self):
        """
        Initialize the state of the DNC.
        """
        # Get the batch size from training_configs
        from dnc_torch_zeligism.training_configs import BATCH_SIZE

        # Initialize controller state
        num_layers = getattr(self.controller, "num_layers", 1)
        print(f"{self.controller_config=}")  # not defined
        hidden_size = self.controller_config.get("hidden_size", 64)

        self.controller_state = (
            torch.zeros(num_layers, BATCH_SIZE, hidden_size),
            torch.zeros(num_layers, BATCH_SIZE, hidden_size),
        )

        # Initialize read_words state
        self.read_words = torch.zeros(BATCH_SIZE, self.memory.num_reads, self.memory.word_size)

    def detach_state(self):
        """
        Detach the state of the DNC from the graph.
        """
        self.controller_state = (
            self.controller_state[0].detach(),
            self.controller_state[1].detach(),
        )
        self.read_words.detach_()
        self.memory.detach_state()

    def debug(self):
        """
        Prints helpful information about the DNC for debugging.
        """
        self.memory.debug()

    def forward(self, inputs):
        """
        Makes one forward pass one the inputs.
        `inputs` should have dimension:
            (sequence_size, batch_size, input_size)
        `read_words` should have dimension:
            (batch_size, num_reads * word_size)
        """

        self.detach_state()

        outputs = []
        for i in range(inputs.size()[0]):
            # We go through the inputs in the sequence one by one.

            # X_t = input ++ read_vectors/read_words
            controller_input = torch.cat(
                [inputs[i].view(BATCH_SIZE, -1), self.read_words.view(BATCH_SIZE, -1)], dim=1
            )
            # Add sequence dimension
            controller_input = controller_input.unsqueeze(dim=0)
            # Run one step of controller
            controller_output, self.controller_state = self.controller(
                controller_input, self.controller_state
            )
            # Remove sequence dimension
            controller_output = controller_output.squeeze(dim=0)

            """ Compute all the interface tensors by passing
            the controller's output to all the layers, and
            then passing the result as an input to memory. """
            interface = self.interface_layer(controller_output)
            print("before memory_update, call self.memory.update() from DNC::forward")
            self.read_words = self.memory.update(interface)

            pre_output = torch.cat([controller_output, self.read_words.view(BATCH_SIZE, -1)], dim=1)
            output = self.output_layer(pre_output)

            outputs.append(output)

        return torch.stack(outputs, dim=0)


@beartype
class DNC_InterfaceLayer(nn.Module):
    """
    The interface layer of the DNC.
    Simply applies linear layers to the hidden state of the controller.
    Each linear layer is associated with an interface vector,
    as described in the paper. The output is reshaped accordingly in LinearView,
    and activations are applied depending on the type of interface vector.
    """

    def __init__(self, input_size, num_writes, num_reads, word_size):
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

    def forward(self, x):
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


@beartype
class LinearView(nn.Module):
    """
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
    """Test the DNC implementation."""
    import numpy as np

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Testing DNC implementation...")

    # Define model parameters
    input_size = 10
    output_size = 5
    controller_config = {"hidden_size": 64, "num_layers": 1}
    memory_config = {"memory_size": 128, "word_size": 20, "num_reads": 4, "num_writes": 1}

    # Create DNC model
    print("Creating DNC model...")
    model = DNC(
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
    # print(f"Input values: {x}")

    # Forward pass
    print("Running forward pass...")
    y = model(x)
    print("==============================================================")

    # print(f"Output shape: {y.shape}")
    # print(f"Output sample:\n{y[0, 0, :].detach().numpy()}")

    # Test memory state
    print("\nMemory state:")
    model.debug()
    quit()

    # Test detach_state
    print("\nTesting state detachment...")
    model.detach_state()
    print("State detached successfully")

    # Store initial weights for comparison
    print("\nStoring initial weights...")
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.clone().detach()

    # Create a simple target and loss function
    print("\nPerforming backpropagation...")
    target = torch.randn(seq_length, batch_size, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Forward pass, compute loss, and backpropagate
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    print("BEFORE optimizer.step")
    optimizer.step()

    print(f"Loss after one update: {loss.item()}")

    # Check weight changes
    print("\nChecking weight changes after backpropagation:")
    weight_changes = {}
    for name, param in model.named_parameters():
        weight_change = torch.abs(param - initial_weights[name]).mean().item()
        weight_changes[name] = weight_change
        print(f"{name}: mean absolute change = {weight_change}")

    # Find largest weight change
    max_change_name = max(weight_changes, key=weight_changes.get)
    print(
        f"\nLargest weight change in: {max_change_name} with change of {weight_changes[max_change_name]}"
    )

    print("\nDNC test completed successfully!")
