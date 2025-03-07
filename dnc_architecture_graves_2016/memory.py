from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from torch import Tensor

from dnc.base import BaseMemory
from dnc_architecture_graves_2016.interface import Interface
from dnc_architecture_graves_2016.memory_config import memory_config
from dnc_architecture_graves_2016.training_config import training_config

# from training_configs import *

"""Memory."""


@beartype
class Memory(BaseMemory):
    def __init__(
        self,
        memory_size: int = 128,
        word_size: int = 20,
        num_writes: int = 1,
        num_reads: int = 1,
        batch_size: int | None = None,
        epsilon: float = 1.0e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        # Initialize memory parameters sizes
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_writes = num_writes
        self.num_reads = num_reads
        self.batch_size = batch_size
        self.epsilon = epsilon

        # Define required shapes for interface validation
        self.required_shapes = {
            "read_keys": (batch_size, num_reads, word_size),
            "read_strengths": (batch_size, num_reads),
            "write_keys": (batch_size, num_writes, word_size),
            "write_strengths": (batch_size, num_writes),
            "erase_vectors": (batch_size, num_writes, word_size),
            "write_vectors": (batch_size, num_writes, word_size),
            "free_gate": (batch_size, num_reads),
            "allocation_gate": (batch_size, num_writes),
            "write_gate": (batch_size, num_writes),
            "read_modes": (batch_size, num_reads, 2 * num_writes + 1),
        }

        # Initialize state of memory
        self.init_state(kwargs)

    def init_state(self, config: Dict[str, Any] = None) -> None:
        """Initialize the state of the memory.

        Args:
            config: Optional configuration dictionary
        """
        # Initialize memory state tensors
        self.state = {
            "memory": torch.zeros(self.batch_size, self.memory_size, self.word_size),
            "read_weights": torch.zeros(self.batch_size, self.num_reads, self.memory_size),
            "write_weights": torch.zeros(self.batch_size, self.num_writes, self.memory_size),
            "precedence_weights": torch.zeros(self.batch_size, self.num_writes, self.memory_size),
            "link": torch.zeros(
                self.batch_size, self.num_writes, self.memory_size, self.memory_size
            ),
            "usage": torch.zeros(self.batch_size, self.memory_size),
        }

        # Initialize memory matrix with specific values
        if "memory" not in self.state:
            self.state["memory"] = torch.zeros(self.batch_size, self.memory_size, self.word_size)

        # Set memory to specific initial values (0.01 in this case)
        self.state["memory"].fill_(0.01)

        # Initialize other memory components
        self.state["usage"] = torch.zeros(self.batch_size, self.memory_size)
        self.state["link"] = torch.zeros(self.batch_size, self.memory_size, self.memory_size)
        self.state["precedence_weights"] = torch.zeros(self.batch_size, self.memory_size)

        # Initialize read/write weights
        self.state["read_weights"] = torch.zeros(self.batch_size, self.num_reads, self.memory_size)
        self.state["write_weights"] = torch.zeros(self.batch_size, self.memory_size)

        # For backward compatibility, also set these as attributes
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.precedence_weights = self.state["precedence_weights"]
        self.link = self.state["link"]
        self.usage = self.state["usage"]

    def detach_state(self) -> None:
        """Detach the state from the computation graph."""
        super().detach_state()

        # Also update the attribute references
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.precedence_weights = self.state["precedence_weights"]
        self.link = self.state["link"]
        self.usage = self.state["usage"]

    def debug(self):
        """
        Debug memory.
        """
        print("-----------------------------------")
        print(self.memory_data[0, ...])
        print()

    def content_based_address(self, memory_data, keys, strengths):
        """
        Content-based addressing.
        Returns the content-based weights for each head.

        First, find the vector of cosine similarities between the key and the words
        in memory, for each head. After that, we simply do a weighted softmax on
        the similarity using `strengths` as the weight.
        The arguments have dimensions as follows:
        `memory_data`: (BATCH_SIZE, memory_size, word_size)
        `keys`:        (BATCH_SIZE, num_heads, word_size)
        `strengths`:   (BATCH_SIZE, num_heads)
        The returned weights have a dimension of (BATCH_SIZE, num_heads, memory_size).
        """

        # For each head, find cosine similarity of the word for that head
        # with each word in memory -> _, num_heads, memory_size, word*word.
        cosine_similarity = F.cosine_similarity(
            keys.unsqueeze(dim=2), memory_data.unsqueeze(dim=1), dim=3, eps=self.epsilon
        )

        # Transform strengths using the oneplus(x) function.
        strengths = 1 + F.softplus(strengths).unsqueeze(dim=2)

        # Get the content-based weights using the weighted softmax method
        content_weights = F.softmax(cosine_similarity * strengths, dim=2)

        return content_weights

    def forward(self, state_dict: Dict[str, Tensor]) -> Tensor:
        """Forward pass through the memory module.

        Args:
            state_dict: Dictionary of interface vectors

        Returns:
            Tensor of read words
        """
        return self.update(state_dict)

    def update(self, interface):
        """
        Updates the current state of the memory. Returns the words read by memory.
        NOTE: the state variables of the memory in `self` should always be
        the previous states until `update()` is done. If a current state
        is needed in an update-subroutine, then it should be passed to it.

        Args:
        `Interface` is a dictionary of of tensors that describe how the memory
        should be updated and how the data should be retrieved and written.

        The names of each tensor in the interface is
        the following (batch dimension not included):
             names                 dim
        1) read_keys          (num_reads, word_size)
        2) read_strengths     (num_reads)
        3) write_keys         (num_writes, word_size)
        4) write_strengths    (num_writes)
        5) erase_vectors      (num_writes, word_size)
        6) write_vectors      (num_writes, word_size)
        7) free_gate          (num_reads)
        8) allocation_gate    (num_writes)
        9) write_gate         (num_writes)
        T) read_modes         (num_reads, num_read_modes)

        A memory update also updates the internal state of the memory.
        The state of the memory contains (batch dimension not included):
                names                 dim
        1) memory_data        (memory_size, word_size)
        2) read_weights       (num_reads, memory_size)
        3) write_weights      (num_writes, memory_size)
        4) precedence_weights (num_writes, memory_size)
        5) link               (num_writes, memory_size, memory_size)
        6) usage              (memory_size)

        A memory update can be divided into these steps:
        1) Read interface vector (the tensors are passed by DNC).
        2) Update current usage using free_gate.
        3) Find allocation weighting by using the usage vector
            and the content-based weightings for the write heads.
        4) Calculate write_weights by finding a write content weighting first,
            by using content-based addressing function C for the weight_keys.
            Next, we use write_gate, allocation_gate, allocation weightings,
            and write content weightings to find write_weights.
        5) Update memory by using erase_vectors and write_vectors.
        6) Update link matrix. See `update_linkage()` for more details.
        7) Calculate content-based read addresses,
            and then update read weights, which depends on content-addressing
            as well as link matrix (forward/backward linkage read weights).
            We interpolate between these three modes to get the final read weights.
        8) Update the state of the DNC (note that `self` still has `t-1` state).
        9) Return the words read from memory by the read heads.
        """

        # Store the interface for debugging
        self.last_interface = interface

        # Calculate the next usage
        usage_t = self.update_usage(interface["free_gate"])

        # Calculate the content-based write addresses
        write_content_weights = self.content_based_address(
            self.memory_data, interface["write_keys"], interface["write_strengths"]
        )
        # Find the next write weightings using the updated usage
        write_weights_t = self.update_write_weights(
            usage_t, interface["write_gate"], interface["allocation_gate"], write_content_weights
        )

        # Write/erase to memory using the write weights we just got
        memory_data_t = self.update_memory_data(
            write_weights_t, interface["erase_vectors"], interface["write_vectors"]
        )

        # Update the link matrix and the precedence weightings
        link_t, precedence_weights_t = self.update_linkage(write_weights_t)

        # Calculate the content-based read addresses (note updated memory)
        read_content_weights = self.content_based_address(
            memory_data_t, interface["read_keys"], interface["read_strengths"]
        )
        # Find the next read weights using linkage matrix
        read_weights_t = self.update_read_weights(
            link_t, interface["read_modes"], read_content_weights
        )

        # Update state of memory
        self.state["memory"] = memory_data_t
        self.state["usage"] = usage_t
        self.state["write_weights"] = write_weights_t
        self.state["link"] = link_t
        self.state["precedence_weights"] = precedence_weights_t
        self.state["read_weights"] = read_weights_t

        # Update attribute references for backward compatibility
        self.memory_data = memory_data_t
        self.usage = usage_t
        self.write_weights = write_weights_t
        self.link = link_t
        self.precedence_weights = precedence_weights_t
        self.read_weights = read_weights_t

        # Return the new read words for each read head from new memory data
        return read_weights_t @ memory_data_t

    def update_usage(self, free_gate):
        """
        Calculates and returns the next/current `usage`.

        Takes `free_gate` from the `interface` vector as an input, and also uses
        previous `write_weights`, previous `read_weights`, and previous `usage`.
        Assumes that the memory has the previous states stored in `self` directly.
        Note that all the mutliplications here are element-wise.
        """

        # First find the aggregate write weights of all write heads per memory cell.
        # This is in case there are more than one write head (i.e. num_writes > 1).
        cell_write_weights = 1 - torch.prod(1 - self.write_weights, dim=1)

        # Usage is retained, and in addition, memory cells that are being used for
        # writing (i.e. have high cell_memory_weights) with low usage should have
        # their usage increased, which is exactly what is done here.
        # Fix: Add unsqueeze to make cell_write_weights have shape (batch_size, memory_size)
        print(f"==> function update_usage, {self.usage.shape=}, {cell_write_weights.shape=}")
        cell_write_weights = (
            cell_write_weights.unsqueeze(1) if cell_write_weights.dim() == 1 else cell_write_weights
        )
        usage_after_writes = self.usage + (1 - self.usage) * cell_write_weights

        # First, recall that there is a free_gate for each read-head.
        # Here, we multiply the free_gate of each read-head with all of its read_weights,
        # which gives us new read_weights scaled by the free_gate per read-head.
        free_read_weights = free_gate.unsqueeze(dim=-1) * self.read_weights

        # Next, we calculate psi, which is interpreted as a memory retention vector.
        psi = torch.prod(1 - free_read_weights, dim=1)

        # Finally, we calculate the next usage as defined in the paper.
        usage = usage_after_writes * psi

        return usage

    def update_write_weights(self, usage, write_gate, allocation_gate, write_content_weights):
        """
        Calculates and returns the next/current `write_weights`.
        It's pretty similar to the one in DeepMind's code.

        Takes the updated usage, the write gate, the allocation gate, and the
        write content weights (to find the complete write weights).
        The updated usage is used here to find `phi` and allocation weightings.
        """

        # Find the allocation weights
        write_allocation_weights = self.write_allocation_weights(
            write_gate * allocation_gate, usage
        )

        # Add a dimension to gates for scalar multiplication along memory cells
        write_gate = write_gate.unsqueeze(dim=-1)
        allocation_gate = allocation_gate.unsqueeze(dim=-1)

        # Calculate `write_weights` using allocation and content-based weights
        write_weights = write_gate * (
            allocation_gate * write_allocation_weights
            + (1 - allocation_gate) * write_content_weights
        )

        return write_weights

    def write_allocation_weights(self, write_alloc_gates, usage):
        """
        Calculates and returns the write weights due to allocation.
        The returned tensor will have size of (BATCH_SIZE, num_writes, memory_size).
        This function is pretty identical to the one in DeepMind's code.
        `write_alloc_gates` is simply the product of `write_gate` and `allocation_gate`.
        It is used, along with `usage`, in case there is more than one write head.

        For more than one write head, the code from DeepMind does what they call a
        "simulated new usage", where it takes into account where the previous write
        heads are writing, and update its own usage based on that. This implies that
        there is some sort of precedence or ordering among the write heads.
        """

        # Add a dimension so that when we index the write head, we get
        # a tensor of size (BATCH_SIZE, 1) to multiply it with allocation weights.
        write_alloc_gates = write_alloc_gates.unsqueeze(dim=-1)

        write_allocation_weights = []
        for i in range(self.num_writes):
            # Get allocation weights per write head and add it to the big list
            write_allocation_weights.append(self.allocation(usage))
            # This is the "simulated new usage" thing. Note that usage can only
            # further increase due to the ith (and previous) write head activity.
            usage += (1 - usage) * write_alloc_gates[:, i, :] * write_allocation_weights[i]

        # Stack allocation weights into one tensor and return
        return torch.stack(write_allocation_weights, dim=1)

    def allocation(self, usage):
        """
        Sort of a subroutine that runs in `update_write_weights(...)`.
        Returns the allocation weightings for one write head given the usage.
        Note that `allocation_weights_per_write` has the same size as `usage`.
        """

        usage = self.epsilon + (1 - self.epsilon) * usage  # Avoid very small values

        # Sort `usage` and get keep its original indices in `phi`.
        sorted_usage, phi = self.usage.sort()

        # We will add this `one` before the `sorted_usage`.
        one = torch.ones(self.batch_size, 1)
        padded_sorted_usage = torch.cat([one, sorted_usage], dim=1)
        # Now we can take the "exclusive" cumprod of the `sorted_usage` by taking
        # the cumprod of `padded_sorted_usage` and dropping the last column.
        cumprod_sorted_usage = padded_sorted_usage.cumprod(dim=1)[:, :-1]

        # Next we find the allocation weights.
        sorted_allocation = (1 - sorted_usage) * cumprod_sorted_usage
        # And unsort them using the original indices in `phi`.
        allocation_weights = sorted_allocation.gather(dim=1, index=phi)

        return allocation_weights

    def update_memory_data(self, weights, erases, writes):
        """
        Update the data of the memory. Returns the updated memory.
        The equation in the paper is I believe equivalent to this:
              memory_data * erase_factor   +   write_words
        M_t = M_t-1 o (1 - w_t^T * e_t) + (w_t^T * v_t)

        Though, I don't think that is how the "erased memory" is calculated in the
        source code. It doesn't do matrix multiplication. Instead, it computes the
        outer product of the weights and the erase vectors for each write head,
        and then it takes the product of (1 - result) through all write heads.
        """

        # Take the outer product of the weights and erase vectors per write head.
        weighted_erase = weights.unsqueeze(dim=-1) * erases.unsqueeze(dim=-2)
        # Take the aggregate erase factor through all write heads.
        erase_factor = torch.prod(1 - weighted_erase, dim=1)

        # Calculate the weighted words to add/write to memory.
        write_words = weights.transpose(1, 2) @ writes

        # Return the updated memory
        return self.memory_data * erase_factor + write_words

    def update_linkage(self, write_weights):
        """
        Updates the temporal linkage.
        Returns a tuple (link, precedence_weights)

        We expand the write weights to form a matrix such that

              [w_1, w_1, w_1]         [w_1, w_2, w_3]
        w_i = [w_2, w_2, w_2],  w_j = [w_1, w_2, w_3]
              [w_3, w_3, w_3]         [w_1, w_2, w_3]

        for each write head, i.e. dim(w_i) = dim(w_j) =
        (BATCH_SIZE, num_writes, memory_size, memory_size),
        where in the above example, memory_size = 3.
        This way, it can be shown that element-wise multiplication with
        the link matrix such as (1 - w_i - w_j) * L
        satisfies L[i,j] = (1 - w[i] - w[j]) @ L_prev[i,j].
        We can arrive to the same conclusion using the same logic for w[i] * p[j].
        (I'm explaining this because it slightly confused me when I first saw it).
        """
        w_i = write_weights.unsqueeze(dim=-1)
        w_j = write_weights.unsqueeze(dim=-2)
        p_j = self.precedence_weights.unsqueeze(dim=-2)  # p{t-1}_j to be pedantic
        link = (1 - w_i - w_j) * self.link + w_i * p_j
        inverted_eye = 1 - torch.eye(self.memory_size).expand_as(link)
        link = link * inverted_eye  # Set diagonal to 0s

        # Calculate precedence weightings
        precedence_weights = write_weights + self.precedence_weights * (
            1 - write_weights.sum(dim=2, keepdim=True)
        )

        return link, precedence_weights

    def update_read_weights(self, link, read_modes, content_weights):
        """
        Update read weights.
        `content_weights` (BATCH_SIZE, num_reads, memory_size)
        """

        # Calculate the directional read weights
        # both dim: (BATCH_SIZE, num_reads, num_writes, memory_size)
        backward_weights = self.directional_read_weights(link, forward=False)
        forward_weights = self.directional_read_weights(link, forward=True)

        # These are the (chosen) ranges of the three modes by definition
        backward_mode_range = range(self.num_writes)
        forward_mode_range = range(self.num_writes, 2 * self.num_writes)
        content_mode_range = range(2 * self.num_writes, 2 * self.num_writes + 1)

        # Extract the tensors for each mode (note their dimensions)
        # forward/backward dim: (BATCH_SIZE, num_reads, num_writes, 1)
        # content dim: (BATCH_SIZE, num_reads, 1)
        backward_mode = read_modes[..., backward_mode_range].unsqueeze(dim=-1)
        forward_mode = read_modes[..., forward_mode_range].unsqueeze(dim=-1)
        content_mode = read_modes[..., content_mode_range]

        # Get the final read weightings depending on the focus of the current
        # mode using the modes weights to interpolate among the three read weights.
        # (We sum the weights across the write heads for backward/forward modes).
        backward_read = torch.sum(backward_weights * forward_mode, dim=2)
        forward_read = torch.sum(forward_weights * forward_mode, dim=2)
        content_read = content_mode * content_weights

        return backward_read + forward_read + content_read

    def directional_read_weights(self, link, forward):
        """
        Calculates the directional read weights.
        Returns a tensor of size (BATCH_SIZE, num_reads, num_writes, memory_size).

        This function is pretty tricky to understand well, and it does only one
        little thing, which is multiply the link  with the read_weights.
        Though, we have to make sure that we do that for every write and read head.
        In other words, for each read head, we want to get `memory_size` weights
        for each link, and since we have `num_writes` links, we will get
        `num_writes` * `memory_size`-sized vectors for each read head (so, times
        `num_heads`). Then, just have to add a dim for `num_writes` in the weights
        to make it "broadcastable" with the link matrix.

        After that, we flip the read and write dimensions and return the weights.
        Also, note that here the matrix multiplication here is transposed because
        the order of the dimensions of the weights is transposed relative to the
        paper. However, that is not the case with the link matrix, so we transpose
        it in the opposite case.
        """

        # Transpose link in case it is forward weightings (note opposite case)
        if forward:
            link = link.transpose(2, 3)

        # Add a dim for write heads and multiply with the link matrix.
        # Notice that dim 1 will be expanded to `num_writes` automatically.
        dir_weights = self.read_weights.unsqueeze(dim=1) @ link

        # Return the directional weights with the flip fix as suggested.
        return dir_weights.transpose(1, 2)

    def summary(self) -> str:
        """Return a string summary of the memory's architecture.

        Returns:
            A string containing memory configuration and requirements.
        """
        # Calculate total elements in state tensors
        state_elements = sum(tensor.numel() for tensor in self.state.values())

        memory_shape = self.state["memory"].shape if "memory" in self.state else "Not initialized"

        return (
            f"\nMemory Summary:\n"
            f"  Memory shape: {memory_shape}\n"
            f"  Memory size: {self.memory_size}\n"
            f"  Word size: {self.word_size}\n"
            f"  Number of read heads: {self.num_reads}\n"
            f"  Number of write heads: {self.num_writes}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Required interface outputs:\n"
            f"    " + "\n    ".join(f"{k}: {v}" for k, v in self.required_shapes.items()) + "\n"
            f"  Interface attached: {self.interface is not None}\n"
            f"  Learnable parameters: {sum(p.numel() for p in self.parameters())} (Memory has no learnable parameters)\n"
            f"  State elements: {state_elements}\n"
        )


def test_memory_with_interface():
    """Test the Memory class with the Interface class."""
    import torch

    from dnc_architecture_graves_2016.interface import Interface
    from dnc_architecture_graves_2016.memory_config import memory_config
    from dnc_architecture_graves_2016.training_config import training_config

    # Combine configurations
    config = {**memory_config, **training_config}

    # Create memory module
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

    # Set interface for memory (validates compatibility)
    memory.set_interface(interface)

    # Create dummy controller output
    controller_output = torch.randn(config["batch_size"], controller_output_size)

    # Process through interface
    interface_vectors = interface({"output": controller_output})

    # Process through memory
    read_words = memory(interface_vectors)

    print(f"Read words shape: {read_words.shape}")
    # Should be (batch_size, num_reads, word_size)

    return read_words


if __name__ == "__main__":
    config = {**memory_config, **training_config}
    print(config)
    memory = Memory(**config)

    # Print memory architecture summary
    print(memory.summary())

    # Test with interface
    test_memory_with_interface()
