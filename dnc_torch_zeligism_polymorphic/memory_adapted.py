"""
Adaptation of the original Memory class from dnc_torch_zeligism to use the polymorphic structure
defined in base.py while preserving the original functionality.
"""

import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from dnc.base import BaseMemory
from dnc_torch_zeligism.training_configs import BATCH_SIZE

print_interface_counter: int = 0


def print_interface(interface: dict[str, torch.Tensor]):
    global print_interface_counter
    print_interface_counter += 1
    print(f"\n==> ENTER print_interface ({print_interface_counter})")
    for key, value in interface.items():
        print(f"{key}: {value.shape=}, mean: {value.mean().item():.6f}")
    print()
    # if print_interface_counter == 10:
    # sys.exit()


def print_tensor(tens: torch.Tensor, msg: str):
    print(f"==> {msg}, {tens.shape=}, norm: {tens.norm():.6f}")


@beartype
class Memory_Adapted(BaseMemory):
    """
    Adaptation of the original Memory class to inherit from BaseMemory.
    """

    def __init__(
        self, memory_size=128, word_size=20, num_writes=1, num_reads=1, batch_size=BATCH_SIZE
    ):
        super().__init__()

        # Initialize memory parameters
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_writes = num_writes
        self.num_reads = num_reads
        self.batch_size = batch_size

        # Initialize EPSILON for stable calculations
        self.EPSILON = 1e-6

        # Define required shapes for interface validation
        self.required_shapes = {
            "read_keys": (num_reads, word_size),
            "read_strengths": (num_reads,),
            "write_keys": (num_writes, word_size),
            "write_strengths": (num_writes,),
            "erase_vectors": (num_writes, word_size),
            "write_vectors": (num_writes, word_size),
            "free_gate": (num_reads,),
            "allocation_gate": (num_writes,),
            "write_gate": (num_writes,),
            "read_modes": (num_reads, 1 + 2 * num_writes),
        }

        # Initialize memory state
        self.init_state()

        # print("INSIDE CONSTRUCTOR)")
        # self.print_memory_data()
        # self.print_memory_state()

    def init_state(self) -> None:
        """Initialize the memory state with detailed diagnostics."""
        # Print configuration details
        print("\nMemory Configuration:")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory size: {self.memory_size}")
        print(f"Word size: {self.word_size}")
        print(f"Num reads: {self.num_reads}")
        print(f"Num writes: {self.num_writes}")

        # Initialize memory state dictionary
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

        # Print initialization details
        # print("\nMemory Initialization Details:")
        #rint(f"Memory shape: {self.state['memory'].shape}")
        #rint(f"Memory sample values: {self.state['memory'][0, 0, :5]}")
        #rint(f"Memory mean: {self.state['memory'].mean().item()}")

        # For backward compatibility, also set these as attributes
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.precedence_weights = self.state["precedence_weights"]
        self.link = self.state["link"]
        self.usage = self.state["usage"]

    def detach_state(self) -> None:
        """Detach memory state from computation graph."""
        for key, tensor in self.state.items():
            self.state[key] = tensor.detach()

        # Update references for backward compatibility
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.link = self.state["link"]
        self.precedence_weights = self.state["precedence_weights"]
        self.usage = self.state["usage"]

    def debug(self) -> None:
        """Print debug information about memory state."""
        print("Memory shape:", self.state["memory"].shape)
        print(
            "Memory min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
                self.state["memory"].min().item(),
                self.state["memory"].mean().item(),
                self.state["memory"].max().item(),
            )
        )
        print(
            "Usage min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
                self.state["usage"].min().item(),
                self.state["usage"].mean().item(),
                self.state["usage"].max().item(),
            )
        )

    def content_based_address(self, memory_data, keys, strengths):
        """
        Performs content-based addressing.

        Args:
            memory_data: Tensor of shape (batch_size, memory_size, word_size)
            keys: Tensor of shape (batch_size, num_keys, word_size)
            strengths: Tensor of shape (batch_size, num_keys)

        Returns:
            Tensor of shape (batch_size, num_keys, memory_size)
        """
        # Compute dot product between keys and memory
        # Result shape: (batch_size, num_keys, memory_size)
        dot = torch.matmul(keys, memory_data.transpose(1, 2))

        # Compute L2 norm of memory vectors
        # Result shape: (batch_size, memory_size, 1)
        memory_norm = torch.norm(memory_data, p=2, dim=2, keepdim=True)

        # Compute L2 norm of key vectors
        # Result shape: (batch_size, num_keys, 1)
        key_norm = torch.norm(keys, p=2, dim=2, keepdim=True)

        # Broadcast for proper division
        memory_norm = memory_norm.transpose(1, 2)  # Now shape is (batch_size, 1, memory_size)

        # Compute cosine similarity
        # dot / (||memory|| * ||keys||)
        # Important: We need to broadcast properly here
        cosine_sim = dot / (torch.matmul(key_norm, memory_norm) + self.EPSILON)

        # Apply strength (temperature) and softmax
        # strengths: (batch_size, num_keys) -> (batch_size, num_keys, 1)
        similarity = F.softmax(cosine_sim * strengths.unsqueeze(2), dim=2)
        return similarity

    def forward(self, interface_vectors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through memory module.

        Args:
            interface_vectors: Dictionary of interface vectors from the interface module

        Returns:
            Read words (tensor of shape (batch_size, num_reads, word_size))
        """
        # This is equivalent to the original update method
        # print("call self.update from Memory_Adapted::forward")
        return self.update(interface_vectors)

    def update(self, interface):
        """
        Updates the current state of the memory. Returns the words read by memory.
        NOTE: the state variables of the memory in `self` should always be
        the previous states until `update()` is done. If a current state
        is needed in an update-subroutine, then it should be passed to it.

        Args:
            interface: Dictionary containing interface vectors

        Returns:
            Tensor of read words
        """
        # print("========> adapter, ENTER update")
        # print_interface(interface)

        # Store the interface for debugging
        self.last_interface = interface

        # Calculate the next usage
        # print("about to call self.update_usage from update()")
        usage_t = self.update_usage(interface["free_gate"])
        # print_tensor(usage_t, "usage_t/free_gate")

        # Calculate the content-based write addresses
        write_content_weights = self.content_based_address(
            self.state["memory"], interface["write_keys"], interface["write_strengths"]
        )

        # Find the next write weightings using the updated usage
        # print_interface(interface)  # OK

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

        # Update state of memory and return read words
        self.state["usage"] = usage_t
        self.state["write_weights"] = write_weights_t
        self.state["memory"] = memory_data_t
        self.state["link"] = link_t
        self.state["precedence_weights"] = precedence_weights_t
        self.state["read_weights"] = read_weights_t

        #print("====> INSIDE update")
        #print(f"{usage_t.shape=}, mean: {usage_t.mean().item():.6f}")
        #print(f"{write_weights_t.shape=}, mean: {write_weights_t.mean().item():.6f}")
        #print(f"{memory_data_t.shape=}, mean: {memory_data_t.mean().item():.6f}")
        #print(f"{link_t.shape=}, mean: {link_t.mean().item():.6f}")
        #print(f"{precedence_weights_t.shape=}, mean: {precedence_weights_t.mean().item():.6f}")
        #print(f"{read_weights_t.shape=}, mean: {read_weights_t.mean().item():.6f}")

        # For backward compatibility
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.link = self.state["link"]
        self.precedence_weights = self.state["precedence_weights"]
        self.usage = self.state["usage"]

        # Return the new read words for each read head from new memory data
        #print("====> EXIT update")
        return read_weights_t @ memory_data_t

    def update_usage(self, free_gate):
        """
        Updates and returns the memory usage vector.

        Args:
            free_gate: Tensor of shape (batch_size, num_reads)

        Returns:
            The updated usage vector
        """
        # First find the aggregate write weights of all write heads per memory cell.
        # This is in case there are more than one write head (i.e. num_writes > 1).
        cell_write_weights = 1 - torch.prod(1 - self.state["write_weights"], dim=1)

        # Usage is retained, and in addition, memory cells that are being used for
        # writing (i.e. have high cell_memory_weights) with low usage should have
        # their usage increased, which is exactly what is done here.
        usage_after_writes = self.state["usage"] + (1 - self.state["usage"]) * cell_write_weights

        # First, recall that there is a free_gate for each read-head.
        # Here, we multiply the free_gate of each read-head with all of its read_weights,
        # which gives us new read_weights scaled by the free_gate per read-head.
        free_read_weights = free_gate.unsqueeze(dim=-1) * self.state["read_weights"]

        # Next, we calculate psi, which is interpreted as a memory retention vector.
        psi = torch.prod(1 - free_read_weights, dim=1)

        # Finally, we calculate the next usage as defined in the paper.
        usage = usage_after_writes * psi

        # print(f"==> inside function update_usage, {usage=}")

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
        usage = self.EPSILON + (1 - self.EPSILON) * usage  # Avoid very small values

        # Sort `usage` and get keep its original indices in `phi`.
        sorted_usage, phi = usage.sort(dim=1)

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
        """
        #print(f"\n========> ENTER update_memory_data")
        #print(f"weights: {weights.shape=}, mean: {weights.mean().item():.6f}")
        #print(f"erases: {erases.shape=}, mean: {erases.mean().item():.6f}")
        #print(f"writes: {writes.shape=}, mean: {writes.mean().item():.6f}")

        # Take the outer product of the weights and erase vectors per write head.
        weighted_erase = weights.unsqueeze(dim=-1) * erases.unsqueeze(dim=-2)
        #print(f"==> adapted, update_memory_data, {weighted_erase.shape=}")

        # Take the aggregate erase factor through all write heads.
        erase_factor = torch.prod(1 - weighted_erase, dim=1)
        #print(f"==> adapted, update_memory_data, {erase_factor.shape=}")

        # Calculate the weighted words to add/write to memory.
        write_words = weights.transpose(1, 2) @ writes
        #print(f"==> adapted, update_memory_data, {write_words.shape=}")

        #print(f"==> adapted, update_memory_data, {self.state['memory'].shape=}")

        # Update memory
        updated_memory = self.state["memory"] * erase_factor + write_words

        # For backward compatibility
        self.memory_data = updated_memory

        # Return the updated memory
        return updated_memory

    def update_linkage(self, write_weights):
        """
        Update the temporal linkage matrix given the new write weights.

        Args:
            write_weights: The write weights for the current timestep.

        Returns:
            Tuple of (updated_link, updated_precedence_weights)
        """
        # Get the current link matrix and precedence weights
        link = self.state["link"]
        precedence_weights = self.state["precedence_weights"]

        # Create a new link matrix instead of modifying in-place
        batch_size = self.batch_size
        memory_size = self.memory_size

        # Create a new tensor for the updated link matrix
        updated_link = torch.zeros_like(link)

        # Update the link matrix for each write head
        for i in range(self.num_writes):
            write_weights_i = write_weights[:, i].unsqueeze(2)

            # Compute the outer product of precedence weights and write weights
            outer_product = precedence_weights.unsqueeze(2) * write_weights_i.transpose(1, 2)

            # Update the link matrix (avoid in-place operations)
            updated_link = updated_link + (1 - link) * outer_product

            # Instead, create a new tensor for this part of the calculation
            write_weights_outer = write_weights_i * write_weights_i.transpose(1, 2)
            link_scale = (1 - write_weights_i) * (1 - write_weights_i.transpose(1, 2))
            updated_link = updated_link * link_scale

        # Update precedence weights (avoid in-place operations)
        # Create a new tensor for updated precedence weights
        updated_precedence_weights = (
            1 - write_weights.sum(dim=1, keepdim=True)
        ) * precedence_weights

        # Add the current write weights
        for i in range(self.num_writes):
            updated_precedence_weights = updated_precedence_weights + write_weights[:, i]

        # Update the state with the new tensors
        self.state["link"] = updated_link
        self.state["precedence_weights"] = updated_precedence_weights

        return updated_link, updated_precedence_weights

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
        backward_read = torch.sum(backward_weights * backward_mode, dim=2)
        forward_read = torch.sum(forward_weights * forward_mode, dim=2)
        content_read = content_mode * content_weights

        return backward_read + forward_read + content_read

    def directional_read_weights(self, link, forward):
        """
        Calculates the directional read weights.
        Returns a tensor of size (BATCH_SIZE, num_reads, num_writes, memory_size).

        This function is pretty tricky to understand well, and it does only one
        little thing, which is multiply the link with the read_weights.
        Though, we have to make sure that we do that for every write and read head.
        """
        # Transpose link in case it is forward weightings (note opposite case)
        if forward:
            link = link.transpose(2, 3)

        # Add a dim for write heads and multiply with the link matrix.
        # Notice that dim 1 will be expanded to `num_writes` automatically.
        dir_weights = self.state["read_weights"].unsqueeze(dim=1) @ link

        # Return the directional weights with the flip fix as suggested.
        return dir_weights.transpose(1, 2)

    def print_memory_state(self):
        print("\n==> print_memory_state in memory_adapted")
        print(f"{self.memory_data.norm()=}")
        print(f"{self.read_weights.norm()=}")
        print(f"{self.write_weights.norm()=}")
        print(f"{self.precedence_weights.norm()=}")
        print(f"{self.link.norm()=}")
        print(f"{self.usage.norm()=}")

    def print_memory_data(self):
        print("\n==> print_memory_data in memory_adapted")
        print("Original memory data mean: ", self.memory_data.mean().item())
        print(f"{self.memory_data[0][0:2]=}")


if __name__ == "__main__":
    """Test the Memory class functionality.
    
    This test creates a Memory instance, initializes it with default parameters,
    and tests the memory update process with a mock interface dictionary.
    """
    import torch
    import torch.nn.functional as F

    from dnc_torch_zeligism.training_configs import (
        BATCH_SIZE,
        MEMORY_SIZE,
        NUM_READS,
        NUM_WRITES,
        WORD_SIZE,
    )

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create memory instance
    memory = Memory_Adapted(
        memory_size=MEMORY_SIZE, word_size=WORD_SIZE, num_writes=NUM_WRITES, num_reads=NUM_READS
    )

    print("\nMemory Test Results:")
    print(f"Memory size: {memory.memory_size}")
    print(f"Word size: {memory.word_size}")
    print(f"Number of read heads: {memory.num_reads}")
    print(f"Number of write heads: {memory.num_writes}")

    # Create mock interface vectors
    interface = {
        "read_keys": torch.randn(BATCH_SIZE, NUM_READS, WORD_SIZE),
        "read_strengths": torch.randn(BATCH_SIZE, NUM_READS),
        "write_keys": torch.randn(BATCH_SIZE, NUM_WRITES, WORD_SIZE),
        "write_strengths": torch.randn(BATCH_SIZE, NUM_WRITES),
        "erase_vectors": torch.sigmoid(torch.randn(BATCH_SIZE, NUM_WRITES, WORD_SIZE)),
        "write_vectors": torch.randn(BATCH_SIZE, NUM_WRITES, WORD_SIZE),
        "free_gate": torch.sigmoid(torch.randn(BATCH_SIZE, NUM_READS)),
        "allocation_gate": torch.sigmoid(torch.randn(BATCH_SIZE, NUM_WRITES)),
        "write_gate": torch.sigmoid(torch.randn(BATCH_SIZE, NUM_WRITES)),
        "read_modes": torch.softmax(torch.randn(BATCH_SIZE, NUM_READS, 2 * NUM_WRITES + 1), dim=2),
    }

    # Print initial memory state
    print("\nInitial Memory State:")
    print(f"Memory data shape: {memory.memory_data.shape}")
    print(f"Memory data mean: {memory.memory_data.mean().item():.6f}")
    print(f"Usage vector mean: {memory.usage.mean().item():.6f}")

    # Test content-based addressing
    content_weights = memory.content_based_address(
        memory.memory_data, interface["read_keys"], interface["read_strengths"]
    )
    print("\nContent-based Addressing Test:")
    print(f"Content weights shape: {content_weights.shape}")
    print(
        f"Content weights sum: {content_weights.sum(dim=2).mean().item():.6f}"
    )  # Should be close to 1.0

    # Test memory update
    print("memory.update in if __name__ == '__main__'")
    read_words = memory.update(interface)

    # Print results after update
    print("\nAfter Memory Update:")
    print(f"Read words shape: {read_words.shape}")
    print(f"Read words mean: {read_words.mean().item():.6f}")
    print(f"Updated memory data mean: {memory.memory_data.mean().item():.6f}")
    print(f"Updated usage vector mean: {memory.usage.mean().item():.6f}")

    # Test memory state detachment
    memory.detach_state()
    print("\nAfter State Detachment:")
    print(f"Memory data requires grad: {memory.memory_data.requires_grad}")

    print("\nMemory Test Completed Successfully!")

    memory.print_memory_data()
    memory.print_memory_state()
