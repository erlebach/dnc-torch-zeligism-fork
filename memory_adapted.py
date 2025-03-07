"""
Adaptation of the original Memory class from dnc_torch_zeligism to use the polymorphic structure
defined in base.py while preserving the original functionality.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from dnc.base import BaseMemory
from dnc_torch_zeligism.training_configs import BATCH_SIZE


def print_interface(interface: dict[str, torch.Tensor]):
    print("\n==> ENTER print_interface")
    for key, value in interface.items():
        print(f"{key}: {value.shape=}, mean: {value.mean().item():.6f}")
    print()


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

        print("INSIDE CONSTRUCTOR)")
        self.print_memory_data()
        self.print_memory_state()

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
        print("\nMemory Initialization Details:")
        print(f"Memory shape: {self.state['memory'].shape}")
        print(f"Memory sample values: {self.state['memory'][0, 0, :5]}")
        print(f"Memory mean: {self.state['memory'].mean().item()}")

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
        print("========> adapter, ENTER update")
        print_interface(interface)

        # Store the interface for debugging
        self.last_interface = interface

        # Calculate the next usage
        usage_t = self.update_usage(interface["free_gate"])
        print_tensor(usage_t, "usage_t/free_gate")

        # Calculate the content-based write addresses
        write_content_weights = self.content_based_address(
            self.state["memory"], interface["write_keys"], interface["write_strengths"]
        )

        # Find the next write weightings using the updated usage
        print_interface(interface)  # OK

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

        print("====> INSIDE update")
        print(f"{usage_t.shape=}, mean: {usage_t.mean().item():.6f}")
        print(f"{write_weights_t.shape=}, mean: {write_weights_t.mean().item():.6f}")
        print(f"{memory_data_t.shape=}, mean: {memory_data_t.mean().item():.6f}")
        print(f"{link_t.shape=}, mean: {link_t.mean().item():.6f}")
        print(f"{precedence_weights_t.shape=}, mean: {precedence_weights_t.mean().item():.6f}")
        print(f"{read_weights_t.shape=}, mean: {read_weights_t.mean().item():.6f}")

        # For backward compatibility
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.link = self.state["link"]
        self.precedence_weights = self.state["precedence_weights"]
        self.usage = self.state["usage"]

        # Return the new read words for each read head from new memory data
        print("====> EXIT update")
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

        print(f"==> update_usage, {usage=}")

        return usage

    def update_write_weights(self, usage, write_gate, allocation_gate, write_content_weights):
        """
        Updates and returns the memory write weights.

        Args:
            usage: Tensor of shape (batch_size, memory_size)
            write_gate: Tensor of shape (batch_size, num_writes)
            allocation_gate: Tensor of shape (batch_size, num_writes)
            write_content_weights: Tensor of shape (batch_size, num_writes, memory_size)

        Returns:
            The updated write weights
        """
        # Calculate allocation weights
        alloc_weights = self.write_allocation_weights(allocation_gate, usage)

        # Calculate the interpolation between content-based addressing and allocation-based addressing
        updated_write_weights = write_gate.unsqueeze(2) * (
            allocation_gate.unsqueeze(2) * alloc_weights
            + (1 - allocation_gate).unsqueeze(2) * write_content_weights
        )

        return updated_write_weights

    def write_allocation_weights(self, write_alloc_gates, usage):
        """
        Calculates allocation weights given usage vector.

        Args:
            write_alloc_gates: Tensor of shape (batch_size, num_writes)
            usage: Tensor of shape (batch_size, memory_size)

        Returns:
            Allocation weights of shape (batch_size, num_writes, memory_size)
        """
        # Calculate the allocation weights for each write head using the memory usage
        alloc_weights = []
        for i in range(self.num_writes):
            alloc_weights.append(self.allocation(usage))
            # Update usage for subsequent write heads
            usage = usage + alloc_weights[i] * (1 - usage)

        # Stack the allocation weights for all write heads
        alloc_weights = torch.stack(alloc_weights, dim=1)

        return alloc_weights

    def allocation(self, usage):
        """
        Calculates allocation weights given usage vector.

        Args:
            usage: Tensor of shape (batch_size, memory_size)

        Returns:
            Allocation weights of shape (batch_size, memory_size)
        """
        # Sort usage in ascending order
        sorted_usage, indices = torch.sort(usage, dim=1)

        # Calculate the product of (1 - sorted_usage) for all memory locations
        # This is a cumulative product along the memory dimension
        # Shape: (batch_size, memory_size)
        prod_sorted_usage = torch.cumprod(1 - sorted_usage, dim=1)

        # Shift the product right by one position and fill with ones
        # This ensures the first location uses (1 - sorted_usage[0])
        shifted_prod = torch.cat([torch.ones(self.batch_size, 1), prod_sorted_usage[:, :-1]], dim=1)

        # Calculate allocation weights for sorted indices
        sorted_allocation = sorted_usage * shifted_prod

        # Gather the sorted allocation weights back to the original indices
        # Create indices for the batch dimension
        batch_indices = torch.arange(self.batch_size).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, self.memory_size)

        # Create indices to gather from sorted to original order
        gather_indices = torch.stack([batch_indices, indices], dim=2)

        # Gather elements from sorted_allocation using the indices
        alloc = torch.gather(sorted_allocation, 1, indices.argsort(dim=1))

        return alloc

    def update_memory_data(self, write_weights, erase_vector, write_vector):
        """Update memory using write weights, erase vector and write vector with detailed debugging."""

        print("\n==> ENTER update_memory_data")
        # Print input values
        print(
            f"write_weights shape: {write_weights.shape}, mean: {write_weights.mean().item():.6f}"
        )
        print(f"erase_vector shape: {erase_vector.shape}, mean: {erase_vector.mean().item():.6f}")
        print(f"write_vector shape: {write_vector.shape}, mean: {write_vector.mean().item():.6f}")

        # Print initial memory state
        print("\nBefore Update:")
        print(
            f"Memory shape: {self.state['memory'].shape}, mean: {self.state['memory'].mean().item():.6f}"
        )
        print(f"First row sample: {self.state['memory'][0, 0, :5].tolist()}")
        print(f"Second row sample: {self.state['memory'][0, 1, :5].tolist()}")

        # Reshape for batch matmul
        expanded_write_weights = write_weights.unsqueeze(3)  # [b, w, m, 1]
        expanded_erase_vector = erase_vector.unsqueeze(2)  # [b, w, 1, d]
        print(f"adapted, {expanded_erase_vector.shape=}")
        # print(f"adapted, {expanded_erase_vector=}")

        # Calculate erase contribution
        erase = expanded_write_weights @ expanded_erase_vector  # [b, w, m, d]
        print(f"adapted, {erase.shape=}")
        # print(f"adapted, {erase=}")
        weighted_erase = erase.sum(dim=1)  # [b, m, d]
        keep = 1 - weighted_erase

        # Debug erase calculations
        print("\nErase Calculations:")
        print(f"erase shape: {erase.shape}, mean: {erase.mean().item():.6f}")
        print(
            f"weighted_erase shape: {weighted_erase.shape}, mean: {weighted_erase.mean().item():.6f}"
        )
        print(f"keep shape: {keep.shape}, mean: {keep.mean().item():.6f}")

        # Calculate write contribution
        expanded_write_weights = write_weights.unsqueeze(3)  # [b, w, m, 1]
        expanded_write_vector = write_vector.unsqueeze(2)  # [b, w, 1, d]
        write = expanded_write_weights @ expanded_write_vector  # [b, w, m, d]
        weighted_write = write.sum(dim=1)  # [b, m, d]

        # Debug write calculations
        print("\nWrite Calculations:")
        print(f"write shape: {write.shape}, mean: {write.mean().item():.6f}")
        print(
            f"weighted_write shape: {weighted_write.shape}, mean: {weighted_write.mean().item():.6f}"
        )

        # Debug final calculation
        memory_keep = self.state["memory"] * keep
        print(f"memory * keep mean: {memory_keep.mean().item():.6f}")
        print(f"+ weighted_write mean: {weighted_write.mean().item():.6f}")

        # Update memory
        self.state["memory"] = self.state["memory"] * keep + weighted_write

        # Print final memory state
        print("\nAfter Update:")
        print(
            f"Memory shape: {self.state['memory'].shape}, mean: {self.state['memory'].mean().item():.6f}"
        )
        print(f"First row sample: {self.state['memory'][0, 0, :5].tolist()}")
        print(f"Second row sample: {self.state['memory'][0, 1, :5].tolist()}")

        # For backward compatibility
        self.memory_data = self.state["memory"]

        # Return the updated memory
        return self.state["memory"]

    def update_linkage(self, write_weights):
        """
        Updates and returns the linkage matrix and precedence weights.

        Args:
            write_weights: Tensor of shape (batch_size, num_writes, memory_size)

        Returns:
            Tuple of updated linkage matrix and precedence weights
        """
        # Create a 2D identity matrix of size memory_size
        eye = torch.eye(self.memory_size).unsqueeze(0).unsqueeze(0)

        batch_size = write_weights.size(0)

        # Initialize link for this time step to be same as eye
        updated_link = self.state["link"].clone()
        updated_precedence = self.state["precedence_weights"].clone()

        # Loop over each write head
        for i in range(self.num_writes):
            # Get current write weights for this head
            write_weights_i = write_weights[:, i].unsqueeze(1)  # (batch_size, 1, memory_size)

            # Calculate precedence vector (follows one-hot write)
            # Decay the precedence weights by the sum of the new write weights
            decay = 1 - write_weights_i.sum(dim=2, keepdim=True)  # (batch_size, 1, 1)

            # Calculate the new precedence weights
            updated_precedence[:, i] = (
                decay.squeeze(1) * updated_precedence[:, i] + write_weights[:, i]
            )

            # Calculate the link matrix update
            outer_product = torch.matmul(
                write_weights_i.transpose(1, 2), updated_precedence[:, i].unsqueeze(1)
            )  # (batch_size, memory_size, 1)

            # Add the outer product to the link matrix, zeroing the diagonal
            updated_link[:, i] = (1 - eye) * (
                updated_link[:, i] * (1 - write_weights_i) * (1 - write_weights_i.transpose(1, 2))
                + outer_product * write_weights_i.transpose(1, 2)
            )

        return updated_link, updated_precedence

    def update_read_weights(self, link, read_modes, content_weights):
        """
        Updates and returns the read weights.

        Args:
            link: Tensor of shape (batch_size, num_writes, memory_size, memory_size)
            read_modes: Tensor of shape (batch_size, num_reads, 3)
            content_weights: Tensor of shape (batch_size, num_reads, memory_size)

        Returns:
            The updated read weights
        """
        # Initialize output variable
        updated_read_weights = torch.zeros_like(self.state["read_weights"])

        # Get forward/backward weights using the directional_read_weights function
        forward_weights = self.directional_read_weights(link, self.state["read_weights"], True)
        backward_weights = self.directional_read_weights(link, self.state["read_weights"], False)

        # Calculate content mode weights (from content addressing)
        content_mode = read_modes[:, :, 0].unsqueeze(2) * content_weights

        # Loop over each read head
        for i in range(self.num_reads):
            # Calculate backward and forward mode contributions
            backward_mode = torch.zeros(self.batch_size, self.memory_size).to(
                content_weights.device
            )
            forward_mode = torch.zeros(self.batch_size, self.memory_size).to(content_weights.device)

            # Add contributions from each write head
            for j in range(self.num_writes):
                backward_mode += read_modes[:, i, j + 1].unsqueeze(1) * backward_weights[:, j, i]
                forward_mode += (
                    read_modes[:, i, j + 1 + self.num_writes].unsqueeze(1)
                    * forward_weights[:, j, i]
                )

            # Combine all modes to get final read weights
            updated_read_weights[:, i] = content_mode[:, i] + backward_mode + forward_mode

        return updated_read_weights

    def directional_read_weights(self, link, read_weights, forward):
        """
        Calculates directional read weights.

        Args:
            link: Tensor of shape (batch_size, num_writes, memory_size, memory_size)
            read_weights: Tensor of shape (batch_size, num_reads, memory_size)
            forward: Boolean indicating whether to move forward

        Returns:
            Directional read weights of shape (batch_size, num_writes, num_reads, memory_size)
        """
        # Initialize output tensor
        batch_size = link.size(0)
        result = torch.zeros(batch_size, self.num_writes, self.num_reads, self.memory_size).to(
            link.device
        )

        # Handle each write head and read head individually
        for w in range(self.num_writes):
            # Get the link matrix for this write head
            write_link = link[:, w]  # (batch_size, memory_size, memory_size)

            # Choose direction
            if not forward:
                write_link = write_link.transpose(1, 2)

            # Apply for each read head
            for r in range(self.num_reads):
                # Get read weights for this read head
                rw = read_weights[:, r]  # (batch_size, memory_size)

                # Calculate directional weights by matrix multiplication
                # write_link: (batch_size, memory_size, memory_size)
                # rw: (batch_size, memory_size) -> (batch_size, memory_size, 1)
                directional_weights = torch.matmul(write_link, rw.unsqueeze(2)).squeeze(2)

                # Store the result
                result[:, w, r] = directional_weights

        return result

    def print_memory_state(self):
        print("\n==> print_memory_state in memory_adapted")
        print(f"{self.memory_data.norm()=}")
        print(f"{self.read_weights.norm()=}")
        print(f"{self.write_weights.norm()=}")
        print(f"{self.precedence_weights.norm()=}")
        print(f"{self.link.norm()=}")

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
