
    def update(self, interface):
        """
        Updates memory given the interface parameters.

        Args:
            interface: Dictionary containing interface vectors

        Returns:
            Tensor of read words
        """
        print("UPDATE, memory_adapted")
        # Get values from interface
        read_keys = interface["read_keys"]
        read_strengths = interface["read_strengths"]
        write_keys = interface["write_keys"]
        write_strengths = interface["write_strengths"]
        erase_vectors = interface["erase_vectors"]
        write_vectors = interface["write_vectors"]
        free_gate = interface["free_gate"]
        allocation_gate = interface["allocation_gate"]
        write_gate = interface["write_gate"]
        read_modes = interface["read_modes"]

        # Update memory state
        # Calculate read/write content weights based on keys and strengths
        read_content_weights = self.content_based_address(
            self.state["memory"], read_keys, read_strengths
        )

        write_content_weights = self.content_based_address(
            self.state["memory"], write_keys, write_strengths
        )

        # Update usage vector using free gate and read weights
        self.state["usage"] = self.update_usage(free_gate)

        # Update write weights using allocation gate and write content weights
        self.state["write_weights"] = self.update_write_weights(
            self.state["usage"], write_gate, allocation_gate, write_content_weights
        )

        # Update memory using write weights, erase vectors, and write vectors
        self.state["memory"] = self.update_memory_data(
            self.state["write_weights"], erase_vectors, write_vectors
        )

        # Update temporal link matrix using write weights
        self.state["link"], self.state["precedence_weights"] = self.update_linkage(
            self.state["write_weights"]
        )

        # Update read weights using link matrix, read modes, and content weights
        self.state["read_weights"] = self.update_read_weights(
            self.state["link"], read_modes, read_content_weights
        )

        # Calculate read words based on memory data and read weights
        read_words = torch.matmul(self.state["read_weights"], self.state["memory"])

        # For backward compatibility
        self.memory_data = self.state["memory"]
        self.read_weights = self.state["read_weights"]
        self.write_weights = self.state["write_weights"]
        self.link = self.state["link"]
        self.precedence_weights = self.state["precedence_weights"]
        self.usage = self.state["usage"]

        return read_words



    def update_memory_data(self, write_weights, erase_vector, write_vector):
        print("UPDATE, memory_data")
        """Update memory using write weights, erase vector and write vector with detailed debugging."""
        print("\n=== Memory Update Debugging (Adapted Implementation) ===")

        # Print input values
        print("\nInput Values:")
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

        # Calculate erase contribution
        erase = expanded_write_weights @ expanded_erase_vector  # [b, w, m, d]
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

