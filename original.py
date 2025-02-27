
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

        # Update state of memory and return read words
        self.usage = usage_t
        self.write_weights = write_weights_t
        self.memory_data = memory_data_t
        self.link = link_t
        self.precedence_weights = precedence_weights_t
        self.read_weights = read_weights_t

        # Return the new read words for each read head from new memory data
        return read_weights_t @ memory_data_t



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
        #print(f"{erase_factor=}")

        # Calculate the weighted words to add/write to memory.
        write_words = weights.transpose(1, 2) @ writes
        #print(f"{write_words=}")
        #print(f"{erase_factor=}")

        # Return the updated memory
        return self.memory_data * erase_factor + write_words

