# parallel_controller_moe.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Integer

from dnc_torch_zeligism_polymorphic.dnc_adapted import DNC_Adapted
from dnc_torch_zeligism_polymorphic.rms_norm_layer import RMSNormLayer

Tensor = torch.Tensor


@beartype
class ParallelControllerMoE(nn.Module):
    """
    Controller with multiple parallel memory units using Mixture of Experts (MoE).

    This class implements a controller that selectively routes input through multiple
    parallel DNC units (experts) based on a router network. The router determines
    which experts to use for each input and how to weight their contributions.
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        controller_config: dict,
        memory_config: dict,
        use_rms_norm: bool = False,
        top_k: int = 2,
        capacity_factor: float = 1.5,
        router_jitter: float = 0.01,
    ):
        """
        Initialize the ParallelControllerMoE.

        Args:
            num_experts: Number of expert memory units.
            input_size: Size of the input features.
            output_size: Size of the output features.
            controller_config: Configuration for the controller.
            memory_config: Configuration for the memory units.
            use_rms_norm: Whether to use RMS normalization.
            top_k: Number of experts to route each input to.
            capacity_factor: Factor to determine expert capacity.
            router_jitter: Noise added to router logits during training.
        """
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.use_rms_norm = use_rms_norm
        self.top_k = min(top_k, num_experts)  # Ensure top_k <= num_experts
        self.capacity_factor = capacity_factor
        self.router_jitter = router_jitter

        # Create router network
        self.router = nn.Linear(input_size, num_experts)

        # Create expert DNC units
        self.experts = nn.ModuleList(
            [
                DNC_Adapted(input_size, output_size, controller_config, memory_config)
                for _ in range(num_experts)
            ]
        )

        # Create RMS normalization layers if needed
        if use_rms_norm:
            self.rms_norms = nn.ModuleList([RMSNormLayer(output_size) for _ in range(num_experts)])

    def _compute_router_probabilities(
        self,
        x: Float[Tensor, "sequence_length batch_size input_size"],
    ) -> tuple[
        Float[Tensor, "seq_len_batch_size num_experts"],
        Integer[Tensor, "seq_len_batch_size num_experts"],
        Float[Tensor, ""],
    ]:
        """
        Compute routing probabilities for each input token.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, input_size).

        Returns:
            Tuple of (routing_weights, expert_indices, load_balancing_loss)
            routing_weights: Float[Tensor, "sequence*length_batch_size num_experts"]
            expert_indices: Integer[Tensor, "sequence_length*batch_size num_experts"]
            load_balancing_loss: Float[Tensor, ""]
        """
        # Reshape input for routing: (sequence_length * batch_size, input_size)
        seq_len, batch_size, _ = x.shape
        flat_x = x.reshape(-1, self.input_size)

        # Compute router logits
        router_logits = self.router(flat_x)

        # Add noise during training
        if self.training and self.router_jitter > 0:
            router_logits += torch.randn_like(router_logits) * self.router_jitter

        # Calculate expert capacity
        # Each expert should handle (tokens_per_batch * capacity_factor / num_experts) tokens
        tokens_per_batch = seq_len * batch_size
        capacity = int(tokens_per_batch * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)  # Ensure minimum capacity of 1

        # Get top-k experts and their routing probabilities
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=1)
        routing_weights = F.softmax(routing_weights, dim=1)

        # Compute load balancing loss (optional, can be used during training)
        # This encourages uniform expert utilization
        router_probs = F.softmax(router_logits, dim=1)
        load = router_probs.sum(0)
        load_balancing_loss = (self.num_experts * load).pow(2).mean()

        return routing_weights, expert_indices, load_balancing_loss

    def forward(
        self,
        x: Float[Tensor, "sequence_length batch_size input_size"],
    ) -> Float[Tensor, "sequence_length batch_size output_size"]:
        """
        Forward pass through the MoE controller.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, input_size).

        Returns:
            Output tensor of shape (sequence_length, batch_size, output_size).
        """
        seq_len, batch_size, _ = x.shape

        # Compute routing probabilities and expert assignments
        routing_weights, expert_indices, load_balancing_loss = self._compute_router_probabilities(x)

        # Initialize output tensor
        output = torch.zeros(seq_len * batch_size, self.output_size, device=x.device)

        # Process through selected experts
        flat_x = x.reshape(seq_len * batch_size, -1)

        for k in range(self.top_k):
            # Get the expert indices and weights for the k-th selection
            expert_idx = expert_indices[:, k]
            weight = routing_weights[:, k].unsqueeze(1)  # Add dimension for broadcasting

            # Group inputs by expert
            # TODO: This is a bit of a hack. We should be able to do this without the loop.
            for expert_id in range(self.num_experts):
                # Find which inputs are routed to this expert
                mask = expert_idx == expert_id
                if not mask.any():
                    continue

                # Get the inputs for this expert (seq_len * batch_size, input_size)
                expert_inputs = flat_x[mask]

                # Reshape back to sequence format for DNC processing
                # This is tricky and requires tracking original positions
                positions = torch.where(mask)[0]
                seq_positions = positions // batch_size
                batch_positions = positions % batch_size

                # Create a padded input tensor for this expert
                expert_seq_len = len(expert_inputs)
                if expert_seq_len == 0:
                    continue

                # Reshape expert inputs for DNC processing
                # We need to carefully reconstruct the sequence structure
                expert_x = torch.zeros(seq_len, batch_size, self.input_size, device=x.device)
                for i, (seq_pos, batch_pos) in enumerate(zip(seq_positions, batch_positions)):
                    # print(f"{expert_inputs[i].shape=}")
                    expert_x[seq_pos, batch_pos] = expert_inputs[i]

                # Process through the expert
                expert_output = self.experts[expert_id](expert_x)

                # Apply RMS normalization if enabled
                if self.use_rms_norm:
                    expert_output = self.rms_norms[expert_id](expert_output)

                # Extract outputs for the positions that were actually processed
                for i, (seq_pos, batch_pos) in enumerate(zip(seq_positions, batch_positions)):
                    output[positions[i]] += weight[positions[i]] * expert_output[seq_pos, batch_pos]

        # Reshape output back to sequence format
        output = output.reshape(seq_len, batch_size, self.output_size)

        # Store load balancing loss for potential use in training
        self.load_balancing_loss = load_balancing_loss

        return output

    def detach_state(self) -> None:
        """Detach the state of all expert DNC units from the computation graph."""
        for expert in self.experts:
            expert.detach_state()  # type: ignore (disable pylance)


# ----------------------------------------------------------------------
# Test the ParallelControllerMoE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    num_experts = 4
    input_size = 10
    output_size = 5
    controller_config = {"hidden_size": 64, "num_layers": 1}
    memory_config = {
        "memory_size": 128,
        "word_size": 20,
        "num_reads": 4,
        "num_writes": 1,
        "batch_size": 8,
    }

    # Create a MoE controller
    model = ParallelControllerMoE(
        num_experts=num_experts,
        input_size=input_size,
        output_size=output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        use_rms_norm=True,
        top_k=2,
    )

    # Generate random input sequence
    seq_length = 5
    batch_size = 8
    x = torch.randn(seq_length, batch_size, input_size)

    # Forward pass
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Load balancing loss: {model.load_balancing_loss.item()}")

    # Test with different top_k values
    for top_k in [1, 2, 3]:
        model = ParallelControllerMoE(
            num_experts=num_experts,
            input_size=input_size,
            output_size=output_size,
            controller_config=controller_config,
            memory_config=memory_config,
            use_rms_norm=True,
            top_k=top_k,
        )
        y = model(x)
        print(f"Top-k: {top_k}, Output shape: {y.shape}")


# Will have to run the training loop to find out if memory is used and updated correctly.
