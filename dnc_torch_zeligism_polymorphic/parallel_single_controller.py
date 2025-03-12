# parallel_controller.py


import sys
from pprint import pprint

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from dnc_torch_zeligism_polymorphic.configuration import (
    controller_config,
    memory_config,
    training_config,
)
from dnc_torch_zeligism_polymorphic.dnc_adapted import DNC_Adapted
from dnc_torch_zeligism_polymorphic.rms_norm_layer import RMSNormLayer


@beartype
class ParallelSingleController(nn.Module):
    """
    Controller with multiple parallel memory units.

    This class implements a controller that processes input through multiple parallel
    DNC units, each with its own memory. The outputs from all units are combined to
    produce the final output.

    The output should be identical to DNC_Adapted.
    """

    def __init__(
        self,
        num_memories: int,
        input_size: int,
        output_size: int,
        controller_config: dict,
        memory_config: dict,
        use_projections: bool = True,
        use_rms_norm: bool = False,
        combination_method: str = "concat",
    ):
        """
        Initialize the ParallelSingleController.

        Args:
            num_memories: Number of parallel memory units.
            input_size: Size of the input features.
            output_size: Size of the output features.
            controller_config: Configuration for the controller.
            memory_config: Configuration for the memory units.
            use_projections: Whether to use projection layers for each branch.
            use_rms_norm: Whether to use RMS normalization.
            combination_method: Method to combine outputs ("concat", "sum", or "mean").
        """
        super().__init__()
        self.num_memories = num_memories
        self.input_size = input_size
        self.output_size = output_size
        self.use_projections = use_projections
        self.use_rms_norm = use_rms_norm
        self.combination_method = combination_method

        #  FORCE FOR DEBUGGING
        use_projections = False

        # Create projection layers if needed
        if use_projections:
            self.projections = nn.ModuleList(
                [nn.Linear(input_size, input_size) for _ in range(num_memories)]
            )

        # Routine specialized for a single parallel memory
        num_memories = 1
        print(f"Force {num_memories=}")
        print(f"{input_size=}, {output_size=}")
        print(f"{controller_config=}")
        print(f"{memory_config=}")

        # Create parallel DNC units
        self.dnc_units = nn.ModuleList(
            [
                DNC_Adapted(input_size, output_size, controller_config, memory_config)
                for _ in range(num_memories)
            ]
        )

        # print(f"{self.dnc_units=}")

        self.use_projections = False
        print(f"{self.use_projections=} is set to False")

        self.use_rms_norm = False
        print(f"{self.use_rms_norm=}")

        # Create RMS normalization layers if needed
        if self.use_rms_norm:
            self.rms_norms = nn.ModuleList([RMSNormLayer(output_size) for _ in range(num_memories)])

        print(f"{combination_method=} should have no effect if only single memory module.")

        # Create output layer based on combination method
        print(f"{combination_method=}")
        if combination_method == "concat":
            print("top if")
            if num_memories > 1:
                print("top if, num_memories = ", num_memories)
                self.output_layer = nn.Linear(output_size * num_memories, output_size)
        elif combination_method in ["sum", "mean"]:
            # No additional layer needed for sum or mean
            pass
        else:
            raise ValueError(f"Unsupported combination method: {combination_method}")

    def forward(
        self,
        x: Float[Tensor, "seq_len batch_size input_size"],
    ) -> Float[Tensor, "seq_len batch_size output_size"]:
        """
        Forward pass through the parallel controller.

        Args:
            x: Input tensor of shape (sequence_length, batch_size, input_size).

        Returns:
            Output tensor of shape (sequence_length, batch_size, output_size).
        """
        # Process input through each parallel branch
        branch_outputs = []
        print("==> forward, ParallelSingleController")

        for i in range(self.num_memories):
            # Apply projection if enabled
            if self.use_projections:
                # We need to apply the projection to each time step
                # Each x_t is a tensor of shape (batch_size, input_size)
                projected_x = torch.stack([self.projections[i](x_t) for x_t in x])
            else:
                projected_x = x

            # Process through DNC unit
            print("......i= ", i)
            branch_output = self.dnc_units[i](projected_x)

            # Apply RMS normalization if enabled
            if self.use_rms_norm:
                branch_output = self.rms_norms[i](branch_output)

            branch_outputs.append(branch_output)

        branch_outputs: list[Float[Tensor, "seq_len batch_size output_size"]] = branch_outputs

        # Combine outputs based on the specified method
        if self.combination_method == "concat":
            if self.num_memories > 1:
                # Concatenate along the feature dimension
                combined = torch.cat(branch_outputs, dim=2)
                # Project back to output_size
                output = self.output_layer(combined)
            else:
                output = branch_outputs[0]

        elif self.combination_method == "sum":
            # Sum the outputs
            output = torch.sum(torch.stack(branch_outputs), dim=0)

        elif self.combination_method == "mean":
            # Average the outputs
            output = torch.mean(torch.stack(branch_outputs), dim=0)
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")

        # specify the type of output for type linter
        output: Float[Tensor, "seq_len batch_size output_size"] = output
        # print(f"output, {output.shape=}, {type(output[0,0,0].item())=}")
        return output

    def detach_state(self) -> None:
        """Detach the state of all DNC units from the computation graph."""
        for dnc in self.dnc_units:
            dnc.detach_state()  # type: ignore


if __name__ == "__main__":
    # Test the ParallelSingleController
    num_memories = 1
    input_size = 8
    output_size = 5
    controller_config = {"hidden_size": 64, "num_layers": 1}
    memory_config["batch_size"] = training_config["batch_size"]

    # Create a parallel controller
    model = ParallelSingleController(
        num_memories=num_memories,
        input_size=input_size,
        output_size=output_size,
        controller_config=controller_config,
        memory_config=memory_config,
        use_projections=True,
        use_rms_norm=True,
        combination_method="concat",
    )

    # Generate random input sequence
    seq_length = 5
    batch_size = 8
    x = torch.randn(seq_length, batch_size, input_size)

    # Forward pass
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test different combination methods
    for method in ["concat", "sum", "mean"]:
        print("========================")
        model = ParallelSingleController(
            num_memories=num_memories,
            input_size=input_size,
            output_size=output_size,
            controller_config=controller_config,
            memory_config=memory_config,
            use_projections=True,
            use_rms_norm=True,
            combination_method=method,
        )
        y = model(x)
        print(f"Combination method: {method}, Output shape: {y.shape}")
