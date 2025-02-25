from abc import ABC, abstractmethod
from typing import Dict, Tuple

from torch import Tensor, nn
from beartype import beartype

from dnc.mixins import StateMixin


@beartype
class BaseController(nn.Module, StateMixin):
    """Base class for controllers with state management.

    # StateMixin derives from ABC, so ABC cannot be included explictly.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.state: Dict[str, Tensor] = {}

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the controller.

        Args:
            x: Input tensor.

        Returns:
            Output tensor (e.g., hidden state).
        """
        pass

    def get_state_dict(self) -> Dict[str, Tensor]:
        """Return the full state dictionary.

        Returns:
            Dictionary containing all state variables.
        """
        return self.state

    def summary(self) -> str:
        """Return a string summary of the controller's architecture.

        Returns:
            A string containing input/output shapes and parameters.
        """
        return (
            f"\nController Summary:\n"
            f"  Input shape: {self.input_shape if hasattr(self, 'input_shape') else 'Not set'}\n"
            f"  Output shape: {self.output_shape if hasattr(self, 'output_shape') else 'Not set'}\n"
            f"  Parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

@beartype
class BaseInterface(nn.Module, StateMixin):
    """Base class for interface modules with state management.

    # StateMixin derives from ABC, so ABC cannot be included explictly.
    """

    def __init__(self):
        super().__init__()
        self.state: Dict[str, Tensor] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}

    @abstractmethod
    def forward(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Decompose the controller's state into interface vectors.

        Args:
            state_dict: Dictionary containing the controller's state.

        Returns:
            Dictionary of interface vectors (e.g., read keys, write vector).
        """
        pass

    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return the shapes of all output vectors.

        Returns:
            Dictionary mapping vector names to their shapes.
        """
        return self.output_shapes

    def summary(self) -> str:
        """Return a string summary of the interface's architecture.

        Returns:
            A string containing input/output shapes and parameters.
        """
        return (
            f"\nInterface Summary:\n"
            f"  Input shape: {self.input_shape if hasattr(self, 'input_shape') else 'Not set'}\n"
            f"  Output shapes:\n"
            f"    " + "\n    ".join(f"{k}: {v}" for k, v in self.output_shapes.items()) + "\n"
            f"  Parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

@beartype
class BaseMemory(nn.Module, ABC, StateMixin):
    """Base class for memory modules with state management.

    # StateMixin derives from ABC, so ABC cannot be included explictly.
    """

    def __init__(self):
        super().__init__()
        self.state: Dict[str, Tensor] = {}
        self.interface: BaseInterface | None = None
        self.required_shapes: Dict[str, Tuple[int, ...]] = {}

    def set_interface(self, interface: BaseInterface) -> None:
        """Set the interface module and validate compatibility.

        Args:
            interface: Interface module to use.

        Raises:
            ValueError: If interface outputs don't match memory requirements.
        """
        self.interface = interface
        self.validate_interface()

    def validate_interface(self) -> None:
        """Validate that the interface outputs match memory requirements.

        Raises:
            ValueError: If shapes don't match or required vectors are missing.
        """
        if not self.interface:
            return

        interface_shapes = self.interface.get_output_shapes()

        for name, required_shape in self.required_shapes.items():
            if name not in interface_shapes:
                raise ValueError(f"Required vector '{name}' not provided by interface")
            if interface_shapes[name] != required_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"required {required_shape}, got {interface_shapes[name]}"
                )

    def summary(self) -> str:
        """Return a string summary of the memory's architecture.

        Returns:
            A string containing memory configuration and requirements.
        """
        return (
            f"\nMemory Summary:\n"
            f"  Memory shape: {self.state['memory'].shape if 'memory' in self.state else 'Not initialized'}\n"
            f"  Required interface outputs:\n"
            f"    " + "\n    ".join(f"{k}: {v}" for k, v in self.required_shapes.items()) + "\n"
            f"  Interface attached: {self.interface is not None}\n"
            f"  Parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

    @abstractmethod
    def forward(self, state_dict: Dict[str, Tensor]) -> Tensor:
        """Forward pass through the memory module.

        Args:
            state_dict: Dictionary containing the controller's state.

        Returns:
            Updated memory tensor.
        """
        pass
