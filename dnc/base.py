from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn

from dnc.mixins import StateMixin


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


class BaseInterface(nn.Module, StateMixin):
    """Base class for interface modules with state management.

    # StateMixin derives from ABC, so ABC cannot be included explictly.
    """

    def __init__(self):
        super().__init__()
        self.state: Dict[str, Tensor] = {}

    @abstractmethod
    def forward(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Decompose the controller's state into interface vectors.

        Args:
            state_dict: Dictionary containing the controller's state.

        Returns:
            Dictionary of interface vectors (e.g., read keys, write vector).
        """
        pass


class BaseMemory(nn.Module, ABC, StateMixin):
    """Base class for memory modules with state management.

    # StateMixin derives from ABC, so ABC cannot be included explictly.
    """

    def __init__(self):
        super().__init__()
        self.state: Dict[str, Tensor] = {}
        self.interface: BaseInterface | None = None

    def set_interface(self, interface: BaseInterface) -> None:
        """Set the interface module.

        Args:
            interface: Interface module to use.
        """
        self.interface = interface

    @abstractmethod
    def forward(self, state_dict: Dict[str, Tensor]) -> Tensor:
        """Forward pass through the memory module.

        Args:
            state_dict: Dictionary containing the controller's state.

        Returns:
            Updated memory tensor.
        """
        pass
