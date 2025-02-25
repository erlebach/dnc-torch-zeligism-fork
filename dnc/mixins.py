from abc import abstractmethod
from typing import Any, Dict

from torch import Tensor
from beartype import beartype

from dnc.utils import validate_method_signatures


@beartype
class StateMixin:
    """Mixin for state management utilities."""

    state: Dict[str, Tensor]  # Type hint for self.state

    @abstractmethod
    def init_state(self, config: Dict[str, Any]) -> None:
        """Initialize the state of the module.

        This method must be implemented by the class using the mixin.
        """
        pass

    def detach_state(self) -> None:
        """Detach the state from the graph."""
        for key, value in self.state.items():
            if isinstance(value, Tensor):
                self.state[key] = value.detach()

    def reset_state(self) -> None:
        """Reset the state dictionary to its initial values."""
        self.init_state()


@beartype
class ValidateMixin:
    """for validating subclass method signatures."""

    def __init_subclass__(cls, **kwargs) -> None:
        """Validate method signatures when a subclass is created.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            TypeError: If a subclass does not implement the required methods or has incorrect signatures.
        """
        super().__init_subclass__(**kwargs)
        validate_method_signatures(cls)
