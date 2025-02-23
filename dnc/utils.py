import inspect


def validate_method_signatures(cls: type) -> None:
    """Validate that subclasses implement required methods with correct signatures.

    Args:
        cls: The class to validate.

    Raises:
        TypeError: If a subclass does not implement the required methods or has incorrect type signatures.
    """
    if cls is cls.__bases__[0]:  # Skip validation for the abstract base class itself
        return

    parent = cls.__bases__[0]  # Get the immediate parent class
    for method_name in ["sum", "difference"]:
        base_method = getattr(parent, method_name, None)
        sub_method = getattr(cls, method_name, None)

        if base_method and sub_method:
            base_sig = inspect.signature(base_method)
            sub_sig = inspect.signature(sub_method)

            # Check parameter names
            if base_sig.parameters.keys() != sub_sig.parameters.keys():
                raise TypeError(
                    f"Method `{method_name}` in `{cls.__name__}` does not match "
                    f"signature {list(base_sig.parameters.keys())} of `{parent.__name__}`"
                )

            # Check parameter types
            for param_name, base_param in base_sig.parameters.items():
                sub_param = sub_sig.parameters[param_name]
                if base_param.annotation != sub_param.annotation:
                    raise TypeError(
                        f"Method `{method_name}` in `{cls.__name__}` has incorrect type "
                        f"for parameter `{param_name}`. Expected {base_param.annotation}, "
                        f"got {sub_param.annotation}"
                    )
        # Add additional signature validation logic here if needed


class Validate:
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
