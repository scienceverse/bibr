"""Input processing subpackage."""


def __getattr__(name):
    if name == "validate_input_file":
        from bibr.input.validate import validate_input_file

        globals()[name] = validate_input_file
        return validate_input_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "validate_input_file",
]
