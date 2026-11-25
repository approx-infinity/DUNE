"""DUNE - Nuclear Reactor Simulator."""

__version__ = "0.3.0"

__all__ = ["__version__", "DUNEReactor", "main"]


def __getattr__(name):
    if name == "DUNEReactor":
        from .reactor import DUNEReactor

        return DUNEReactor
    if name == "main":
        from .DUNEReactor import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
