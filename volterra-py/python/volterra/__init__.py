"""volterra: active nematics simulation, in Rust.

The whole API lives in the compiled extension; this module re-exports it and
carries the type stubs beside it (`__init__.pyi`, `py.typed`), so a type checker
sees the same names the interpreter does.
"""

from .volterra import *  # noqa: F401,F403
from . import volterra as _ext

__doc__ = _ext.__doc__
if hasattr(_ext, "__all__"):
    __all__ = _ext.__all__
