"""Shared test configuration: 64-bit JAX and paths to reference implementations."""
from pathlib import Path
import importlib.util

import jax
import pytest

jax.config.update("jax_enable_x64", True)

HANDOFF = Path(__file__).resolve().parents[3]


def load_reference(relative, name):
    """Import a module from the hand-off repository by path, or skip the test."""
    path = HANDOFF / relative
    if not path.exists():
        pytest.skip(f"reference implementation not available: {relative}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
