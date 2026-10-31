"""Sphinx configuration: the API reference is generated from the docstrings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "bspf-core"
author = "BSPF project"
release = "0.1.0.dev0"

extensions = [
    "sphinx.ext.autodoc",        # pull documentation from docstrings
    "sphinx.ext.napoleon",       # NumPy-style sections (Parameters, Returns, ...)
    "sphinx.ext.mathjax",        # render :math: and .. math::
    "sphinx.ext.viewcode",       # link documentation to highlighted source
    "sphinx.ext.intersphinx",    # link to NumPy/SciPy/JAX documentation
    "myst_parser",               # Markdown pages
]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": True}
autodoc_typehints = "none"
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
myst_enable_extensions = ["dollarmath"]
html_theme = "furo"
html_title = "bspf-core"
exclude_patterns = ["_build"]
