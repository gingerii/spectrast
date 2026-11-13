"""Sphinx configuration for the Spectrast documentation (scverse-style)."""
import os
import sys
from datetime import datetime

# Make the package importable for autodoc even when not pip-installed (RTD installs it via
# .readthedocs.yaml, but this keeps local `sphinx-build docs docs/_build` working too).
sys.path.insert(0, os.path.abspath("../src"))

# -- Project ----------------------------------------------------------------
project = "Spectrast"
author = "Ian Gingerich"
copyright = f"{datetime.now():%Y}, {author}"
try:
    from spectrast import __version__ as release
except Exception:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("spectrast")
version = release

# -- Extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # numpydoc-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",                   # renders .md pages AND executes/renders .ipynb tutorials
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_mock_imports = ["torch", "scanpy", "anndata", "sklearn", "matplotlib"]  # optional/heavy at build

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Notebooks are pre-executed and committed WITH outputs (scverse convention); RTD just renders.
nb_execution_mode = "off"

myst_enable_extensions = ["colon_fence", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "squidpy": ("https://squidpy.readthedocs.io/en/stable", None),
}

# -- HTML -------------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Spectrast"
html_theme_options = {
    "repository_url": "https://github.com/gingerii/spectrast",
    "use_repository_button": True,
    "use_issues_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "home_page_in_toc": True,
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Signatures -------------------------------------------------------------
# Co-organization and orientation ship in the package but are held out of these docs (a separate
# publication). These parameters of documented functions serve only those features, so the rendered
# signature leaves them out; their docstrings already do.
HIDDEN_PARAMS = {
    "spectrast.gr.fingerprint_presence": {"orientation", "orient_n_perm", "orient_kw"},
    "spectrast.diff.differential": {"pairs"},
}


def _hide_params(app, what, name, obj, options, signature, return_annotation):
    hidden = HIDDEN_PARAMS.get(name)
    if not hidden or signature is None:
        return None
    import inspect
    sig = inspect.signature(obj)
    kept = [p for p in sig.parameters.values() if p.name not in hidden]
    return str(sig.replace(parameters=kept)), return_annotation


def setup(app):
    app.connect("autodoc-process-signature", _hide_params)
