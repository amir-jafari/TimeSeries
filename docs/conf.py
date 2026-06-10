"""Sphinx configuration for TimeSeries Toolbox."""
import os
import sys
import shutil
import pathlib

# Make the package importable
sys.path.insert(0, os.path.abspath(".."))

# ── Copy notebooks into the Sphinx source tree at build time ─────────────────
_NB_SRC = pathlib.Path(__file__).parent.parent / "TimeSeriesSRC" / "Examples" / "NoteBooks"
_NB_DST = pathlib.Path(__file__).parent / "_notebooks"
if _NB_SRC.exists():
    shutil.copytree(str(_NB_SRC), str(_NB_DST), dirs_exist_ok=True)

# Ensure matplotlib uses a non-interactive backend for notebook execution
os.environ.setdefault("MPLBACKEND", "Agg")

# ── Project information ───────────────────────────────────────────────────────
project   = "TimeSeries Toolbox"
author    = "Amir Jafari, Martin Hagan, Lilian S. De Rivera"
copyright = "2024, Amir Jafari, Martin Hagan, Lilian S. De Rivera"
release   = "0.1.0"
version   = "0.1"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",          # pull docstrings automatically
    "sphinx.ext.napoleon",         # Google / NumPy docstring styles
    "sphinx.ext.viewcode",         # [source] links on every page
    "sphinx.ext.intersphinx",      # cross-link numpy, scipy, matplotlib docs
    "sphinx.ext.mathjax",          # render LaTeX math
    "sphinx.ext.autosummary",      # summary tables + per-item pages
    "sphinx_copybutton",           # copy button on code blocks
    "myst_parser",                 # include .md files as pages
    "nbsphinx",                    # include Jupyter notebooks
]

# ── autodoc ───────────────────────────────────────────────────────────────────
autosummary_generate  = True
autodoc_member_order  = "bysource"
autodoc_default_options = {
    "members":          True,
    "undoc-members":    True,
    "show-inheritance": True,
}
# Suppress warnings for modules that import matplotlib/scipy at module level
autodoc_mock_imports = []

# ── napoleon ──────────────────────────────────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = True

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3",        None),
    "numpy":      ("https://numpy.org/doc/stable",     None),
    "scipy":      ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable",    None),
}

# ── nbsphinx ──────────────────────────────────────────────────────────────────
# 'auto'  → use pre-executed outputs when present, execute otherwise
# 'never' → always use pre-executed outputs (fastest, safest for CI)
nbsphinx_execute = "auto"
nbsphinx_timeout = 300          # seconds per notebook

# Pass environment so inline plots use Agg
nbsphinx_execute_arguments = []

# Kernel name that matches the installed Python environment
nbsphinx_kernel_name = "python3"

# Do not add a "[ ]" prompt before code cells
nbsphinx_prolog = ""

# ── General ───────────────────────────────────────────────────────────────────
templates_path   = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_notebooks/README.md",
    "_notebooks/README",
]
source_suffix    = {".rst": "restructuredtext", ".md": "markdown"}
# nbsphinx registers .ipynb automatically — do NOT add it to source_suffix

# ── HTML theme ────────────────────────────────────────────────────────────────
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/amir-jafari/TimeSeries/",
    "source_branch":     "main",
    "source_directory":  "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url":  "https://github.com/amir-jafari/TimeSeries",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.'
                '37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-'
                '.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-'
                '.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 '
                '.67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-'
                '.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 '
                '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.'
                '55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

html_static_path = ["_static"]
html_title       = "TimeSeries Toolbox"
html_short_title = "TSToolbox"

# ── MathJax ──────────────────────────────────────────────────────────────────
mathjax3_config = {
    "tex": {
        "inlineMath":  [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# ── Copy-button — skip prompts in code blocks ─────────────────────────────────
copybutton_prompt_text      = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True