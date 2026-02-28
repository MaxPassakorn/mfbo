# Configuration file for the Sphinx documentation builder.

import os
import sys

# Make project importable (mfbo)
sys.path.insert(0, os.path.abspath("../.."))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mfbo'
copyright = '2026, Passakorn Paladaechanan'
author = 'Passakorn Paladaechanan'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",                 # Markdown support
    "sphinx.ext.autodoc",          # Auto-generate API docs
    "sphinx.ext.napoleon",         # Google / NumPy docstrings
    "sphinx.ext.viewcode",         # Link to source code
    "sphinx.ext.mathjax",          # REQUIRED for LaTeX math
]

master_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
