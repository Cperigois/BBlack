# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import BBlack
import sphinx
import SphinxExtensions

sys.path.insert(0, os.path.abspath("../../.."))

project = 'BBLack'
copyright = '2025, Carole Périgois'
author = 'Carole Périgois'
release = '02/02/2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    "display_gitlab": True,
    "gitlab_user": "Cperigois",
    "gitlab_repo": "Princess",  # Repo name
    "gitlab_version": "master",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

html_theme = 'sphinxdoc'
html_static_path = ['_static']


if os.environ.get("READTHEDOCS"):
    html_context["commit"] = os.environ.get("READTHEDOCS_VERSION")