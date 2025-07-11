# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../tensorclouds"))

project = "tensorclouds"
copyright = "2024, Center for Bits and Atoms, MIT Media Lab Molecular Machines"
author = "Allan Costa, Ilan Mitnikov, Manvitha Ponnapati"
release = "-"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.viewcode',
    # 'sphinx.ext.napoleon',
    # 'autoapi.extension',
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = ["../tensorclouds"]
autoapi_template_dir = "_templates/apidoc"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_use_implicit_namespaces = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "insegel"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
