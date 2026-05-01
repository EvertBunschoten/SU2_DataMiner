# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'SU2 DataMiner'
copyright = '2025, E.C.Bunschoten'
author = 'E.C.Bunschoten'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']
html_logo="../images/SU2DataMiner_logo.png"
templates_path = ['_templates']
exclude_patterns = []

add_module_names = False
show_authors=True 
numpydoc_show_class_members = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options={"show_nav_level": 3}
html_static_path = ['_static']
