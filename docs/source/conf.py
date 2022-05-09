# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Mocafe'
copyright = '2021, Franco Pradelli'
author = 'Franco Pradelli'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx_gallery.gen_gallery',
              'sphinxcontrib.bibtex',
              'sphinxcontrib.youtube',
              'sphinx_rtd_theme',]

# configure sphinx-gallery
sphinx_gallery_conf = {
     'examples_dirs': '../../demo',   # path to your example scripts
     'gallery_dirs': 'demo_doc',  # path to where to save gallery generated output
}

# add logo
html_logo = '200mocafe_logo.png'
# configure bibfile
bibtex_bibfiles = ['ref.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ["css/class.css", "css/functions.css"]