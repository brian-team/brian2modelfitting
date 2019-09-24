# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

needs_sphinx = '1.8'


brian2modelfitting_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'brian2modelfitting'))
# -- Project information -----------------------------------------------------

project = 'Brian2modelfitting'
copyright = '2019, brian-team'
author = 'brian-team'

# The full version, including alpha/beta/rc tags
pkg_version = {}
with open(os.path.join(brian2modelfitting_dir, 'version.py')) as fp:
    exec(fp.read(), pkg_version)
release = pkg_version['version']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# Make `...` link to Python classes/functions/methods/...
default_role = 'py:obj'

# autodoc configuration
autodoc_default_options = {'inherited-members': True}

# -- Options for HTML output -------------------------------------------------
# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:
    # ReadTheDocs theme
    try:
        import sphinx_rtd_theme
        html_theme = "sphinx_rtd_theme"
        html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    except ImportError:
        pass  # use the default theme


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# #
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = '_static/brian-logo.png'


# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {'python': ('https://docs.python.org/', None),
                       'brian2': ('https://brian2.readthedocs.org/en/stable/', None),
                       'matplotlib': ('http://matplotlib.org/', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)}
