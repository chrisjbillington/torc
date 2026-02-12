import sys
import os

sys.path.insert(0, os.path.abspath('..'))

# Mock heavy GUI dependencies that aren't available on readthedocs:
autodoc_mock_imports = ['pyqtgraph', 'PySide6', 'OpenGL']

from torc import __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

root_doc = 'index'
project = 'torc'
copyright = '2025, Chris Billington'
version = __version__
release = '.'.join(__version__.split('.')[:2])

autodoc_member_order = 'bysource'
autoclass_content = 'both'
add_module_names = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
html_theme = 'sphinx_rtd_theme'
