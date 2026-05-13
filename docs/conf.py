import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('..'))

# Mock heavy GUI dependencies before importing torc, since torc.py imports them
# at module level and they aren't available in the ReadTheDocs build environment.
_gui_mocks = [
    'pyqtgraph',
    'pyqtgraph.opengl',
    'pyqtgraph.Qt',
    'pyqtgraph.Qt.QtCore',
    'PySide6',
    'OpenGL',
    'OpenGL.GL',
]
for _mod in _gui_mocks:
    sys.modules[_mod] = MagicMock()

from torc import __version__

# Also mock for autodoc when it processes source files:
autodoc_mock_imports = ['pyqtgraph', 'PySide6', 'OpenGL']

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
