from ._version import version_info, __version__

from .save_actions import *

def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'ascent_widgets',
        'require': 'ascent_widgets/extension'
}]
