import ipywidgets as widgets
from traitlets import Unicode, Bool, validate, TraitError
from IPython.display import clear_output

@widgets.register
class SaveActionsWidget(widgets.DOMWidget):
    _view_name = Unicode('SaveActionsView').tag(sync=True)
    _view_module = Unicode('ascent_widgets').tag(sync=True)
    _view_module_version = Unicode('0.0.0').tag(sync=True)

     # Attributes
    filename = Unicode('', help="The file name.").tag(sync=True)
    status = Unicode('', help="Status of the save operation.").tag(sync=True)
    disabled = Bool(False, help="Enable or disable user changes.").tag(sync=True)

    # Basic validator for the filename
    #@validate('filename')
    #def _valid_filename(self, proposal):
    #    raise TraitError('Invalid file name: it must end in .json or .yaml')
    #    if not proposal['filename'].endswith((".json", ".yaml")):
    #        self.status = 'Invalid file name: it must end in .json or .yaml'
    #        self.disabled = True
    #        raise TraitError('Invalid file name: it must end in .json or .yaml')
    #    else:
    #        self.status = ''
    #        self.disabled = False
    #    return proposal['filename']

    def __init__(self, kernelUtils, *args, **kwargs):
        widgets.DOMWidget.__init__(self, *args, **kwargs)

        self.is_connected = True

        self.kernelUtils = kernelUtils

        self.on_msg(self._handle_msg)

    def _handle_msg(self, msg, *args, **kwargs):
        if self.is_connected and ("content" in msg["content"]["data"]):
            content = msg["content"]["data"]["content"]
            if content['event'] == 'button' and content['code'] == 'save':
                if not self.filename.endswith((".json", ".yaml")):
                    self.status = 'Invalid file name {}. It must end in .json or .yaml'.format(self.filename)
                    self.disabled = True
                else:
                    self.disabled = False
                    self.status = self.kernelUtils.save_actions(self.filename)["status"]
                self.is_connected = (self.status is not None)
            if content['event'] == 'filename_changed':
                self.filename = content['filename']
