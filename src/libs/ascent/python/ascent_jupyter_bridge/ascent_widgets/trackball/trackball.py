###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

import ipywidgets as widgets
from traitlets import Unicode, validate, Int, List, Dict
from IPython.display import clear_output

class TrackballWidget(widgets.DOMWidget):
    _view_name = Unicode('TrackballView').tag(sync=True)
    _view_module = Unicode('ascent_widgets').tag(sync=True)
    _view_module_version = Unicode('0.0.0').tag(sync=True)
    
    width = Int(800).tag(sync=True)
    height = Int(800).tag(sync=True)
    image = Unicode('').tag(sync=True)
    
    camera_info = Dict({'position': [], 'look_at': [], 'up': [], 'fov': 60}).tag(sync=True)

    scene_bounds = List([]).tag(sync=True)

    def __init__(self, kernelUtils, *args, **kwargs):
        widgets.DOMWidget.__init__(self, *args, **kwargs)

        self.is_connected = True

        self.on_msg(self._handle_msg)
                
        self.kernelUtils = kernelUtils
        self.kernelUtils.set_disconnect_callback(self.disconnect)
        
        try:
            self._update_scene_bounds()

            self._update_camera_info_from_ascent()
            
            self._update_image()
        except KeyError:
            clear_output(wait=True)
            self.close()
            self.kernelUtils.kernel.stderr("no images found, ensure jupyter_ascent has excecuted actions and re-execute the widget")

    #TODO notify the user and stop trying to handle clicks
    def disconnect(self):
        self.is_connected = False

    def _update_image(self):
        self.image = self.kernelUtils.get_images()[0]
    
    def _update_camera_info(self, camera_info):
        self.camera_info = camera_info
    
    def _update_camera_info_from_ascent(self):
        ascent_camera_info = self.kernelUtils.get_ascent_info()['images'][0]['camera']
        self._update_camera_info(ascent_camera_info)
        
    def _update_scene_bounds(self):
        self.scene_bounds = self.kernelUtils.get_ascent_info()['images'][0]['scene_bounds']
    
    def _handle_msg(self, msg, *args, **kwargs):
        if self.is_connected:
            content = msg["content"]["data"]["content"]
            if content['event'] == 'keydown' or content['event'] == 'button':
                code = content['code']
                if code == 87 or code == 'move_forward': #W
                    self.kernelUtils.forward()
                elif code == 65 or code == 'move_left': #A
                    self.kernelUtils.left()
                elif code == 83 or code == 'move_back': #S
                    self.kernelUtils.back()
                elif code == 68 or code == 'move_right': #D
                    self.kernelUtils.right()
                elif code == 'move_up':
                    self.kernelUtils.up()
                elif code == 'move_down':
                    self.kernelUtils.down()
                elif code == 'move_right':
                    self.kernelUtils.right()
                elif code == 'move_left':
                    self.kernelUtils.left()
                elif code == 'roll_c':
                    self.kernelUtils.roll_c()
                elif code == 'roll_cc':
                    self.kernelUtils.roll_cc()
                elif code == 'pitch_up':
                    self.kernelUtils.pitch_up()
                elif code == 'pitch_down':
                    self.kernelUtils.pitch_down()
                elif code == 'yaw_right':
                    self.kernelUtils.yaw_right()
                elif code == 'yaw_left':
                    self.kernelUtils.yaw_left()
                elif code == 'next':
                    #TODO this can fail
                    resp = self.kernelUtils.next_frame()
                    self.is_connected = (resp is not None)

                self._update_camera_info_from_ascent()
                
            elif content['event'] == 'mouseup':
                camera_info = content['camera_info']
                self._update_camera_info(camera_info)
                self.kernelUtils.look_at(camera_info['position'],
                               camera_info['look_at'],
                               camera_info['up'])

            self._update_image()
        else:
            clear_output(wait=True)
            self.close()
            self.kernelUtils.kernel.stderr("disconnected - wait to reconnect or check the simulation hasn't ended\n")
