import ipywidgets as widgets
from traitlets import Unicode, validate, Int, List, Dict
from IPython.display import clear_output
import json
import copy
import pkg_resources
import os

class KernelUtils():
    def __init__(self, kernel):
        self.kernel = kernel
        self.disconnect_callback = None
        self.kernel.set_disconnect_callback(self.disconnect)

    #TODO this is copied from client.py
    def set_disconnect_callback(self, f):
        self.disconnect_callback = f

    def disconnect(self):
        if self.disconnect_callback is not None:
            self.disconnect_callback()

    def custom_send(self, code):
        self.kernel.client.writemsg({
            "type": "custom",
            "code": code,
        })
        return self.kernel.client.read()

    #TODO kwargs
    def send_transform(self, call_info):
        call_obj = {
            "name": call_info[0],
            "args": call_info[1:],
            "kwargs": [],
        }
        return self.custom_send({
            "type": "transform",
            "code": call_obj,
        })

    def get_images(self):
        return self.custom_send({
            "type": "get_images",
        })["code"]

    #TODO may be faster to send parts of info rather than whole thing
    def get_ascent_info(self):
        return json.loads(
            self.custom_send({
                "type": "get_ascent_info",
            })["code"]
        )

    def next_frame(self):
        self.kernel.connect_wait(copy.deepcopy(self.kernel.last_used_backend))
        return self.custom_send({
            "type": "next",
        })

    def look_at(self, position, look_at, up):
        self.send_transform(['look_at', str(position), str(look_at), str(up)])

    def up(self):
        self.send_transform(['move_up', '0.1'])

    def down(self):
        self.send_transform(['move_up', '-0.1'])

    def forward(self):
        self.send_transform(['move_forward', '0.1'])

    def back(self):
        self.send_transform(['move_forward', '-0.1'])

    def right(self):
        self.send_transform(['move_right', '0.1'])

    def left(self):
        self.send_transform(['move_right', '-0.1'])

    def roll_c(self):
        self.send_transform(['roll', '-2*numpy.pi/36'])

    def roll_cc(self):
        self.send_transform(['roll', '2*numpy.pi/36'])

    def pitch_up(self):
        self.send_transform(['pitch', '2*numpy.pi/36'])

    def pitch_down(self):
        self.send_transform(['pitch', '-2*numpy.pi/36'])

    def yaw_right(self):
        self.send_transform(['yaw', '-2*numpy.pi/36'])

    def yaw_left(self):
        self.send_transform(['yaw', '2*numpy.pi/36'])

class TrackballWidget(widgets.DOMWidget):
    _view_name = Unicode('TrackballView').tag(sync=True)
    _view_module = Unicode('ascent_widgets').tag(sync=True)
    _view_module_version = Unicode('0.0.0').tag(sync=True)
    
    width = Int(966).tag(sync=True)
    height = Int(700).tag(sync=True)
    image = Unicode('').tag(sync=True)
    
    camera_info = Dict({'position': [], 'look_at': [], 'up': [], 'fov': 60}).tag(sync=True)

    scene_bounds = List([]).tag(sync=True)

    def __init__(self, kernelUtils, *args, **kwargs):
        widgets.DOMWidget.__init__(self, *args, **kwargs)

        self.is_connected = True

        self.on_msg(self._handle_msg)
                
        #TODO make my own DomWidget subclass that all my widgets inherit from to avoid repeating this line?
        self.kernelUtils = kernelUtils
        self.kernelUtils.set_disconnect_callback(self.disconnect)
        
        try:
            self._update_scene_bounds()

            self._update_camera_info_from_ascent()
            
            self._update_image()
        except KeyError:
            clear_output(wait=True)
            self.close()
            self.kernelUtils.kernel.stderr("no images found, ensure server_ascent has excecuted actions and re-execute the widget")

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
                if code == 87 or code == 'move_up': #W
                    self.kernelUtils.up()
                elif code == 65 or code == 'move_left': #A
                    self.kernelUtils.left()
                elif code == 83 or code == 'move_down': #S
                    self.kernelUtils.down()
                elif code == 68 or code == 'move_right': #D
                    self.kernelUtils.right()
                elif code == 'move_forward':
                    self.kernelUtils.forward()
                elif code == 'move_back':
                    self.kernelUtils.back()
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
                
                self._update_image() #TODO move this to bottom of function?
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


#TODO move this to a different file and make more general?
def build_trackball(kernel):
    #VIEWS_PATH = pkg_resources.resource_filename(__name__, 'views/')
    #display(Javascript(filename=os.path.join(VIEWS_PATH, 'trackball', 'trackball.js')))
    kernelUtils = KernelUtils(kernel)
    s = TrackballWidget(kernelUtils)
    display(s)
