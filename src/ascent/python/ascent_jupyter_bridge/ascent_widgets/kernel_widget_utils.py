import json
import copy

# this class is an extension for the kernel that acts as an interface between the kernel and widgets
class KernelWidgetUtils():
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
            "kwargs": {},
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
