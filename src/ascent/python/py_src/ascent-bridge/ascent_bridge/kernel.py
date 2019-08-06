# Copyright 2019 Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function, unicode_literals, absolute_import

from IPython.display import display as ipydisplay, Image
from ipykernel.ipkernel import IPythonKernel
from ipywidgets import widgets
import importlib
import base64

from .client import SocketClient, get_backend_list

import copy
import time

from . import mywidgets

class BridgeKernel(IPythonKernel):
    banner = "Ascent Bridge"

    def __init__(self, klass=SocketClient, **kwargs):
        assert(issubclass(klass, SocketClient))
        IPythonKernel.__init__(self, **kwargs)
        self.klass = klass
        self.client = None
        # self.execution_count doesn't update
        self.exe_count = 0
        self.magics = {
            "%connect": self.connect_magic,
            "%disconnect": lambda args: self.client.disconnect(),
            "%exec_local": self.exec_local,
            "%trackball": lambda args: mywidgets.build_trackball(self),
        }
        self.last_used_backend = None
        self.disconnect_callback = None

    def out(self, name, message, silent=False):
        if silent: return
        self.send_response(self.iopub_socket, "stream", dict(name=name, text=message))

    def stdout(self, *args, **kwargs):
        self.out("stdout", *args, **kwargs)

    def stderr(self, *args, **kwargs):
        self.out("stderr", *args, **kwargs)

    def display(self, msg):
        cond = ("module" in msg and
                "attr" in msg and
                "args" in msg)

        if cond:
            try:
                # ModuleNotFoundError
                mod = importlib.import_module(msg["module"])
                args = msg["args"]
                if "decode_bytes" in msg:
                    for key in msg["decode_bytes"]:
                        args[key] = base64.decodebytes(args[key].encode("ascii"))

                # AttributeError
                cb = getattr(mod, msg["attr"])
                if isinstance(args, list):
                    obj = cb(*args)
                elif isinstance(args, dict):
                    obj = cb(**args)
                else:
                    self.stderr("display warning: 'args' is not list or dict")
                    return
                ipydisplay(obj)
            except Exception as e:
                self.stderr("display error: %s" % e)
        else:
            self.stderr("display error: message must contain 'module', 'attr', and 'args'")

    def exec_local(self, code):
        code = code.split('\n', 1)[1]
        try:
            exec(code, {"kernel": self})
        except Exception as e:
            print("Kernel exec error: %s." % e)

    #TODO this is copied from client.py
    def set_disconnect_callback(self, f):
        self.disconnect_callback = f

    #called by client when it disconnects
    def disconnect(self):
        if self.client is not None:
            self.client = None
        if self.disconnect_callback is not None:
            self.disconnect_callback()

    def connect(self, cfg):
        if self.client is not None:
            self.client.disconnect()

        self.config_data = cfg

        self.client = self.klass.from_config(cfg, stdout=self.stdout,
                stderr=self.stderr, display=self.display)

        is_connected = self.client.is_connected

        if is_connected:
            self.client.set_disconnect_callback(self.disconnect)
            print("to disconnect: %disconnect")
        else:
            self.client.disconnect()
            self.client = None
            self.stderr("unable to connect\n")
            return 1

        return 0

    def connect_magic(self, args):
        #TODO could this be better?
        if self.client is not None:
            self.client.check_connection()
            if self.client is not None:
                print("already connected to backend")
                return

        w = self.pick_backend()

        if w is None:
            self.stdout("no backends available")
        else:
            ipydisplay(w)

    def connect_wait(self, cfg, backend_wait_file=15):
        if self.client is not None:
            self.client.disconnect()

        self.stdout("connecting to %s" % cfg["codename"])
        key = "%s %s %s" % (cfg["codename"], cfg["date"], cfg["hostname"])

        while not self.backend_exists(cfg) and backend_wait_file > 0:
            self.stdout("waiting for backend socket - %d\n" % backend_wait_file)
            self.stdout("%s\n" % key)
            time.sleep(1)
            backend_wait_file -= 1
            backends = get_backend_list()

        if not self.backend_exists(cfg):
            self.stderr("Backend not found. The simulation may have ended.\n")
        else:
            self.connect(cfg)

    def backend_exists(self, cfg):
        try:
            backends = get_backend_list()
        except FileNotFoundError:
            print("config file not found, will wait")
            return False

        return any(backend["codename"] == cfg["codename"] and
                   backend["hostname"] == cfg["hostname"]
                   for key, backend in backends.items())

    def pick_backend(self):
        backends = get_backend_list()
        fields_to_display = ["codename", "date", "protocol", "argv"]

        if backends is None or len(backends) == 0:
            return None

        names = sorted(list(backends.keys()))
        select = widgets.Dropdown(options=names, value=names[len(names) - 1])
        button = widgets.Button(description="Connect")

        max_len = 0
        for f in fields_to_display:
            if len(f) > max_len:
                max_len = len(f)

        # setup labels to display fields from `fields_to_display`
        label_hboxes = []
        dynamic_labels = {}
        for f in fields_to_display:
            static_label = widgets.Label(value="{} :".format(f))
            dynamic_label = widgets.Label(value="")
            dynamic_labels[f] = dynamic_label
            label_hboxes.append(widgets.HBox([static_label, dynamic_label]))

        # callback when changing the dropdown
        def update_dynamic_labels(elem):
            obj = backends[select.value]
            for f in fields_to_display:
                dynamic_labels[f].value = obj[f]

        update_dynamic_labels(None)
        select.observe(update_dynamic_labels)

        def cb(elem):
            obj = backends[select.value]
            self.last_used_backend = copy.deepcopy(obj)
            self.connect(obj)
        button.on_click(cb)

        # order the widgets
        widgs = [select]
        widgs.extend(label_hboxes)
        widgs.append(button)

        # when this function is called the widget is displayed
        return widgets.VBox(widgs)

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        arg0 = code.strip().split()[0]
        eval_result = None
        if arg0 in self.magics.keys():
            self.magics[arg0](code)
        elif self.client is not None:
            eval_result = self.client.execute(code)
            self.exe_count += 1
        else:
            self.stdout("no backend - use %connect\n", silent)

        return dict(status="ok", execution_count=self.exe_count, user_expression=eval_result)

    def do_complete(self, code, cursor_pos):
        cursor_start = 0
        cursor_end = 0

        # builtins
        first_word = code.split(" ")[0]
        matches = [m for m in self.magics.keys() if first_word != "" and m.find(first_word) == 0]
        if len(matches) > 0:
            cursor_end = cursor_pos
        elif self.client is not None:
            data = self.client.complete(code, cursor_pos)
            if data is not None:
                matches = data["matches"]
                cursor_start = data["cursor_start"]
                cursor_end = data["cursor_end"]

        return dict(matches=matches, cursor_start=cursor_start, cursor_end=cursor_end,
                metadata=None, status="ok")

    def do_shutdown(self, restart):
        if self.client is not None:
            self.client.disconnect()
        return {"restart": False}

if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=BridgeKernel)
