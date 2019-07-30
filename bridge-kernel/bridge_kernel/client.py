# Copyright 2019 Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function, unicode_literals

import base64
import enum
import errno
import json
import os
import select
import shutil
import socket
import ssl
import struct
import sys
import uuid

from .tunnel import open_ssh_tunnel

# TODO: handle better
if sys.version_info >= (3, 0):
    python3 = True
else:
    python3 = False

if struct.calcsize("!I") != 4:
    print("**warning: unsigned int is not 4 bytes**")


#
# The frame is always 4 bytes for the message length
#
#                      1       3
#  0 1 2 3 4 5 6 7 8 9 0     9 0 1
# +----------------------....-----+
# |            Length             |
# |                               |
# |          Up to 2^32-1         |
# +-------------------------------+
#

# The "!" is for network byte order.
# See https://docs.python.org/2/library/struct.html for a discussion on the python struct
# class for interpreting strings as packed binary data

frame_format = "!I"
frame_length = struct.calcsize(frame_format)
max_message_length = (2**(8 * frame_length)) - 1

# How much to read an write per recv/send call
CHUNK_SIZE = 1024

config_dir = os.path.join(os.environ["HOME"], ".config", "bridge_kernel")
localhost = socket.gethostname()


def make_config_dir():
    os.mkdir(config_dir)


def clean_config_dir():
    if os.path.isdir(config_dir):
        shutil.rmtree(config_dir)
    make_config_dir()


def get_backend_list():
    if not os.path.isdir(config_dir):
        return None

    backends = {}
    for f in os.listdir(config_dir):
        if os.path.splitext(f)[1] == ".json":
            filepath = os.path.join(config_dir, f)
            with open(filepath, "r") as _:
                data = json.load(_)
            data["name"] = f
            data["filepath"] = filepath
            data["hostname"] = data["hosts"][-1]
            data["codename"] = os.path.basename(data["argv"].split(" ")[0])
            backends["%s %s %s" % (data["codename"], data["date"], data["hostname"])] = data
    return backends


def validate_config(config):
    if not isinstance(config, dict):
        print("error: expected dict")
        return False

    if "encryption" not in config:
        config["encryption"] = "none"

    base_cond = ("user" in config and
            "date" in config and
            "hosts" in config and
            "argv" in config and
            "protocol" in config and
            "encryption" in config)

    ipc_cond = ("uds" in config)

    openssl_cond = ("port" in config,
            "ca-pem64" in config,
            "creds-pem64" in config)

    if base_cond:
        if config["protocol"] == "ipc":
            return ipc_cond
        elif config["protocol"] == "tcp":
            return openssl_cond
        else:
            print("error: unknown protocol %s" % config["protocol"])
    else:
        print("error: invalid config")
    return False


def load_config(path):
    if os.path.isfile(path):
        with open(path, "r") as _:
            config = json.load(_)
        if not validate_config(config):
            return None
        return config
    else:
        print("error: {} does not exist".format(path))
        return None


class MessageType(enum.Enum):
    execute = "execute"
    complete = "complete"
    inspect = "inspect"
    stdout = "stdout"
    stderr = "stderr"
    display = "display"
    idle = "idle"
    disconnect = "disconnect"
    interrupt = "interrupt"


class SocketClient():
    def __init__(self, config, stdout, stderr, display):
        self._sock = None
        self.is_connected = False
        self.debug = lambda *x: x
        self.config = config
        self.stdout = stdout
        self.stderr = stderr
        self.display = display
        self._stdout = stdout
        self._stderr = stderr
        self._display = display
        self.disconnect_callback = None

    @classmethod
    def from_config(cls, config, stdout=sys.stdout.write, stderr=sys.stderr.write, display=None):
        if not validate_config(config):
            raise TypeError("invalid config dict")
        inst = cls(config, stdout=stdout, stderr=stderr, display=display)
        inst.connect(config)
        return inst

    @classmethod
    def from_config_path(cls, path, stdout=sys.stdout.write, stderr=sys.stderr.write, display=None):
        config = load_config(path)
        if config is None:
            stderr("invalid path {}\n".format(path))
            return None
        return cls.from_config(config, stdout, stderr, display)

    def __del__(self):
        self.disconnect()

    def _connect(self, connect_args):
        self._sock.settimeout(5)
        try:
            self._sock.connect(connect_args)
            self._sock.settimeout(None)
        except socket.error as serr:
            self.stderr("cannot connect: {}\n".format(serr))
            return

        self.stdout("connected to %s\n" % self.config["server"])

        self.is_connected = True
        # wait for an "idle" message before returning
        self.read()

    def connect(self, config):
        if self.is_connected:
            self.stderr("already connected")
            return

        # TODO: it might make more sense to make config fields
        # members in self
        server = self.config["hosts"][-1]
        config["server"] = server
        open_ssh_tunnel(lambda s: print(s), config, server, ssh_port=22, ipc_wait_file=5)

        if config["protocol"] == "ipc":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            args = (config["uds"])
        elif config["protocol"] == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            args = ("127.0.0.1", config["port"])
        else:
            raise RuntimeError("unknown protocol %s" % config["protocol"])

        if config["encryption"] == "tls":
            client_creds_temp_file = os.path.join(config_dir, str(uuid.uuid4()))
            ca_cert_temp_file = os.path.join(config_dir, str(uuid.uuid4()))

            with open(client_creds_temp_file, "w") as f:
                data = base64.b64decode(config["creds-pem64"])
                f.write(data.decode("utf-8") if python3 else data)
            with open(ca_cert_temp_file, "w") as f:
                data = base64.b64decode(config["ca-pem64"])
                f.write(data.decode("utf-8") if python3 else data)

            # switch to PROTOCOL_TLS when supported
            sslctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            sslctx.options != ssl.OP_NO_SSLv2
            sslctx.options != ssl.OP_NO_SSLv3

            sslctx.verify_mode = ssl.CERT_REQUIRED
            sslctx.check_hostname = True
            sslctx.load_cert_chain(client_creds_temp_file)
            sslctx.load_verify_locations(cafile=ca_cert_temp_file)
            self._sock = sslctx.wrap_socket(sock, server_hostname="bridge_kernel.server")
        else:
            if config["protocol"] == "tcp":
                self.stderr("warning: unencrypted tcp is insecure")
            self._sock = sock

        self._connect(args)

        if config["encryption"] == "tls":
            os.remove(client_creds_temp_file)
            os.remove(ca_cert_temp_file)

    def disconnect(self, error=False):
        if not error and self.is_connected:
            self.writemsg({"type": MessageType.disconnect.value})
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self.config = None
        self.is_connected = False

    def set_disconnect_callback(self, f):
        self.disconnect_callback = f

    # Convert to `bytes` type
    def _tobytes(self, s):
        if not isinstance(s, bytes):
            return s.encode("utf-8")
        return s

    # Convert to `str` type
    def _tostr(self, s):
        if not isinstance(s, str):
            return s.decode("ascii")
        return s

    def _has_message(self, timeout=0):
        r, _, _ = select.select([self._sock], [], [], timeout)
        return len(r) > 0

    #
    # _read - non-throwing socket revc for known "errors"
    #          _read wlll not throw on select errors, socket errors, or
    #          OSError (TODO: why OSError?)
    #          and automatically disconnect the client if there is an exception
    #          or the message is "" (server disconnected)
    #
    def _read(self, num_bytes):
        try:
            self.debug("entering recv")
            message = self._sock.recv(num_bytes)
            self.debug("passed recv")

            # if we get an empty message (and didn"t want it) the server
            # has disconnected
            if len(message) == 0 and num_bytes > 0:
                self.disconnect(error=True)
                return None
            else:
                return message

        except select.error as selecterr:
            (code, msg) = selecterr.args
            if code != errno.EINTR:
                self.debug("select error: {}, {}".format(code, msg))
                self.disconnect(error=True)
                return None
            else:
                self.debug("select error: interrupt")
                return None

        except socket.error as sockerr:
            (code, msg) = sockerr.args
            if code != errno.EINTR:
                self.debug("socket error: {}, {}".format(code, msg))
                self.disconnect(error=True)
                return None
            else:
                self.debug("socket error: interrupt")
                return None

        except OSError:
            self.debug("os error")
            self.disconnect(error=True)
            return None

    # returns a dictonary or None
    def readmsg(self):
        if not self.is_connected:
            self.stderr("read: not connected!")
            return None

        msg = self._read(frame_length)
        if msg is None:
            return None

        length = struct.unpack(frame_format, msg)[0]

        chunks = []
        bytes_read = 0
        if length > 0:
            while bytes_read < length:
                to_read = min(CHUNK_SIZE, length - bytes_read)
                chunk = self._read(to_read)
                if chunk is None:
                    return
                chunks.append(chunk)
                bytes_read += len(chunk)
            message = b"".join(chunks)
        else:
            message = None

        try:
            obj = json.loads(message)
        except ValueError:
            print("invalid message format")
            print(length)
            print(message[length-10:])
            return None

        return obj

    # obj is a dict
    def writemsg(self, obj):
        if not self.is_connected:
            self.stderr("write: not connected!")
            return

        if not isinstance(obj, dict):
            self.stderr("write: expected type {}".format(dict))
            return

        code = self._tobytes(json.dumps(obj))

        length = len(code)
        if length > max_message_length:
            self.stderr("error: message too long: {} > {}".format(length, max_message_length))
            return

        message = struct.pack(frame_format, length)
        message += code
        message_length = len(message)

        bytes_sent = 0
        while bytes_sent < message_length:
            try:
                sent = self._sock.send(message)
            except socket.error as sockerr:
                (code, msg) = sockerr.args
                self.debug("socket error: {}, {}".format(code, msg))
                self.disconnect(error=True)
                return

            if sent == 0:
                self.stderr("error: socket connection broken")
                self.disconnect(error=True)
                return
            bytes_sent += sent

    def flush(self):
        if not self.is_connected:
            self.stdout("flush: not connected!")
            return None
        while self._has_message():
            message = self.readmsg()
            if message is not None:
                if (message["type"] == MessageType.stdout.value or message["type"] == MessageType.stderr.value):
                    self.stdout(message["code"])

    def read(self, timeout=0):
        if not self.is_connected:
            return None

        obj = None

        while True:
            message = self.readmsg()

            if message is None:
                return None
            elif message["type"] == MessageType.stdout.value:
                self.stdout(self._tostr(message["code"]))
            elif message["type"] == MessageType.stderr.value:
                self.stderr(self._tostr(message["code"]))
            elif message["type"] == MessageType.display.value:
                if self.display is not None:
                    self.display(message)
                else:
                    self.stdout("[display] - no graphics display available")
            elif message["type"] == MessageType.idle.value:
                return obj

            # general messages are captured and returned to the caller
            else:
                obj = message

    def _execute(self, code):
        if not self.is_connected:
            self.stdout("_execute: not connected!")
            return
        self.writemsg({
            "type": MessageType.execute.value,
            "code": code,
            "code_length": len(code)})
        return self.read()

    def execute(self, code):
        try:
            ret = self._execute(code)
            if not self.is_connected:
                self.stdout("disconnected\n")
                if self.disconnect_callback is not None:
                    self.disconnect_callback()
            return ret
        except KeyboardInterrupt:
            # TODO: this is an interesting case where select (even 2x select)
            # reports that there is data to be recieved but recv hangs.
            # If you look at server input there is indeed stdout/err that is
            # lost somewhere. Added a second Kboard catch so that we don"t
            # have to kill the client if this block happends...
            try:
                self.interrupt()
                self.read()
            except KeyboardInterrupt:
                print("warning: you may miss some output")

    def complete(self, code, cursor_pos):
        if not self.is_connected:
            self.stdout("complete: not connected!")
            return None

        self.writemsg({
            "type": MessageType.complete.value,
            "cursor_pos": cursor_pos,
            "code": code,
            "code_length": len(code)})

        message = self.read()

        if message is None:
            return None

        if message["type"] == MessageType.complete.value:
            return message

        return None

    def interrupt(self):
        if not self.is_connected:
            self.stdout("interrupt: not connected!\n")
            return

        self.writemsg({"type": MessageType.interrupt.value})
        self.stdout("interrupted\n")

    def inspect(self, path):
        if not self.is_connected:
            self.stdout("inspect: not connected!\n")
            return

        self.writemsg({"type": MessageType.inspect.value, "path": path})

        message = self.read()

        if message is None:
            return None

        if message["type"] == MessageType.inspect.value:
            return message

        return None

    def set_debug(self, status):
        if status:
            self.debug = self.stdout
        else:
            self.debug = lambda *x: x

    def set_silent(self, status=True):
        if status:
            self.stdout = lambda x: None
        else:
            self.stdout = self._stdout
