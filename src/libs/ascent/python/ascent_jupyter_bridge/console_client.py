#!/usr/bin/env python

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# Copyright (c) Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function, unicode_literals

import enum
import os
import readline

from bridge_kernel.client import SocketClient, get_backend_list, load_config


# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass


info_str = """
console_client commands start with '%' (hash); any command starting with a hash is parsed as a
console_client command, other commands are sent as raw execute requests to the backend, with the
exception of (white-space stripped) 'quit' and 'quit()', which ask that you quit using the hash
command. Nothing stops you from quitting this instance with expression other than 'quit' and
'quit()' that stops the backend so use a bit of caution.
"""


class Status(enum.Enum):
    quit = 0
    normal = 1


class ConsoleClientCompletor():
    def __init__(self, tcr):
        self.tcr = tcr

    def complete(self, code, state):
        response = None
        if state == 0:
            self.matches = self.tcr.complete(code)
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        return response


class ConsoleClient():
    def __init__(self, config, klass=SocketClient, readline_delims="{}[]()=+-/*^~;,# \t\n", prompt='> '):
        self.klass = klass
        self.client = None
        self.is_debug = False
        self.prompt = prompt
        self.config = config
        self.completor = ConsoleClientCompletor(self)
        readline.parse_and_bind('tab: complete')
        readline.set_completer_delims(readline_delims)
        readline.set_completer(self.completor.complete)
        self._setup_builtins()
        self._connect_client()

    def __del__(self):
        self._disconnect_client()

    def _setup_builtins(self):
        self._builtins = {
            '%help': {
                'callback': self._help,
            },

            '%quit': {
                'callback': self._quit,
                'info': "disconnect from the remote instance and quit console_client"
            },

            '%quit_inst': {
                'callback': self._shutdown_backend,
                'info': "quit the backend"
            },

            '%disconnect': {
                'callback': self._disconnect_client,
                'info': "disconnect from the current backend and select a new one"
            },

            '%debug': {
                'callback': self._toggle_client_debug,
                'info': "toggles the client's debug prints",
            },

            '%flush': {
                'callback': self._flush_client,
                'info': "flush the clients queued messages"
            },

            '%complete': {
                'callback': lambda args: print(self.complete(args)),
                'info': "print the raw message from a complete request"
            },

            '%inspect': {
                'callback': self._inspect,
                'info': "print the raw message from an inspect request"
            }
        }

    def _process_builtins(self, code):
        split = code.split(' ')
        cmd = split[0]
        if len(split) > 1:
            args = ' '.join(split[1:])
        else:
            args = ''
        if cmd == '%help':
            self._help()
        elif cmd in self._builtins.keys():
            return self._builtins[cmd]['callback'](args) or Status.normal
        else:
            print("error: no builtin %{}".format(cmd))

    def _add_builtin(self, name, callback, info):
        if name[0] != '%':
            name = '%' + name
        self._builtins[name] = {
            'callback': callback,
            'info': info
        }

    def _help(self, *args):
        print(info_str)
        print("builtins:")
        for k, v in self._builtins.items():
            if k == '%help':
                continue
            if 'info' in v:
                info = v['info']
            else:
                info = ""
            print("  {:<15}  {}".format(k, info))

    def _connect_client(self):
        self._disconnect_client()

        self.client = self.klass.from_config(self.config)
        if not self.client.is_connected:
            print("couldn't connect")
            return

        self.client.set_debug(self.is_debug)

    def _disconnect_client(self, *args):
        if self.client is not None and self.client.is_connected:
            self.client.disconnect()

    def _quit(self, *args):
        if self.client is not None and self.client.is_connected:
            self.client.disconnect()
        return Status.quit

    def _shutdown_backend(self, *args):
        if self.client is not None and self.client.is_connected:
            if (input('this will quit the code, are you sure? (Y/y)> ').lower() == 'y'):
                self.client.execute("quit()")
        else:
            print("no client connected")

    def _toggle_client_debug(self, *args):
        if self.client is None:
            return

        if self.is_debug:
            print("turning off debug prints")
            self.is_debug = False
        else:
            print("turning on debug prints")
            self.is_debug = True
        self.client.set_debug(self.is_debug)

    def _flush_client(self, *args):
        if self.client is not None:
            self.client.flush()

    def _inspect(self, text):
        if self.client is not None:
            print(self.client.inspect(text))

    def complete(self, code):
        if len(code) > 0 and code[0] == '%':
            return [x for x in self._builtins.keys() if x[0:len(code)] == code]

        if self.client is not None and self.client.is_connected:
            data = self.client.complete(code, len(code))
            if data is not None:
                matches = data['matches']
                cursor_start = data['cursor_start']
                if cursor_start > 0:
                    matches = ["{}{}".format(code[0:cursor_start], m) for m in matches]
                return matches
            return []

    def repl(self):
        print("%help for console_client help")
        stat = Status.normal
        while self.client is not None and self.client.is_connected:
            try:
                code = input(self.prompt)
                if code.strip() == "quit" or code.strip() == "quit()":
                    print("please use %quit or %quit_inst")
                elif len(code) > 0 and code.strip()[0] == "%":
                    stat = self._process_builtins(code.strip())
                else:
                    self.client.execute(code)
            except KeyboardInterrupt:
                print()
        return stat


def select_backend():
    try:
        while True:
            backends = get_backend_list()
            if backends is None or len(backends) == 0:
                print("no backends available")
                return None

            keys = sorted(list(backends.keys()))
            for i, k in enumerate(keys):
                print("%d) %s" % (i, k))

            req = input("> ")
            if req == "%quit":
                return None
            try:
                index = int(req)
                if index >= 0 and index < len(keys):
                    return backends[keys[index]]
            except ValueError:
                pass
    except KeyboardInterrupt:
        print()
        return None


cls = ConsoleClient


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentTypeError

    def existing_file(path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise ArgumentTypeError("file doesn't exist: %s" % path)
        return path

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=existing_file, help="path to config file")
    args = parser.parse_args()

    if args.path is not None:
        config = load_config(args.path)
        if config is not None:
            cls(config).repl()
    else:
        while True:
            config = select_backend()
            if config is None:
                break
            if cls(config).repl() == Status.quit:
                break
