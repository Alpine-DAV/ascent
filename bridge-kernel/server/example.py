# Copyright 2019 Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from mpi4py import MPI
import time

import os
import socket

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

try:
    from base64 import encodebytes
except ImportError:
    from base64 import encodestring as encodebytes

from mpi_server import MPIServer

import ascent.mpi
import json
import utils
import numpy

# these will be available in global ns
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a = ascent.mpi.Ascent()
ascent_opts = conduit.Node()
ascent_opts["mpi_comm"].set(comm.py2f())
ascent_opts["actions_file"] = ""
a.open(ascent_opts)

# demonstriate streaming output
def bar(n=10):
    for i in range(n):
        print(i)
        time.sleep(1)

def ranksum():
    return comm.allreduce(rank, MPI.SUM)

# demo image
def show_png_url(url="https://www.llnl.gov/sites/all/themes/llnl_bootstrap/logo.png"):
    image(urlopen(url).read(), "png")

# demo mpi_print
def hello_from_all():
    mpi_print("hello from %d" % rank)


def display_images(info):
    for img in info["images"].children():
        with open(img.node()["image_name"], 'rb') as f:
            img_content = f.read()
            image(img_content, "png")

def get_encoded_images(info):
    images = []
    for img in info["images"].children():
        with open(img.node()["image_name"], 'rb') as f:
            images.append(encodebytes(f.read()).decode('utf-8'))
    return images

def run_transformation(transformation, info, *args, **kwargs):
    actions = transformation(info, *args, **kwargs)
    #TODO this is temporary
    info['actions'][0]['scenes/s1/renders/r1/image_name'] = "out_ascent_render_3d"
    a.execute(actions)

#TODO pass this in rather than relying on it being in globals()
#TODO store MPIServer in global variable instead of passing it
#TODO why doesn't if rank==0 at the top work?
def handle_message(server, message):
    with server._redirect():
        info = conduit.Node()
        a.info(info)
        data = message["code"]

        if data["type"] == "transform":
            call_obj = data["code"]

            #TODO better way to do this so i don't have to import everything into this file
            #eval args
            args = []
            for arg in call_obj["args"]:
                args.append(eval(arg))

            #TODO eval kwargs
            kwargs = call_obj["kwargs"]

            run_transformation(utils.utils_dict[call_obj["name"]], info, *args, *kwargs)

            server.root_send({"type": "transform"})
        elif data["type"] == "get_images":
            server.root_send({
                "type": "images",
                "code": get_encoded_images(info),
            })
        elif data["type"] == "get_ascent_info":
            server.root_send({
                    "type": "ascent_info",
                    "code": info.to_json(),
                })
        elif data["type"] == "next":
            mpi_print("NEXT CODE")
            a.publish(ascent_data())
            #TODO use actions from before
            yaml = """

            - 
              action: "add_scenes"
              scenes: 
                s1: 
                  plots: 
                    p1: 
                      type: "volume"
                      field: "energy"
                  renders:
                    r1:
                      image_width: "966"
                      image_height: "700"
            - 
              action: "execute"
            - 
              action: "reset"

            """
            generated = conduit.Generator(yaml, "yaml")
            actions = conduit.Node()
            generated.walk(actions)
            a.execute(actions)

            server.root_send({"type": "next"})

my_prefix = "%s_%s" % (os.path.splitext(os.path.basename(sys.argv[0]))[0], socket.gethostname())
MPIServer(comm=comm, ns=globals(), prefix=my_prefix).serve()
