# Copyright (c) Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from glob import glob
import os
from os.path import join, abspath, dirname, exists, relpath, split
from setuptools import setup
import shutil

from ipykernel.kernelspec import write_kernel_spec, make_ipkernel_cmd

install_requires = ["enum34", "jupyter_core", "ipywidgets", "ipykernel", "IPython", "numpy", "matplotlib"]

distname = "ascent_jupyter_bridge"
kernelname = "ascent_jupyter_bridge"
widgetsname = "ascent_widgets"

setup_args = dict(
    name=distname,
    description="Ascent Bridge for existing backends",
    packages=[distname, widgetsname, '{}.trackball'.format(widgetsname)],
    install_requires=install_requires,
    zip_safe=False
)

# Kernelspec installations
dest = join(abspath(dirname(__file__)), "kernelspec_data")
if exists(dest):
    shutil.rmtree(dest)

write_kernel_spec(path=dest, overrides=dict(
    argv=make_ipkernel_cmd("%s.kernel" % distname),
    display_name="Ascent Bridge",
    name=kernelname
))

setup_args["data_files"] = [
    (join("share", "jupyter", "kernels", kernelname), glob(join(dest, "*"))),
    ('etc/jupyter/nbconfig/notebook.d', [join(widgetsname, '{}.json'.format(widgetsname))])
]

js_dir = join(widgetsname, 'js', 'lib')
# install HTML/JS/CSS files
for path, directories, files in os.walk(join(js_dir)):
    # directory relative to js_dir
    rel_dir = relpath(path, join(js_dir))
    # destination directory in jupyter
    dest = join("share", "jupyter", "nbextensions", widgetsname, rel_dir)
    # file paths relative to setup.py
    rel_files = [join(path, f) for f in files]
    setup_args["data_files"].append((dest, rel_files))

#setup_args["package_data"] = {distname: ["ascent_widgets/**/*"]}

setup(**setup_args)
