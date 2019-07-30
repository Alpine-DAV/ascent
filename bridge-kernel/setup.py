# Copyright 2019 Lawrence Livermore National Security, LLC and other
# Bridge Kernel Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from glob import glob
from os.path import join, abspath, dirname, exists
from setuptools import setup
import shutil

from ipykernel.kernelspec import write_kernel_spec, make_ipkernel_cmd

install_requires = ["enum34", "jupyter_core", "ipywidgets", "ipykernel", "IPython", "numpy", "matplotlib"]

distname = "bridge_kernel"
kernelname = "bridge"

setup_args = dict(
    name=distname,
    description="Bridge Kernel for existing backends",
    packages=[distname],
    install_requires=install_requires
)

# Kernelspec installations
dest = join(abspath(dirname(__file__)), "kernelspec_data")
if exists(dest):
    shutil.rmtree(dest)

write_kernel_spec(path=dest, overrides=dict(
    argv=make_ipkernel_cmd("%s.kernel" % distname),
    display_name="Bridge Kernel",
    name=kernelname
))

setup_args["data_files"] = [(join("share", "jupyter", "kernels", kernelname), glob(join(dest, "*")))]

setup(**setup_args)
