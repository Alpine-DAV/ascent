.. ############################################################################
.. # Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Ascent.
.. #
.. # For details, see: http://ascent.readthedocs.io/.
.. #
.. # Please also read ascent/LICENSE
.. #
.. # Redistribution and use in source and binary forms, with or without
.. # modification, are permitted provided that the following conditions are met:
.. #
.. # * Redistributions of source code must retain the above copyright notice,
.. #   this list of conditions and the disclaimer below.
.. #
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. #
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. #
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
.. # POSSIBILITY OF SUCH DAMAGE.
.. #
.. ############################################################################


Tutorial Setup
=================

The tutorial examples are installed with Ascent to the subdirectory ``examples/ascent/tutorial/``.  Below are several options for using pre-built Ascent installs and links to info about building Ascent. If you have access to Docker, the easiest way to test the waters is via the ``alpinedav/ascent`` Docker image.

SC19 Tutorial Option
~~~~~~~~~~~~~~~~~~~~~~~

For SC19, we plan to have several instances of our Ascent Docker iamge up and running the jupyter notebook server.
We will provide IP addresses and login info to attendees.

Using Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Docker installed you can obtain a Docker image with a ready-to-use ascent install from `Docker Hub <https://hub.docker.com/r/alpinedav/ascent/>`_. This image also includes a jupyter install to support running Ascent's tutorial notebooks.

Fetch the latest Ascent image:

.. code::

    docker pull alpinedav/ascent

After the download completes, create and run a container using this image:

.. code::

    docker run -p 8000:8000 run -p 8888:8888 -p 10000:10000 -t -i alpinedav/ascent

(The ``-p`` is used to forward ports between the container and your host machine, we use these ports to allow web servers on the container to serve data to the host.)


You will now be at a bash prompt in you container.

To add the proper paths to Python and MPI to your environment run:

.. code::

    source ascent_docker_setup_env.sh

The ascent source code is at ``/home/user/ascent/src/``, and the install is at ``/home/user/ascent/install-debug/``.
The tutorial examples are at ``/home/user/ascent/install-debug/examples/ascent/tutorial/`` and the tutorial notebooks are at ``/home/user/ascent/install-debug/examples/ascent/tutorial/ascent_intro/notebooks/``.


To launch the a jupyter notebook server run:

.. code::

    ./ascent_docker_run_jupyter.sh

This will launch a notebook server on port 8888. Assuming you forwarded port 8888 from the Docker container to your host machine, you should be able to connect to the notebook server using http://localhost:8888. The current password for the notebook server is: ``ascentsc19``


NERSC Cori Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a public ascent install for use on NERSC's Cori System. This install was built with the default
gnu compiler (8.2.0). You need to use `module load gcc` to build and run the installed examples.


The install is located at ``/project/projectdirs/alpine/software/ascent/current/cori/gnu/ascent-install``.
You can copy the tutorial examples from this install and use them as follows:

.. code::

    #
    # source helper script that loads the default gcc module, sets python paths, and ASCENT_DIR env var
    #
    source /project/projectdirs/alpine/software/ascent/current/cori/ascent_cori_setup_env_gcc.sh
    
    #
    # make your own dir to hold the tutorial examples
    #
    mkdir ascent_tutorial
    cd ascent_tutorial
    
    #
    # copy the examples from the public install
    #
    cp -r /project/projectdirs/alpine/software/ascent/current/cori/gnu/ascent-install/examples/ascent/tutorial/* .
    
    #
    # build cpp examples and run the first one
    #
    cd ascent_intro/cpp
    make
    ./ascent_first_light_example
    
    #
    # run a python example
    #
    cd ..
    cd python
    python ascent_first_light_example.py  

Build and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build and install Ascent yourself see :doc:`QuickStart`.


