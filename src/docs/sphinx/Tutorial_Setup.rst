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

SC19 Tutorial Option
~~~~~~~~~~~~~~~~~~~~~~~

For SC19


Build and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build and install Ascent yourself see :doc:`QuickStart`.

NERSC Cori Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a public ascent install for use on NERSC's Cori System. This install was built with the default
intel compiler (18.0.1.163).

The install is located at ``/project/projectdirs/alpine/software/ascent/current/cori/ascent-install``.
You can copy the tutorial examples from this install and build them as follows:

.. code::

    cp -r /project/projectdirs/alpine/software/ascent/current/cori/ascent-install/examples/ascent/tutorial/ascent_intro .
    cd ecp_2019
    make ASCENT_DIR=project/projectdirs/alpine/software/ascent/current/cori/ascent-install/

Using Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Docker installed you can obtain a Docker image with a ready-to-use ascent install from `Docker Hub <https://hub.docker.com/r/alpinedav/ascent/>`_.

Fetch the latest Ascent image:

.. code::

    docker pull alpinedav/ascent

After the download completes, create and run a container using this image:

.. code::

    docker run -p 8000:8000 -p 10000:10000 -t -i alpinedav/ascent

(The ``-p`` is used to forward ports between the container and your host machine, we use these ports to allow web servers on the container to serve data to the host.)


You will now be at a bash prompt in you container.

To add the proper paths to Python and MPI to your environment run:

.. code::

    source ascent_docker_setup.sh

The ascent source code is at ``/ascent/src/``, and the install is at ``/ascent/install-debug``.


These demos assume you have an Ascent install. If you have access to Docker, the easiest way to test the waters is via the ``alpinedav/ascent`` Docker image. For more details about using this image see :ref:`demos_using_docker`. You can also build Ascent with `Spack <https://spack.io/>`_. For more details see :ref:`building_with_spack`.

.. .. _demos_using_docker:
..
.. Running Demos using Docker
.. -----------------------------------
..
.. If you have Docker installed you can obtain a Docker image with a ready-to-use ascent install from `Docker Hub <https://hub.docker.com/r/alpinedav/ascent/>`_.
..
.. Fetch the latest Ascent image:
..
.. .. code::
..
..     docker pull alpinedav/ascent
..
.. After the download completes, create and run a container using this image:
..
.. .. code::
..
..     docker run -p 8000:8000 -p 10000:10000 -t -i alpinedav/ascent
..
.. (The ``-p`` is used to forward ports between the container and your host machine, we use these ports to allow web servers on the container to serve data to the host.)
..
..
.. You will now be at a bash prompt in you container.
..
.. To add the proper paths to Python and MPI to your environment run:
..
.. .. code::
..
..     source ascent_docker_setup.sh
..
.. The ascent source code is at ``/ascent/src/``, and the install is at ``/ascent/install-debug``.
..
.. Next, try running an included python example:
..
.. .. code::
..
..     cd /ascent/install-debug/examples/ascent/python
..     python ascent_python_render_example.py
..
.. You should see some verbose output and ``out_ascent_render_3d.png`` will be created.
..
..
.. To view output files you can use a simple Python web server to expose files from the container to your host machine:
..
.. .. code::
..
..     python -m SimpleHTTPServer 10000
..
..
.. With this server running, open up a web browser on your host machine and view localhost:10000. You should be able to click on ``out_ascent_render_3d.png`` and view the rendered result in your web browser.
..
.. You should now be ready to run the other demos, remember to use the Python web server can help you browse results in the Docker container.


Running Jupyter Notebook Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


