.. ############################################################################
.. # Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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

Tutorials
==============================

ECP 2019 Annual Meeting
-------------------------

**In Situ Visualization and Analysis with Ascent**

**Date:** Thursday, January 17th, 2019

**Time:** 8:30am - 12:00pm

**Location:** TBD

**Authors:**

Hank Childs (University of Oregon); Matthew Larsen (Lawrence Livermore National Laboratory); Cyrus Harrison (Lawrence Livermore National Laboratory); Kenneth Moreland (Sandia National Laboratories); David Rogers (Los Alamos National Laboratory) 

**Abstract:**

In situ visualization and analysis is an important capability for addressing slow I/O on modern supercomputers. With this 3-hour tutorial, we will spend the majority of our time (two hours) going into detail on Ascent, an in situ visualization and analysis library being developed by ECP ALPINE. Ascent is from the makers of ParaView Catalyst and VisIt LibSim, and it will soon be able to directly connect with both of those products. The tutorial will be practical in nature: how to integrate Ascent into a simulation code, Ascent’s data model, build and linking issues, and capabilities. The remaining hour will be spent highlighting other visualization efforts in ECP, such as in situ-specific visualization algorithms, VTK-m, and CINEMA. 

**Slides:** Coming soon ... 

Tutorial Setup
----------------------------------------

Build and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build and install Ascent yourself see :doc:`QuickStart`.

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

NERSC Cori Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example Programs
----------------------------------------

You can find the tutorial example source code and a Makefile in your Ascent install 
in the ``examples/ascent/tutorial/ecp_2019`` directory.


