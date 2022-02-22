.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _quick_start:

================================
Quick Start
================================

Running Ascent via Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to try out Ascent is via our Docker container. If you have Docker installed you can obtain a Docker image with a ready-to-use ascent install from `Docker Hub <https://hub.docker.com/r/alpinedav/ascent/>`_. This image also includes a Jupyter install to support running Ascent's tutorial notebooks.


To start the Jupyter server and run the tutorial notebooks, run:

.. code::

    docker run -p 8888:8888 -t -i alpinedav/ascent-jupyter

(The ``-p`` is used to forward ports between the container and your host machine, we use these ports to allow web servers on the container to serve data to the host.)

This container automatically launches a Jupyter Notebook server on port 8888. Assuming you forwarded port 8888 from the Docker container to your host machine, you should be able to connect to the notebook server using http://localhost:8888. The password for the notebook server is: ``learn``



Installing Ascent and Third Party Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quickest path to install Ascent and its dependencies is via :ref:`uberenv <building_with_uberenv>`:

.. code:: bash

    git clone --recursive https://github.com/alpine-dav/ascent.git
    cd ascent
    python scripts/uberenv/uberenv.py --install --prefix="build"


After this completes, ``build/ascent-install`` will contain an Ascent install.

We also provide spack settings for several well known HPC clusters, here is an example of how to use our settings for OLCF's Summit System:

.. code:: bash

    python scripts/uberenv/uberenv.py --install --prefix="build" --spack-config-dir="scripts/uberenv_configs/spack_configs/configs/olcf/summit_gcc_9.3.0_cuda_11.0.3/"


For more details about building and installing Ascent see :ref:`building`. This page provides detailed info about Ascent's CMake options, :ref:`uberenv <building_with_uberenv>` and :ref:`Spack <building_with_spack>` support. We also provide info about :ref:`building for known HPC clusters using uberenv <building_known_hpc>` and a :ref:`Docker example <building_with_docker>` that leverages Spack.


Public Installs of Ascent
~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide public installs of Ascent for the default compilers at a few  DOE HPC centers.

Summary table of public ascent installs:

.. list-table::
   :widths: 10 15 10 10 20 50
   :header-rows: 1

   * - Site
     - System
     - Compiler
     - Runtime
     - Modules
     - Install Path

   * - OLCF
     - Summit
     - gcc 9.3.0
     - CUDA 11.0.3
     -  ``gcc/9.3.0`` ``cuda/11.0.3``
     - ``/sw/summit/ums/ums010/ascent/current/summit/cuda/gnu/ascent-install/``

   * - OLCF
     - Summit
     - gcc 9.3.0
     - OpenMP
     - ``gcc/9.3.0``
     - ``/sw/summit/ums/ums010/ascent/current/summit/openmp/gnu/ascent-install/``

   * - NERSC
     - Permutter
     - gcc 9.3.0
     - CUDA 11.4.0
     - ``PrgEnv-gnu`` ``cpe-cuda/21.12``  ``cudatoolkit/21.9_11.4``
     - ``/global/cfs/cdirs/alpine/software/ascent/current/perlmutter/cuda/gnu/ascent-install/``

   * - LLNL LC
     - CZ TOSS 3 (Pascal)
     - gcc 4.9.3
     - OpenMP
     - (none)
     - ``/usr/gapps/conduit/software/ascent/current/toss_3_x86_64_ib/openmp/gnu/ascent-install``


`Here is a link to the scripts we use to build public Ascent installs. <https://github.com/Alpine-DAV/ascent/tree/develop/scripts/spack_install>`_

See :ref:`tutorial_setup_public_installs` for more details on using these installs.


Using Ascent in Your Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The install includes examples that demonstrate how to use Ascent in a CMake-based build system and via a Makefile.

CMake-based build system example (see: ``examples/ascent/using-with-cmake``):

.. literalinclude:: ../../examples/using-with-cmake/CMakeLists.txt
   :lines: 6-


CMake-based build system example with MPI (see: ``examples/ascent/using-with-cmake-mpi``):

.. literalinclude:: ../../examples/using-with-cmake-mpi/CMakeLists.txt
   :lines: 6-


Makefile-based build system example (see: ``examples/ascent/using-with-make``):

.. literalinclude:: ../../examples/using-with-make/Makefile
   :lines: 6-

Makefile-based build system example with MPI (see: ``examples/ascent/using-with-make-mpi``):

.. literalinclude:: ../../examples/using-with-make-mpi/Makefile
   :lines: 6-














