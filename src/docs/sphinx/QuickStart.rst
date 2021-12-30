.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _quick_start:

================================
Quick Start
================================


Installing Ascent and Third Party Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quickest path to install Ascent and its dependencies is via :ref:`uberenv <building_with_uberenv>`:

.. code:: bash

    git clone --recursive https://github.com/alpine-dav/ascent.git
    cd ascent
    python scripts/uberenv/uberenv.py --install --prefix="build"


After this completes, ``build/ascent-install`` will contain an Ascent install.

We also provide spack settings for several well known HPC clusters, here is an example of how to use our settings for NERSC's Cori System:

.. code:: bash

    python scripts/uberenv/uberenv.py --install --prefix="build" --spack-config-dir="scripts/uberenv_configs/spack_configs/nersc/cori/"


For more details about building and installing Ascent see :ref:`building`. This page provides detailed info about Ascent's CMake options, :ref:`uberenv <building_with_uberenv>` and :ref:`Spack <building_with_spack>` support. We also provide info about :ref:`building for known HPC clusters using uberenv <building_known_hpc>` and a :ref:`Docker example <building_with_docker>` that leverages Spack.


Public Installs of Ascent
~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide public installs of Ascent for the default compilers at a few  DOE HPC centers.

Summary table of public ascent installs:

.. list-table::
   :widths: 10 15 10 10 50
   :header-rows: 1

   * - Site
     - System
     - Compiler
     - Runtime
     - Install Path

   * - LLNL LC
     - CZ TOSS 3 (Pascal)
     - gcc 4.9.3
     - OpenMP
     - ``/usr/gapps/conduit/software/ascent/current/toss_3_x86_64_ib/openmp/gnu/ascent-install``

   * - NERSC
     - Cori
     - gcc 8.2.0
     - OpenMP
     - ``/project/projectdirs/alpine/software/ascent/current/cori/gnu/ascent-install/``

   * - OLCF
     - Summit
     - gcc 6.4.0
     - OpenMP
     - ``/gpfs/alpine/world-shared/csc340/software/ascent/current/summit/openmp/gnu/ascent-install/``

   * - OLCF
     - Summit
     - gcc 6.4.0
     - CUDA
     - ``/gpfs/alpine/world-shared/csc340/software/ascent/current/summit/cuda/gnu/ascent-install/``


See :ref:`tutorial_setup_public_installs` for more details on using these installs.


Using Ascent in Your Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The install includes examples that demonstrate how to use Ascent in a CMake-based build system and via a Makefile.

CMake-based build system example (see: ``examples/ascent/using-with-cmake``):

.. literalinclude:: ../../examples/using-with-cmake/CMakeLists.txt
   :lines: 6-


Makefile-based build system example (see: ``examples/ascent/using-with-make``):

.. literalinclude:: ../../examples/using-with-make/Makefile
   :lines: 6-













