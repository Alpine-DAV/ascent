.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _build_env:

Setting Up A Development Environment
====================================
The type of development environment needed depends on the use case.
In most cases, all that is needed is a build of Ascent. The exception
is Viskores filter development, which requires separate builds of Viskores
and VTK-h.

The list of common development use cases:
  * C++ and python filter development using Conduit Mesh Blueprint data
  * Connecting a new library to Ascent
  * Viskores filter development


I Want To Develop C++ and Python Code Directly In Ascent
--------------------------------------------------------
C++ and python filter can be directly developed inside of an Ascent build.
All that is required is a development build of Ascent. Please see :ref:`building`
for an overview of the different ways to build Ascent.

build_ascent
""""""""""""""
We recommend using :ref:`build_ascent.sh <build_ascent>` to setup a development environment with Ascent's
third-party dependencies. This script will create an `ascent-config.cmake` file
that can serve as a CMake initial cache file (or host-config).

.. code:: bash

    git clone --recursive https://github.com/alpine-dav/ascent.git
    cd ascent
    env prefix=tpls build_ascent=false ./scripts/build_ascent/build_ascent.sh
    cmake -C tpls/ascent-config.cmake -S src -B build

I Want To Develop Viskores and VTK-h Pipelines
-----------------------------------------------
If you want to add new features to VTK-h, its source is developed inside
the Ascent repo in the `src/libs/vtkh` directory.

If your changes also require new features in Viskores, you will need to build
and install your own version of Viskores. 

Once built and installed, update the CMake configure file with the locations
of the installs in the CMake variables ``VISKORES_DIR``.

.. note::

    Not all of Ascent dependencies are built with default options, branches, and commits, and
    that knowledge is built into the uberenv build. When building dependencies
    manually, consult :ref:`building` for specific build options for each
    dependency.

Here is the current version of Viskores  we are using:

.. literalinclude:: ../../../../hashes.txt
    :linenos:
    :language: python

Building the Ascent Source Code
-------------------------------
The CMake configure file should contain all the necessary locations to build
Ascent. Here are some example commands to create and configure a build from
the top-level directory. If the specific paths are different, adjust them
accordingly.

.. code:: bash

    mkdir build
    cd build
    cmake -C ../uberenv_libs/boden.llnl.gov-macos_1013_x86_64-clang@9.0.0-apple-ascent.cmake ../src
    make -j8
