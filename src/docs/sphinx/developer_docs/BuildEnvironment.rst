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
is VTK-m filter development, which requires separate builds of VTK-m
and VTK-h.

The list of common development use cases:
  * C++ and python filter development using Conduit Mesh Blueprint data
  * Connecting a new library to Ascent
  * VTK-m filter development


I Want To Develop C++ and Python Code Directly In Ascent
--------------------------------------------------------
C++ and python filter can be directly developed inside of an Ascent build.
All that is required is a development build of Ascent. Please see :ref:`building`
for an overview of the different ways to build Ascent.

Manual Build
""""""""""""
An guide to building Ascent's dependencies manually can be found at
:ref:`building_manually`.
When building manually, we recommended you create a CMake configure file like the
example shown below rather than specifying all the CMake options on the command line.


Spack-Based Build
"""""""""""""""""
The spack-based build system is controlled through the
`uberenv python <https://github.com/Alpine-DAV/ascent/blob/develop/scripts/uberenv/uberenv.py>`_
script which is located in `scripts/uberenv` directory. The most basic way to build
Ascent is using this script.

.. code:: bash

    git clone --recursive https://github.com/alpine-dav/ascent.git
    cd ascent
    python scripts/uberenv/uberenv.py


.. note::

    This command is slightly different than what is in the :ref:`quick_start` guide.
    Specifically, the uberenv command does not contain the `--install --prefix:'build'`
    options.

If the uberenv script succeeds, then a CMake configure file is created in the `uberenv_libs`
directory that can be used to create a build of Ascent. The file name is a combination of
the system name, system type, and compiler used. For example, on an OSX system the following
file was created that contains the CMake variables need for an Ascent build:

.. code:: bash

    ls uberenv_libs/
    boden.llnl.gov-macos_1013_x86_64-clang@9.0.0-apple-ascent.cmake

.. code:: cmake

    ##################################
    # spack generated host-config
    ##################################
    # macos_1013_x86_64-clang@9.0.0-apple
    ##################################

    # cmake from spack
    # cmake executable path: /Users/larsen30/research/ascent_docs/ascent/uberenv_libs/spack/opt/spack/darwin-highsierra-x86_64/clang-9.0.0-apple/cmake-3.9.6-lkrrqgruseaa7kcmtehvmanupghfuwcb/bin/cmake

    #######
    # using clang@9.0.0-apple compiler spec
    #######

    # c compiler used by spack
    set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

    # cpp compiler used by spack
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

    # fortran compiler used by spack
    set(ENABLE_FORTRAN "ON" CACHE BOOL "")

    set(CMAKE_Fortran_COMPILER "/opt/local/bin/gfortran" CACHE PATH "")

    set(BUILD_SHARED_LIBS "ON" CACHE BOOL "")

    set(ENABLE_TESTS "ON" CACHE BOOL "")

    # conduit from spack
    set(CONDUIT_DIR "/Users/larsen30/ascent_docs/ascent/uberenv_libs/spack/opt/spack/darwin-highsierra-x86_64/clang-9.0.0-apple/conduit-master-qr6ffvlcnihvcuhjsb3a5kbj5cee4ueo" CACHE PATH "")

    # Python Support
    # Enable python module builds
    set(ENABLE_PYTHON "ON" CACHE BOOL "")

    # python from spack
    set(PYTHON_EXECUTABLE "/Users/larsen30/ascent_docs/ascent/uberenv_libs/spack/opt/spack/darwin-highsierra-x86_64/clang-9.0.0-apple/python-2.7.15-im7d5b5gswfkvjfqzec4cbpxwaf6g3kw/bin/python2.7" CACHE PATH "")

    set(ENABLE_DOCS "ON" CACHE BOOL "")

    # sphinx from spack
    set(SPHINX_EXECUTABLE "/Users/larsen30/ascent_docs/ascent/uberenv_libs/spack/opt/spack/darwin-highsierra-x86_64/clang-9.0.0-apple/py-sphinx-1.8.2-chmw6atszhupa6shnswvsdwypzoynznj/bin/sphinx-build" CACHE PATH "")

    # MPI Support
    set(ENABLE_MPI "OFF" CACHE BOOL "")

    # CUDA Support
    set(ENABLE_CUDA "OFF" CACHE BOOL "")

    set(ENABLE_OPENMP "OFF" CACHE BOOL "")

    # vtk-h support
    # vtk-m from spack
    set(VTKM_DIR "/Users/larsen30/ascent_docs/ascent/uberenv_libs/spack/opt/spack/darwin-highsierra-x86_64/clang-9.0.0-apple/vtkm-ascent_ver-w7fnfd3otmpyywvlhsdsdq6x7cau5xgx" CACHE PATH "")

    # vtk-h from spack
    set(VTKH_DIR "/Users/larsen30/ascent_docs/ascent/uberenv_libs/spack/opt/spack/darwin-highsierra-x86_64/clang-9.0.0-apple/vtkh-ascent_ver-o2qjba4ojyudlswyoqns2qb63xyqagnb" CACHE PATH "")

    # mfem not built by spack
    # adios support
    # adios not built by spack
    ##################################
    # end spack generated host-config
    ##################################



I Want To Develop VTK-m and VTK-h Code
--------------------------------------
If you want to add new features to VTK-m and VTK-h, and expose those features in 
Ascent. In addition to the steps in the previous section, you will need to build
and install VTK-m and VTK-h. The following information builds on the
previous section, altering a spack-based build to instead use manually built versions
of VTK-m and VTK-h. If all the dependencies were built manually, then this section
can be safely skipped.

First follow the instructions in :ref:`building_vtkm` and
:ref:`building_vtkh`.

Once built and installed, update the CMake configure file with the locations
of the installs in the CMake variables ``VTKM_DIR`` and ``VTKH_DIR``, respectively.

.. note::

    Not all of Ascent dependencies are built with default options, branches, and commits, and
    that knowledge is built into the uberenv build. When building dependencies
    manually, consult :ref:`building` for specific build options for each
    dependency.

Here are the current versions of vtkm and vtkh we build and test against:

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
