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

    uberenv_libs/zeliak-darwin-ventura-m1-apple-clang@=14.0.0-ascent-qnsb6ehgctlevtgmxhdgywrf3opgju7j-patch.cmake

.. code:: cmake

    ##################################
    # spack generated host-config
    ##################################
    # darwin-ventura-m1-apple-clang@=14.0.0
    ##################################

    # cmake from spack 
    # cmake executable path: /Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/cmake-3.26.3-6clohsvnn4k3szexhttgm47f4grlygyr/bin/cmake

    #######
    # using apple-clang@=14.0.0 compiler spec
    #######

    # c compiler used by spack
    set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

    # cpp compiler used by spack
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

    # fortran compiler used by spack
    set(ENABLE_FORTRAN "ON" CACHE BOOL "")

    set(CMAKE_Fortran_COMPILER "/usr/local/bin/gfortran" CACHE PATH "")

    set(BLT_EXE_LINKER_FLAGS " -Wl,-rpath,/usr/local/gfortran/lib/" CACHE PATH "")

    set(BUILD_SHARED_LIBS "ON" CACHE BOOL "")

    set(ENABLE_TESTS "ON" CACHE BOOL "")

    # conduit from spack 
    set(CONDUIT_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/conduit-0.8.8-uo35y47k55nl77lfde2g6f4dmwoguiqj" CACHE PATH "")

    # Python Support
    # Enable python module builds
    set(ENABLE_PYTHON "ON" CACHE BOOL "")

    # python from spack 
     # NOTE: Pathed by uberenv to use spack view path instead of spack build path
    set(PYTHON_EXECUTABLE "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack_view/bin/python3" CACHE PATH "")

    # python module install dir
    set(ENABLE_DOCS "ON" CACHE BOOL "")

    # sphinx from spack 
     # NOTE: Pathed by uberenv to use spack view path instead of spack build path
    set(SPHINX_EXECUTABLE "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack_view/bin/sphinx-build" CACHE PATH "")

    set(ENABLE_SERIAL "ON" CACHE BOOL "")

    # MPI Support
    set(ENABLE_MPI "ON" CACHE BOOL "")

    set(MPI_C_COMPILER "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/mpich-4.1.1-2pbdhcszk35hgvwhxpvnjvcgj2oanyyr/bin/mpicc" CACHE PATH "")

    set(MPI_CXX_COMPILER "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/mpich-4.1.1-2pbdhcszk35hgvwhxpvnjvcgj2oanyyr/bin/mpic++" CACHE PATH "")

    set(MPI_Fortran_COMPILER "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/mpich-4.1.1-2pbdhcszk35hgvwhxpvnjvcgj2oanyyr/bin/mpif90" CACHE PATH "")

    set(MPIEXEC_EXECUTABLE "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/mpich-4.1.1-2pbdhcszk35hgvwhxpvnjvcgj2oanyyr/bin/mpiexec" CACHE PATH "")

    set(ENABLE_FIND_MPI "ON" CACHE BOOL "")

    # CUDA Support
    set(ENABLE_CUDA "OFF" CACHE BOOL "")

    set(ENABLE_OPENMP "OFF" CACHE BOOL "")

    # ROCm Support
    set(ENABLE_HIP "OFF" CACHE BOOL "")

    # vtk-h support 
    # vtk-h
    set(ENABLE_VTKH "ON" CACHE BOOL "")

    # vtk-m from spack
    set(VTKM_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/vtk-m-2.1.0-l5yh5lkvapxdil5fvhhj4l4udm2x6dg7" CACHE PATH "")

    set(VTKm_ENABLE_CUDA "OFF" CACHE BOOL "")

    # vtk-m not using ROCm
    # mfem from spack 
    set(MFEM_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/mfem-4.5.2-tspz4yld6oqq55tymvxufwoylsjjg36r" CACHE PATH "")

    set(ZLIB_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/zlib-1.2.13-4qnh2d4b2t32ve5r442d2rpfkrcxlgzs" CACHE PATH "")

    # devil ray
    set(ENABLE_DRAY "ON" CACHE BOOL "")

    set(ENABLE_APCOMP "ON" CACHE BOOL "")

    # occa not built by spack 
    # RAJA from spack 
    set(RAJA_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/raja-2022.10.4-cklp6el764fcceuwjvjtjybem2zhmomw" CACHE PATH "")

    # umpire from spack 
    set(UMPIRE_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/umpire-2022.03.1-4bfvyx5kaoe5q3vscpm7wyg2vm6bdrwc" CACHE PATH "")

    # camp from spack 
    set(CAMP_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/camp-2022.10.1-4bwinpswbgafdxotd4r5rnxtbsh6a44k" CACHE PATH "")

    # adios2 support
    # adios2 not built by spack 
    # Fides support
    # fides not built by spack 
    # GenTen support
    # genten not built by spack 
    # caliper from spack 
    set(CALIPER_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/caliper-2.9.0-orleesu5q52au3jrxwszygt25ggtux27" CACHE PATH "")

    set(ADIAK_DIR "/Users/harrison37/Work/alpine/ascent/uberenv_libs/spack/opt/spack/darwin-ventura-m1/apple-clang-14.0.0/adiak-0.2.2-bkvtpmg5r6jts3vrel6tptok3xs6dyzd" CACHE PATH "")

    ##################################
    # end spack generated host-config
    ##################################




I Want To Develop VTK-m and VTK-h Pipelines
---------------------------------------------
If you want to add new features to VTK-h, its source is developed inside
the Ascent repo in the `src/libs/vtkh` directory.

If your changes also require new features in VTK-m, you will need to build
and install your own version of VTK-m. 
Follow the examples in :ref:`building_manually` to create a VTK-m build.


Once built and installed, update the CMake configure file with the locations
of the installs in the CMake variables ``VTKM_DIR``.

.. note::

    Not all of Ascent dependencies are built with default options, branches, and commits, and
    that knowledge is built into the uberenv build. When building dependencies
    manually, consult :ref:`building` for specific build options for each
    dependency.

Here is the current version of VTK-m  we are using:

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
