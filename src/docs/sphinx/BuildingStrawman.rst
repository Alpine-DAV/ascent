.. ############################################################################
.. # Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Conduit.
.. #
.. # For details, see: http://software.llnl.gov/alpine/.
.. #
.. # Please also read alpine/LICENSE
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


Building Alpine
=================

Overview
--------


Alpine uses CMake for its build system.
Building Alpine creates two separate libraries:

    * libalpine : a version for execution on a single node
    * libalpine_par : a version for distributed memory parallel

The CMake variable( ENABLE_MPI ON | OFF ) controls the building the parallel version of Alpine and included proxy-apps.

The build dependencies vary according to which pipelines and proxy-applications are desired.
For a minimal build with no parallel components, the following are required:
    
    * Conduit
    * VTK-m
      
      * Boost
    
    * C++ and Fortran compilers


Build Dependencies
------------------

Alpine
^^^^^^^^

  * Conduit
  * IceT
  * One or more pipelines

Conduit
"""""""
  * MPI
  * Python + NumPy (Optional)
  * HDF5 (Optional)
  * Fortran compiler (Optional)

IceT
""""
  IceT is only needed for the parallel version of Alpine.
  
  * MPI

Pipelines
"""""""""

* EAVL: 
    
    * OSMesa (7.5.2) is used for rendering data set annotations and is required
    * OpenMP 3.5+ 
    * CUDA 6.5 + (Optional) 

* VTK-m: 
  
    * Boost
    * TBB (Optional)  Intel's Threaded Building Blocks
    * CUDA 6.5+ (Optional)

.. note:: When building Stawman with VTK-m 1.0 and CUDA, nvcc becomes confused and emits warnings about calling host functions from device functions. When combined with template meta-programming, this can emit quite a large amount of text during compilation. These warning messages have been suppressed in later versions of VTK-m.
  
* HDF5
  
    * Conduit with conduit_relay HDF5 support.
    

.. note:: 

    Alpine uses VTK-m 1.0 which must be configured with rendering on, among other options. 
    For a full list of options that need to be set, consult `/uberenv_libs/spack/var/spack/repos/builtin/packages/vtkm/package.py`.
    If you plan to use CUDA, a patch must be applied to VTK-m to prevent a compile error. 
    Using the build script will apply this patch automatically, but if compiling manually, the patch must be applied.
    The patch can be found in the source repo at `/uberenv_libs/spack/var/spack/repos/builtin/packages/vtkm/vtkm_patch.patch`


Getting Started
---------------
Clone the Straman repo:

* From Github

.. code:: bash
    
    git clone https://github.com/llnl/alpine.git


* From LLNL's CZ Stash Instance (LLNL Users)

.. code:: bash
    
    git clone https://{USER_NAME}@lc.llnl.gov/stash/scm/vis/alpine.git


Configure a build:

``config-build.sh`` is a simple wrapper for the cmake call to configure alpine. 
This creates a new out-of-source build directory ``build-debug`` and a directory for the install ``install-debug``.
It optionally includes a ``host-config.cmake`` file with detailed configuration options. 


.. code:: bash
    
    cd alpine
    ./config-build.sh


Build, test, and install Alpine:

.. code:: bash
    
    cd build-debug
    make -j 8
    make test
    make install



Build Options
-------------

Straman's build system supports the following CMake options:

* **BUILD_SHARED_LIBS** - Controls if shared (ON) or static (OFF) libraries are built. *(default = ON)* 
* **ENABLE_TESTS** - Controls if unit tests are built. *(default = ON)* 

* **ENABLE_DOCS** - Controls if the Alpine documentation is built (when sphinx and doxygen are found ). *(default = ON)*

* **ENABLE_FORTRAN** - Controls if Fortran components of Alpine are built. This includes the Fortran language bindings and Cloverleaf3D . *(default = ON)*
* **ENABLE_PYTHON** - Controls if the alpine python module and related tests are built. *(default = OFF)*

 The Alpine python module will build for both Python 2 and Python 3. To select a specific Python, set the CMake variable PYTHON_EXECUTABLE to path of the desired python binary. The alpine python module requires the Conduit python module.

* **ENABLE_OPENMP** - Controls if EAVL and proxy-apps are configured with OpenMP. *(default = OFF)*
* **ENABLE_CUDA** - Controls if VTK-m and EAVL are configured with GPU support. *(default = OFF)*
* **ENABLE_MPI** - Controls if parallel versions of proxy-apps and Alpine are built. *(default = ON)*


 We are using CMake's standard FindMPI logic. To select a specific MPI set the CMake variables **MPI_C_COMPILER** and **MPI_CXX_COMPILER**, or the other FindMPI options for MPI include paths and MPI libraries.

 To run the mpi unit tests on LLNL's LC platforms, you may also need change the CMake variables **MPIEXEC** and **MPIEXEC_NUMPROC_FLAG**, so you can use srun and select a partition. (for an example see: src/host-configs/chaos_5_x86_64.cmake)

* **CONDUIT_DIR** - Path to an Conduit install *(required for parallel version)*. 

* **ICET_DIR** - Path to an ICET install *(required for parallel version)*. 

* **EAVL_DIR** - Path to an EAVL install *(optional)*. 

* **VTKM_DIR** - Path to an VTK-m install *(optional)*. 

* **OSMESA_DIR** - Path to an VTK-m install *(required for EAVL)*. 

* **HDF5_DIR** - Path to a HDF5 install *(optional)*. 



Host Config Files
-----------------
To handle build options, third party library paths, etc we rely on CMake's initial-cache file mechanism. 


.. code:: bash
    
    cmake -C config_file.cmake


We call these initial-cache files *host-config* files, since we typically create a file for each platform or specific hosts if necessary. 

The ``config-build.sh`` script uses your machine's hostname, the SYS_TYPE environment variable, and your platform name (via *uname*) to look for an existing host config file in the ``host-configs`` directory at the root of the alpine repo. If found, it passes the host config file to CMake via the `-C` command line option.

.. code:: bash
    
    cmake {other options} -C host-configs/{config_file}.cmake ../


You can find example files in the ``host-configs`` directory. 

These files use standard CMake commands. CMake *set* commands need to specify the root cache path as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")

It is  possible to create your own configure file, and an boilerplate example is provided in `/host-configs/boilerplate.cmake`

.. warning:: If compiling all of the dependencies yourself, it is important that you use the same compilers for all dependencies. For
             example, different MPI and Fortran compilers (e.g., Intel and GCC) are not compatible with one another.

Bootstrapping Third Party Dependencies 
--------------------------------------

You can use ``bootstrap-env.sh`` (located at the root of the alpine repo) to help setup your development environment on OSX and Linux. 
This script uses ``scripts/uberenv/uberenv.py``, which leverages **Spack** (http://software.llnl.gov/spack) to build the external third party libraries and tools used by Alpine. 
Fortran support in is optional, dependencies should build without fortran. 
After building these libraries and tools, it writes an initial *host-config* file and adds the Spack built CMake binary to your PATH, so can immediately call the ``config-build.sh`` helper script to configure a alpine build.

.. code:: bash
    
    #build third party libs using spack
    source bootstrap-env.sh
    
    #copy the generated host-config file into the standard location
    cp uberenv_libs/`hostname`*.cmake to host-configs/
    
    # run the configure helper script
    ./config-build.sh

    # or you can run the configure helper script and give it the 
    # path to a host-config file 
    ./config-build.sh uberenv_libs/`hostname`*.cmake


.. .. note::
..     There is a known issue on some OSX systems when building with Fortran dependencies.
..     This is caused by the native compilers being 64-bit while the Fortran compiler is 32-bit.

Compiler Settings for Third Party Dependencies 
----------------------------------------------
You can edit ``scripts/uberenv/compilers.yaml`` to change the compiler settings
passed to Spack. See the `Spack Compiler Configuration <http://software.llnl.gov/spack/basic_usage.html#manual-compiler-configuration>`_   
documentation for details.

For OSX, the defaults in ``compilers.yaml`` are clang from X-Code and gfortran from https://gcc.gnu.org/wiki/GFortranBinaries#MacOS. 

.. note::
    The bootstrapping process ignores ``~/.spack/compilers.yaml`` to avoid conflicts
    and surprises from a user's specific Spack settings on HPC platforms.

Building with Spack
-------------------

.. note::
  Alpine developers use ``scripts/uberenv/uberenv.py`` to setup third party libraries for Alpine 
  development.  Due to this, the process builds more libraries than necessary for most use cases.
  For example, we build independent installs of Python 2 and Python 3 to make it easy 
  to check Python C-API compatibility during development. In the near future, we plan to 
  provide a Spack package to simplify deployment.



Using Alpine in Another Project
---------------------------------

Under ``src/examples`` there are examples demonstrating how to use Alpine in a CMake-based build system (``using-with-cmake``) and via a Makefile (``using-with-make``). Under ``src/examples/proxies``  you can find example integrations using alpine in the Lulesh, Kripke, and Cloverleaf3D proxy-applications.

Building Alpine in a Docker Container
---------------------------------------

Under ``src/examples/docker/master/ubuntu`` there is an example ``Dockerfile`` which can be used to create an ubuntu-based docker image with a build of the Alpine github master branch. There is also a script that demonstrates how to build a Docker image from the Dockerfile (``example_build.sh``) and a script that runs this image in a Docker container (``example_run.sh``). The Alpine repo is cloned into the image's file system at ``/alpine``, the build directory is ``/alpine/build-debug``, and the install directory is ``/alpine/install-debug``.

