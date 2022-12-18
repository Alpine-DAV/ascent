.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _building:

Building Ascent
===============

This page provides details on several ways to build Ascent from source.

For the shortest path from zero to Ascent, see :doc:`QuickStart`.

To build third party dependencies we recommend using :ref:`uberenv <building_with_uberenv>` which leverages Spack or :ref:`Spack directly<building_with_spack>`.
We also provide info about :ref:`building for known HPC clusters using uberenv <building_known_hpc>`.
and a :ref:`Docker example <building_with_docker>` that leverages Spack.

Overview
--------

Ascent uses CMake for its build system.
Building Ascent creates two separate libraries:

    * libascent : a version for execution on a single node
    * libascent_mpi : a version for distributed-memory parallelism

The CMake variable( ENABLE_MPI ON | OFF ) controls the building the parallel version of Ascent and included proxy-apps.

The build dependencies vary according to which pipelines and proxy-applications are desired.
For a minimal build with no parallel components, the following are required:

    * Conduit
    * C++ compilers

We recognize that building on HPC systems can be difficult, and we have provide two separate build strategies.

    * A spack based build
    * Manually compile dependencies using a CMake configuration file to keep compilers and libraries consistent

Most often, the spack based build should be attempted first. Spack will automatically download and build all
the third party dependencies and create a CMake configuration file for Ascent. Should you encounter build issues
that are not addressed here, please ask questions using our github `issue tracker <https://github.com/Alpine-DAV/ascent/issues>`_.


Build Dependencies
------------------

Ascent requires Conduit and provides optional features that depend on third-party libraries:


.. list-table::
   :header-rows: 1
   
   * - Feature
     - Required TPLS

   * - VTK-h Rendering and Filtering Pipelines
     - VTk-m (Serial, OpenMP, CUDA, Kokkos)

   * - MFEM High-Order to Low-Order Refinement for VTK-h Pipelines
     - MFEM

   * - Devil Ray High-Order Ray Tracer Pipelines
     - RAJA (Serial, OpenMP, CUDA, HIP), Umpire, MFEM

   * - General Expressions
     - RAJA (Serial, OpenMP, CUDA, HIP), Umpire

   * - JIT Expressions
     - OCCA, Umpire


For a detailed account of features and what underpin them see :ref:`feature_map`.



Getting Started
---------------
Clone the Ascent repo:

* From Github

.. code:: bash

    git clone --recursive https://github.com/Alpine-DAV/ascent.git


``--recursive`` is necessary because we are using a git submodule to pull in BLT (https://github.com/llnl/blt).
If you cloned without ``--recursive``, you can checkout this submodule using:

.. code:: bash

    cd ascent
    git submodule init
    git submodule update



Configure a build:

``config-build.sh`` is a simple wrapper for the cmake call to configure ascent.
This creates a new out-of-source build directory ``build-debug`` and a directory for the install ``install-debug``.
It optionally includes a ``host-config.cmake`` file with detailed configuration options.


.. code:: bash

    cd ascent
    ./config-build.sh


Build, test, and install Ascent:

.. code:: bash

    cd build-debug
    make -j 8
    make test
    make install



Build Options
-------------

Ascent's build system supports the following CMake options:

Main CMake Options
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Default

   * - ``BUILD_SHARED_LIBS``
     - Controls if shared (ON) or static (OFF) libraries are built.
     - *(default = ON)*

   * - ``ENABLE_FORTRAN``
     - Controls if Fortran components of Ascent are built. This includes the Fortran language bindings and Cloverleaf3D.
     - *(default = ON)*

   * - ``ENABLE_PYTHON``
     - Controls if the Ascent Python module and related tests are built.
     - *(default = OFF)*

   * - ``ENABLE_MPI``
     - Controls if MPI parallel versions of Ascent and proxy-apps are built.
     - *(default = ON)*

   * - ``ENABLE_SERIAL``
     - Controls if Serial (non-MPI) version of Ascent and proxy-apps are built.
     - *(default = ON)*

   * - ``ENABLE_CUDA``
     - Controls if Ascent uses CUDA.
     - *(default = OFF)*

   * - ``ENABLE_OPENMP``
     - Controls if the proxy-apps and Ascent use with OpenMP. 
     - *(default = OFF)*

   * - ``ENABLE_DRAY``
     - Controls if Devil Ray is built. Requires RAJA + Umpire. (Devil Ray is now developed as part of Ascent)
     - *(default = OFF)*

   * - ``ENABLE_APCOMP``
     - Controls if AP Compositor is built. (AP Compositor is now developed as part of Ascent)
     - *(default = OFF)*

   * - ``ENABLE_VTKH``
     - Controls if VTK-h is built.. Requires VTK-m. (VTK-h is now developed as part of Ascent)
     - *(default = OFF)*

   * - ``ENABLE_EXAMPLES``
     - Controls if Ascent examples are built.
     - *(default = ON)*

   * - ``ENABLE_UTILS``
     - Controls if Ascent utilities are built.
     - *(default = ON)*

   * - ``ENABLE_TESTS``
     - Controls if Ascent tests are built.
     - *(default = ON)*

   * - ``ENABLE_LOGGING``
     - Controls if data logging is built.
     - *(default = ON)*

   * - ``ENABLE_DOCS``
     - Controls if the Ascent documentation is built (when sphinx is available).
     - *(default = ON)*

   * - (Devil Ray Specific Options)
     - 
     - 

   * - ``DRAY_ENABLE_STATS``
     - Controls if Devil Ray Status support is built.
     - *(default = ON)*

   * - ``DRAY_USE_DOUBLE_PRECISION``
     - Controls if Devil Ray is built with 64-bit floats
     - *(default = OFF, use 32-bit precision floats)*


CMake Options for Third-party Library Paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Name
     - Description

   * - ``CONDUIT_DIR``
     - Path to a Conduit install **(required)**

   * - ``CALIPER_DIR``
     - Path to a Caliper install (optional)

   * - ``ADIAK_DIR``
     - Path to an Adiak install (optional) Caliper support requires Adiak.

   * - ``RAJA_DIR``
     - Path to a RAJA install (optional)

   * - ``UMPIRE_DIR``
     - Path to a Umpire install (optional)

   * - ``OCCA_DIR``
     - Path to an OCCA install (optional)

   * - ``VTKM_DIR``
     - Path to a VTK-m install (optional)

   * - ``KOKKOS_DIR``
     - Path to a Kokkos install (optional)

   * - ``ADIOS2_DIR``
     - Path to a ADIOS 2 install (optional)

   * - ``FIDES_DIR``
     - Path to a FIDES install (optional)

   * - ``BABELFLOW_DIR``
     - Path to a BabelFlow install (optional)
     
   * - ``PMT_DIR``
     - Path to a ParallelMergeTree install (optional)
     
   * - ``StreamStat_DIR``
     - Path to a StreamStat install (optional)
     
   * - ``TopoFileParser_DIR``
     - Path to a TopoFileParser install (optional)
     
   * - ``BLT_SOURCE_DIR``
     - Path to a BLT install (default = ``blt``)


Additional Build Notes
^^^^^^^^^^^^^^^^^^^^^^

* **Python** - The Ascent Python module builds for both Python 2 and Python 3. To select a specific Python, set the CMake variable PYTHON_EXECUTABLE to path of the desired python binary. The Ascent Python module requires the Conduit Python module.

* **MPI** - We use CMake's standard FindMPI logic. To select a specific MPI set the CMake variables ``MPI_C_COMPILER`` and ``MPI_CXX_COMPILER``, or the other FindMPI options for MPI include paths and MPI libraries. To run the mpi unit tests, you may also need change the CMake variables ``MPIEXEC_EXECUTABLE`` and ``MPIEXEC_NUMPROC_FLAG``, so you can use a different launcher, such as srun and set number of MPI tasks used.

* **BLT** - Ascent uses BLT (https://github.com/llnl/blt) as the foundation of its CMake-based build system. It is included as a submodule in Ascent's git repo, and also available in our release tarballs. The ``BLT_SOURCE_DIR`` CMake option defaults to ``src/blt``, where we expect the blt submodule. The most compelling reason to override is to share a single instance of BLT across multiple projects.


Host Config Files
-----------------
To handle build options, third party library paths, etc we rely on CMake's initial-cache file mechanism.


.. code:: bash

    cmake -C config_file.cmake


We call these initial-cache files *host-config* files, since we typically create a file for each platform or specific hosts if necessary.

The ``config-build.sh`` script uses your machine's hostname, the SYS_TYPE environment variable, and your platform name (via *uname*) to look for an existing host config file in the ``host-configs`` directory at the root of the ascent repo. If found, it passes the host config file to CMake via the `-C` command line option.

.. code:: bash

    cmake {other options} -C host-configs/{config_file}.cmake ../


You can find example files in the ``host-configs`` directory.

These files use standard CMake commands. CMake *set* commands need to specify the root cache path as follows:

.. code:: cmake

    set(CMAKE_VARIABLE_NAME {VALUE} CACHE PATH "")

It is  possible to create your own configure file, and an boilerplate example is provided in `/host-configs/boilerplate.cmake`

.. warning:: If compiling all of the dependencies yourself, it is important that you use the same compilers for all dependencies. For example, different MPI and Fortran compilers (e.g., Intel and GCC) are not compatible with one another.


.. _building_with_uberenv:

Building Ascent and Third Party Dependencies
--------------------------------------------------
We use **Spack** (http://spack.io) to help build Ascent's third party dependencies on OSX and Linux.

Uberenv (``scripts/uberenv/uberenv.py``) automates fetching spack, building and installing third party dependencies, and can optionally install Ascent as well.  To automate the full install process, Uberenv uses the Ascent Spack package along with extra settings such as Spack compiler and external third party package details for common HPC platforms.


Uberenv Options for Building Third Party Dependencies
------------------------------------------------------

``uberenv.py`` has a few options that allow you to control how dependencies are built:

 ==================== ============================================== ================================================
  Option               Description                                     Default
 ==================== ============================================== ================================================
  --prefix             Destination directory                          ``uberenv_libs``
  --spec               Spack spec                                     linux: **%gcc**
                                                                      osx: **%clang**
  --spack-config-dir   Folder with Spack settings files               linux: (empty)
                                                                      osx: ``scripts/uberenv_configs/spack_configs/darwin/``
  -k                   Ignore SSL Errors                              **False**
  --install            Fully install ascent not just dependencies     **False**
 ==================== ============================================== ================================================

The ``-k`` option exists for sites where SSL certificate interception undermines fetching
from github and https hosted source tarballs. When enabled, ``uberenv.py`` clones spack using:

.. code:: bash

    git -c http.sslVerify=false clone https://github.com/llnl/spack.git

And passes ``-k`` to any spack commands that may fetch via https.


Default invocation on Linux:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %gcc

Default invocation on OSX:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %clang \
                                      --spack-config-dir scripts/uberenv_configs/spack_configs/darwin/


The uberenv `--install` installs ascent\@develop (not just the development dependencies):

.. code:: bash

    python scripts/uberenv/uberenv.py --install


For details on Spack's spec syntax, see the `Spack Specs & dependencies <http://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_ documentation.


Compiler Settings for Third Party Dependencies
----------------------------------------------

You can edit yaml files under ``scripts/uberenv_configs/spack_configs/configs/{platform}`` or use the **--spack-config-dir** option to specify a directory with compiler and packages yaml files to use with Spack. See the `Spack Compiler Configuration <http://spack.readthedocs.io/en/latest/getting_started.html#manual-compiler-configuration>`_
and `Spack System Packages
<http://spack.readthedocs.io/en/latest/getting_started.html#system-packages>`_
documentation for details.

For macOS, the defaults in ``scripts/uberenv_configs/spack_configs/configs/darwin/compilers.yaml`` are X-Code's clang and gfortran from https://gcc.gnu.org/wiki/GFortranBinaries#MacOS.

.. note::
    The bootstrapping process ignores ``~/.spack/compilers.yaml`` to avoid conflicts
    and surprises from a user's specific Spack settings on HPC platforms.

When run, ``uberenv.py`` checkouts a specific version of Spack from github as ``spack`` in the
destination directory. It then uses Spack to build and install Conduit's dependencies into
``spack/opt/spack/``. Finally, it generates a host-config file ``{hostname}.cmake`` in the
destination directory that specifies the compiler settings and paths to all of the dependencies.


.. _building_known_hpc:

Building with Uberenv on Known HPC Platforms
--------------------------------------------------

`Here is a link to the scripts we use to build public Ascent installs. <https://github.com/Alpine-DAV/ascent/tree/develop/scripts/spack_install>`_


Building Third Party Dependencies for Development
--------------------------------------------------

You can use ``scripts/uberenv/uberenv.py`` to help setup your development environment on OSX and Linux. ``uberenv.py`` leverages **Spack** (https://spack.io/) to build the external third party libraries and tools used by Ascent.
Fortran support in is optional, dependencies should build without fortran.
After building these libraries and tools, it writes an initial *host-config* file and adds the Spack built CMake binary to your PATH, so can immediately call the ``config-build.sh`` helper script to configure a ascent build.

.. code:: bash

    #build third party libs using spack
    python scripts/uberenv/uberenv.py

    #copy the generated host-config file into the standard location
    cp uberenv_libs/`hostname`*.cmake host-configs/

    # run the configure helper script
    ./config-build.sh

    # or you can run the configure helper script and give it the
    # path to a host-config file
    ./config-build.sh uberenv_libs/`hostname`*.cmake



.. _building_with_spack:

Building with Spack
-------------------

Currently, we maintain our own fork of Spack for stability. As part of the uberenv python
script, we automatically clone our
`Spack fork. <https://github.com/Alpine-DAV/spack/tree/ascent/develop>`_

.. warning::
  Installing Ascent from the Spack develop branch will most likely fail. We build and test spack
  installations with uberenv.py.

To install Ascent and also build all of its dependencies as necessary run:

.. code:: bash

  spack install ascent


The Ascent Spack package provides several
`variants <http://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_
that customize the options and dependencies used to build Ascent.

Please see the `Ascent Spack package source <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/ascent/package.py>`_ (or use ``spack info ascent``) to learn about variants.


Uberenv Spack Configurations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the `Spack configs <https://github.com/Alpine-DAV/spack_configs/tree/main/configs/alpinedav>`_ we use to build our CI Containers for concrete examples of using ``pacakges.yaml`` and ``compilers.yaml`` to specify system packages and compiler details to Spack.


Using Ascent in Another Project
---------------------------------

Under ``src/examples`` there are examples demonstrating how to use Ascent in a CMake-based build system (``using-with-cmake``, ``using-with-cmake-mpi``) and via a Makefile (``using-with-make``, ``using-with-make-mpi``). You can read more details about these examples :ref:`using_in_your_project`.

Under ``src/examples/proxies``  you can find example integrations using ascent in the Lulesh, Kripke, and Cloverleaf3D proxy-applications.
In ``src/examples/synthetic/noise`` you can find an example integration using our synthetic smooth noise application.


.. _building_with_docker:

Building Ascent in a Docker Container
---------------------------------------

Under ``src/examples/docker/master/ubuntu`` there is an example ``Dockerfile`` which can be used to create an ubuntu-based docker image with a build of the Ascent github master branch. There is also a script that demonstrates how to build a Docker image from the Dockerfile (``example_build.sh``) and a script that runs this image in a Docker container (``example_run.sh``). The Ascent repo is cloned into the image's file system at ``/ascent``, the build directory is ``/ascent/build-debug``, and the install directory is ``/ascent/install-debug``.


.. _building_manually:

Building Ascent Dependencies Manually
-------------------------------------

In some environments, a spack build of Ascent's dependencies can fail or a user may prefer to build the dependencies manually.

Here is a `script <https://github.com/Alpine-DAV/ascent/blob/develop/scripts/build_ascent/build_ascent.sh>`_ that demonstrates how to build Ascent and its main dependencies without device support:

.. literalinclude:: ../../../scripts/build_ascent/build_ascent.sh
   :language: bash


Here is a `script <https://github.com/Alpine-DAV/ascent/blob/develop/scripts/build_ascent/build_ascent_hip.sh>`_ that shows how to build Ascent and its main dependencies with ROCm/HIP device support:

.. literalinclude:: ../../../scripts/build_ascent/build_ascent_hip.sh
   :language: bash

Here is a `script <https://github.com/Alpine-DAV/ascent/blob/develop/scripts/build_ascent/build_ascent_cuda.sh>`_ that shows how to build Ascent and its main dependencies with CUDA device support:

.. literalinclude:: ../../../scripts/build_ascent/build_ascent_cuda.sh
   :language: bash

Here is script that shows how to build additional dependencies for bflow-stats (babelflow+pmt+streamstats+topo_reader):

.. code:: bash

   root_dir=$(pwd)
    
   # babelflow v1.0.1
   git clone  --recursive https://github.com/sci-visus/BabelFlow.git
   git checkout v1.0.1

   # pmt v1.0.2
   git clone https://bitbucket.org/cedmav/parallelmergetree.git
   git checkout v1.0.2

   # STREAMSTATS
   git clone https://github.com/xuanhuang1/STREAMSTAT.git

   # topo_reader
   git clone https://github.com/xuanhuang1/topo_reader.git

   # build
   # build babelflow 1.0.1

   babelflow_src_dir=${root_dir}/BabelFlow
   babelflow_build_dir=${root_dir}/BabelFlow/build
   babelflow_install_dir=${root_dir}/BabelFlow/install
   
   cmake -S ${babelflow_src_dir} -B ${babelflow_build_dir} \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${babelflow_install_dir} \
      -DBUILD_SHARED_LIBS=ON \
      -DCRAYPE_LINK_TYPE=dynamic \
      -DENABLE_MPI=ON \
      -DENABLE_FIND_MPI=OFF 
  
   cmake --build ${babelflow_build_dir} -j6
   cmake --install ${babelflow_build_dir}


   # build parallelmergetree 1.0.2                                                                           

   parallelmergetree_src_dir=${root_dir}/parallelmergetree
   parallelmergetree_build_dir=${root_dir}/parallelmergetree/build
   parallelmergetree_install_dir=${root_dir}/parallelmergetree/install

   cmake -S ${parallelmergetree_src_dir} -B ${parallelmergetree_build_dir} \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${parallelmergetree_install_dir} \
      -DBUILD_SHARED_LIBS=ON \
      -DCRAYPE_LINK_TYPE=dynamic \
      -DLIBRARY_ONLY=ON\
      -DBabelFlow_DIR=${babelflow_install_dir}

   cmake --build ${parallelmergetree_build_dir} -j6
   cmake --install ${parallelmergetree_build_dir}


   # build topo_reader

   topo_reader_src_dir=${root_dir}/topo_reader/TopologyFileParser
   topo_reader_build_dir=${root_dir}/topo_reader/TopologyFileParser/build
   topo_reader_install_dir=${root_dir}/topo_reader/TopologyFileParser/install

   cmake -S ${topo_reader_src_dir} -B ${topo_reader_build_dir} \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${topo_reader_install_dir} \
      -DBUILD_SHARED_LIBS=ON \
      -DCRAYPE_LINK_TYPE=dynamic \
      -DFUNCTION_TYPE=double

   cmake --build ${topo_reader_build_dir} -j6
   cmake --install ${topo_reader_build_dir}


   # build STREAMSTAT

   STREAMSTAT_src_dir=${root_dir}/STREAMSTAT/
   STREAMSTAT_build_dir=${root_dir}/STREAMSTAT/build
   STREAMSTAT_install_dir=${root_dir}/STREAMSTAT/install

   cmake -S ${STREAMSTAT_src_dir} -B ${STREAMSTAT_build_dir} \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${STREAMSTAT_install_dir} \
      -DBUILD_SHARED_LIBS=ON \
      -DCRAYPE_LINK_TYPE=dynamic 

   cmake --build ${STREAMSTAT_build_dir} -j6
   cmake --install ${STREAMSTAT_build_dir}
   
   # add the path in ascent build
   
   #ascent_src_dir=${root_dir}/ascent/src
   #ascent_build_dir=${root_dir}/ascent/build
   #ascent_install_dir=${root_dir}/ascent/install
   
   #cmake -S ${ascent_src_dir} -B ${ascent_build_dir} \ 
   #   -DBABELFLOW_DIR=${babelflow_install_dir} \
   #   -DPMT_DIR=${parallelmergetree_install_dir} \
   #   -DStreamStat_DIR=${STREAMSTAT_install_dir}\
   #   -DTopoFileParser_DIR=${topo_reader_install_dir}


.. _paraview_ascent_support:

ParaView Support
----------------

Ascent ParaView support is in `src/examples/paraview-vis <https://github.com/Alpine-DAV/ascent/blob/develop/src/examples/paraview-vis>`_ directory.
This section describes how to configure, build and run the example
integrations provided with Ascent and visualize the results insitu
using ParaView. ParaView pipelines are provided for all example
integrations.  We describe in details the ParaView pipeline for
``cloverleaf3d`` in the :ref:`paraview_visualization` section.

.. _paraview_setup_spack:

Setup spack
^^^^^^^^^^^
Install spack, modules and shell support.

- Clone the spack repository:

  .. code:: bash

            git clone https://github.com/spack/spack.git
            cd spack
            source share/spack/setup-env.sh

- If the ``module`` command does not exist:

  - install ``environment-modules`` using the package manager for your system.

  - run ``add.modules`` to add the ``module`` command to your ``.bashrc`` file

  - Alternatively run ``spack bootstrap``

.. _paraview_install:

Install ParaView and Ascent
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- For MomentInvariants (optional module in ParaView for pattern
  detection) visualization patch ParaView to enable this module:

  - Download the `MomentInvariants patch <https://github.com/Alpine-DAV/ascent/blob/develop/src/examples/paraview-vis/paraview-package-momentinvariants.patch>`_

  - Patch paraview: ``patch -p1 < paraview-package-momentinvariants.patch``

- Install ParaView (any version >= 5.7.0). When running on Linux we prefer ``mpich``,
  which can be specified by using ``^mpich``.

  - ``spack install paraview+python3+mpi+osmesa``

  - for CUDA use: ``spack install paraview+python3+mpi+osmesa+cuda``

- Install Ascent

  - ``spack install ascent~vtkh+python``

  - If you need ascent built with vtkh you can use ``spack install
    ascent+python``. Note that you need specific versions of
    ``vtkh`` and ``vtkm`` that work with the version of Ascent built.  Those
    versions can be read from ``scripts/uberenv/project.json``
    by cloning ``spack_url``, branch ``spack_branch``.
    ``paraview-package-momentinvariants.patch`` is already setup to
    patch ``vtkh`` and ``vthm`` with the correct versions, but make sure
    it is not out of date.

- Load required modules: ``spack load conduit;spack load python;spack load py-numpy;spack load py-mpi4py;spack load paraview``

.. _paraview_run_integrations:

Setup and run example integrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can test Ascent with ParaView support by running the available
integrations. Visualization images will be generated in the current
directory. These images can be checked against the images in
``src/examples/paraview-vis/tests/baseline_images``.

- Test ``proxies/cloverleaf3d``

  - Go to a directory where you intend to run cloverleaf3d integration
    (for ``summit.olcf.ornl.gov`` use a member work directory such as
    ``cd $MEMBERWORK/csc340``) so that the compute node can write there.

  - Create links to required files for cloverleaf3d:

    - ``ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview_ascent_source.py``

    - ``ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview-vis-cloverleaf3d.py paraview-vis.py``
      for surface visualization.

    - Or ``ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/paraview-vis-cloverleaf3d-momentinvariants.py paraview-vis.py``
      for MomentInvariants visualization (Optional)

    -
      .. code:: bash

              ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/ascent_actions.json
              ln -s $(spack location --install-dir ascent)/examples/ascent/paraview-vis/expandingVortex.vti
              ln -s $(spack location --install-dir ascent)/examples/ascent/proxies/cloverleaf3d/clover.in

  - Run the simulation

    ``$(spack location --install-dir mpi)/bin/mpiexec -n 2 $(spack location --install-dir ascent)/examples/ascent/proxies/cloverleaf3d/cloverleaf3d_par > output.txt 2>&1``

  - Examine the generated images

- Similarily test ``proxies/kripke``, ``proxies/laghos``,
  ``proxies/lulesh``, ``synthetic/noise``. After you create the
  apropriate links similarily with ``cloverleaf3d`` you can run these
  simulations with:

  - ``$(spack location --install-dir mpi)/bin/mpiexec -np 8 ./kripke_par --procs 2,2,2 --zones 32,32,32 --niter 5 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4 > output.txt 2>&1``

  - ``$(spack location --install-dir mpi)/bin/mpiexec -n 8 ./laghos_mpi -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.6 -visit -pa > output.txt 2>&1``

  - ``$(spack location --install-dir mpi)/bin/mpiexec -np 8 ./lulesh_par -i 10 -s 32 > output.txt 2>&1``

  - ``$(spack location --install-dir mpi)/bin/mpiexec -np 8 ./noise_par  --dims=32,32,32 --time_steps=5 --time_delta=1 > output.txt 2>&1``


Setup and run on summit.olcf.ornl.gov
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Execute section :ref:`paraview_setup_spack`

- Configure spack

  - add a file ``~/.spack/packages.yaml`` with the following content as detailed next.
    This insures that we use spectrum-mpi as the MPI runtime.

    .. code:: yaml

          packages:
            spectrum-mpi:
              buildable: false
              externals:
              - modules:
                - spectrum-mpi/10.3.1.2-20200121
                spec: spectrum-mpi@10.3.1.2-20200121
            cuda:
              buildable: false
              externals:
              - modules:
                - cuda/10.1.168
                spec: cuda@10.1.168


  - Load the correct compiler:

    .. code:: bash

          module load gcc/7.4.0
          spack compiler add
          spack compiler remove gcc@4.8.5


- Compile spack packages on a compute node

  - For busy summit I got ``internal compiler error`` when
    compiling ``llvm`` and ``paraview`` on the login node. To fix this, move
    the spack installation on ``$MEMBERWORK/csc340`` and
    compile everything on a compute node.

    - First login to a compute node: ``bsub -W 2:00 -nnodes 1 -P CSC340 -Is /bin/bash``

    - Install all spack packages as in :ref:`paraview_install` with ``-j80`` option (there are 84 threads)

    - Disconnect from the compute node: ``exit``.

- Continue with :ref:`paraview_run_integrations` but run the integrations as described next:

  - Execute cloverleaf
    ``bsub $(spack location --install-dir ascent)/examples/ascent/paraview-vis/summit-moment-invariants.lsf``
  - To check if the integration finished use: ``bjobs -a``

Nightly tests
^^^^^^^^^^^^^

We provide a docker file for Ubuntu 18.04 and a script that installs
the latest ParaView and Ascent, runs the integrations provided with
Ascent, runs visualizations using ParaView pipelines and checks the
results. See ``tests/README-docker.md`` for how to create the docker
image, run the container and execute the test script.

Notes
^^^^^

- Global extents are computed for uniform and rectilinear topologies but
  they are not yet computed for a structured topology (lulesh). This
  means that for lulesh and datasets that have a structured topology we
  cannot save a correct parallel file that represents the whole dataset.

- For the ``laghos`` simulation accessed through Python extracts
  interface, only the higher order mesh is accessible at the moment,
  which is a uniform dataset. The documentation shows a non-uniform mesh but
  that is only available in the ``vtkm`` pipelines.



