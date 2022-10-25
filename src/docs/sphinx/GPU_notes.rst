.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################


Ascent GPU Notes
==================

Number of Ranks Per Node
------------------------
On each node, it is important to only launch one MPI rank per GPU.
If your cluster contains multiple GPUs per node, it is important that consecutive ranks be assigned to each node, which is the default behavior of most job schedulers.
By default, each CUDA capable GPU device is queried, and a rank is assigned a device based on the rank number modulo the number of devices.
Collisions could result in a run-time failure or significant delays. 
This default behavior can be overridden. Please see the Ascent options for more details.

Using RocProf on AMD GPUs
-------------------------
RocProf is a command line tool provided by ROCm that is implemented on top of the rocProfiler and rocTracer APIs.
RocProf allows for application tracing and counter collection on AMD GPUs, and can be a useful tool for confirming your application is running on the GPU as expected.  
Below you will find a basic guide to using rocProf with Ascent. 

Helpful Flags
^^^^^^^^^^^^^
**Help Flag:**

.. list-table::
   
   * - /rocm/bin/rocprof -h
   
**Housekeeping Flags:**

.. list-table::
   
   * - --timestamp <on|off> **:** turn on/off gpu kernel dispatch timestamps
         
   * - --basenames <on|off> **:** turn on/off turncating gpu kernel names such as template parameters and arguments types
       
   * - -o <output csv file> **:** save direct counter data to a specified file name
         
   * - -d <directory> **:** send profiling data to a specified directory
         
   * - -t <temporary directory> **:** change the directory where files are created from /tmp to temporary directory, allowing you to save these files
        
**Profiling Flags:**

.. list-table::
   
   * - -i input<.txt|.html> **:** specify an input file (note: all output files will be named input.\*)
         
   * - --stats **:** generate a file of kernel execution stats called <output file>.stats.csv
         
   * - --hsa-trace **:** trace GPU kernels, host HSA events, and HIP memory copies. Generates JSON file that is compatible with chrome-tracing
         
   * - --hip-trace **:** trace HIP API calls. Generates JSON file that is compatible with chrome-tracing

  ==================== ============================================== ================================================
  Option               Description                                     Default
  ==================== ============================================== ================================================
   -i input<.txt|.html> specify an input file (note: all output files
                        will be named input.\*)
   --stats              generate a file of kernel execution stats 
                        called <output file>.stats.csv
   --hsa-trace          trace GPU kernels, host HSA events, and HIP 
                        memory copies. Generates JSON file that is 
                        compatible with chrome-tracing
   --hip-trace          trace HIP API calls. Generates JSON file 
                        that is compatible with chrome-tracing
   --prefix             Destination directory                          ``uberenv_libs``
   --spec               Spack spec                                     linux: **%gcc**
                                                                       osx: **%clang**
   --spack-config-dir   Folder with Spack settings files               linux: (empty)
                                                                       osx: ``scripts/uberenv_configs/spack_configs/darwin/``
   -k                   Ignore SSL Errors                              **False**
   --install            Fully install ascent not just dependencies     **False**
  ==================== ============================================== ================================================

Enabling RocProf in Your Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order for rocProf to profile your application, you must compile your build with the following modules and library flags: 
  * module load PrgEnv-cray
  * module load craype-accel-amd-gfx90a
  * module load rocm/5.2.0
  * 
  * export CXX="$(which CC) -x hip"
  * export CXXFLAGS="-ggdb -03 -std=c++17 -Wall"
  * export LD="$(which CC)"
  * export LDFLAGS="${CXXFLAGS} -L${ROCM_PATH}\lib"
  * export LIBS="-lamdhip64"

Many of the modules and flags required for rocProf are required in order to execute on the GPU. 
Below are the modules and flags that are required by Ascent and its dependencies to enable HIP on Crusher: 
 ::
 module load cmake/3.22.2
 module load craype-accel-amd-gfx90a
 module load rocm/5.2.0
 module load cray-mpich

 # we want conduit to use this
 module load cray-hdf5-parallel/1.12.1.1

 # GPU-aware MPI
 export MPICH_GPU_SUPPORT_ENABLED=1

 export AMREX_AMD_ARCH=gfx90a

 export CC=$(which cc)
 export CXX="$(which CC) -x hip"
 export FC=$(which ftn)
 export CFLAGS="-I${ROCM_PATH}/include"
 export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed -ggdb -03 -std=c++17 -Wall"
 export LD="$(which CC)"
 export LDFLAGS="${CXXFLAGS} -L${ROCM_PATH}/lib -lamdhip64"
 export LIBS="-lamdhip64"
 

