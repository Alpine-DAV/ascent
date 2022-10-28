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


Enabling RocProf in Your Build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order for rocProf to profile your application, you must compile your build with the following modules and library flags: 

.. code-block:: sh

   module load PrgEnv-cray    
   module load craype-accel-amd-gfx90a    
   module load rocm/5.2.0    
    
   export CXX="$(which CC) -x hip"   
   export CXXFLAGS="-ggdb -03 -std=c++17 -Wall" 
   export LD="$(which CC)"
   export LDFLAGS="${CXXFLAGS} -L${ROCM_PATH}/lib"
   export LIBS="-lamdhip64"

Many of the modules and flags required for rocProf are required in order to execute on the GPU. 
Below are the modules and flags that are required by Ascent and its dependencies to enable HIP on Crusher: 

.. code-block:: sh

   module load PrgEnv-cray    
   module load craype-accel-amd-gfx90a    
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
   export CXX=$(which CC)
   export FC=$(which ftn)
   export CFLAGS="-I${ROCM_PATH}/include"
   export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
   export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
 
When combined, the modules and flags needed to enable both HIP and rocProf on Crusher become the following:

.. code-block:: sh

   module load PrgEnv-cray    
   module load craype-accel-amd-gfx90a    
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
   export CXXFLAGS="-I${ROCM_PATH}/include -ggdb -O3 -std=c++17 -Wall -Wno-pass-failed"
   export LD=${CC}
   export LDFLAGS="${CXXFLAGS} -L${ROCM_PATH}/lib -lamdhip64"
   export LIBS="-lamdhip64"
 

Running on the GPU
^^^^^^^^^^^^^^^^^^
This section will demonstrate submitting a job on Crusher via slurm and a batch script. 
The ```job.sh``` batch script will define submissions options, load necessary modules, and launch the parallel job. 
Below is an example ```job.sh``` that launches the Kripke simulation, one of the example integrations provided by Ascent. 

.. code-block:: sh

   #!/bin/bash
   #SBATCH -A csc340
   #SBATCH -t 00:10:00
   #SBATCH -N 1
   #SBATCH -J kripke_gpu
   #SBATCH -o kripke.output
   #SBATCH -e kripke.error
   
   
   module load PrgEnv-cray
   module load craype-accel-amd-gfx90a
   module load rocm/5.2.0
   module load cmake/3.22.2
   module load cray-mpich
   module load cray-hdf5-parallel/1.12.1.1
   
   srun -n 6 --ntasks-per-node 6 -G 6 --gpus-per-node 6 ./kripke_par --procs 1,1,6  --zones 60,60,60 --niter 3 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4

In the above example, the ``#SBATCH`` options are: 

 =========================== ==============================================================================================
 Option                      Description
 =========================== ==============================================================================================
  -A                          Account to charge submitted job 
  -t                          Requested walltime for submitted job
  -N                          Number of nodes
  -J                          Job Name
  -o                          stdout file name
  -e                          stderr file name
 =========================== ==============================================================================================

The parallel job is launched using ``srun`` with the follow options:

 =========================== ==============================================================================================
 Option                      Description
 =========================== ==============================================================================================
  -n                          Number of MPI ranks
  --ntasks-per-node           Number of MPI ranks per node 
  -G                          Number of GPUs
  --gpus-per-node             Number of GPUs per node
 =========================== ==============================================================================================


For more information on running jobs on Crusher, see their user guide: `https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#running-jobs`.  
And for more information about running Kripke, see :ref:`kripke_integration`. 

Profiling with RocProf
^^^^^^^^^^^^^^^^^^^^^^
You can profile your job using rocProf by adding the necessary options to the ``job.sh`` batch script, following the ``srun`` launch command (and respective options). 
One way to confirm your job is executing on the GPU, is to use rocProf with the --hip-trace flag, this will generate a file that will show the HIP function calls and kernels. 
But an easy way to verify your job is NOT running on the GPU is if rocProf produces no output files. 

Below is an example of using rocProf that will generate an output file for each MPI rank. 
Note: this could result in several GBs of data depending on the size of your job.
There are typically 6 files generated per MPI rank.

.. code-block:: sh

   #!/bin/bash
   #SBATCH -A csc340
   #SBATCH -t 00:10:00
   #SBATCH -N 1
   #SBATCH -J kripke_gpu
   #SBATCH -o kripke.output
   #SBATCH -e kripke.error
   
   
   module load PrgEnv-cray
   module load craype-accel-amd-gfx90a
   module load rocm/5.2.0
   module load cmake/3.22.2
   module load cray-mpich
   module load cray-hdf5-parallel/1.12.1.1
   
   srun -n 6 --ntasks-per-node 6 -G 6 --gpus-per-node 6 \
     'rocprof -o ${SLURM_JOBID}-${SLURM_PROCIDd}.csv --hip-trace ./kripke_par --procs 1,1,6  --zones 60,60,60 --niter 3 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4'

Below is an example of using rocProf that will only generate an output file for the MPI root rank. 

.. code-block:: sh

   #!/bin/bash
   #SBATCH -A csc340
   #SBATCH -t 00:10:00
   #SBATCH -N 1
   #SBATCH -J kripke_gpu
   #SBATCH -o kripke.output
   #SBATCH -e kripke.error
   
   
   module load PrgEnv-cray
   module load craype-accel-amd-gfx90a
   module load rocm/5.2.0
   module load cmake/3.22.2
   module load cray-mpich
   module load cray-hdf5-parallel/1.12.1.1
   
   srun -n 6 --ntasks-per-node 6 -G 6 --gpus-per-node 6 \
     'if [${SLURM_PROCID -eq 0 ]; rocprof --hip-trace ./kripke_par --procs 1,1,6  --zones 60,60,60 --niter 3 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4; \
     else ./kripke_par --procs 1,1,6  --zones 60,60,60 --niter 3 --dir 1:2 --grp 1:1 --legendre 4 --quad 4:4; fi'

Helpful RocProf Flags
^^^^^^^^^^^^^^^^^^^^^

**Help Flag:**

 =========================== ==============================================================================================
 Option                      Description                                  
 =========================== ==============================================================================================
  /rocm/bin/rocprof -h        Display command line options                                                               
 =========================== ==============================================================================================
   
**Housekeeping Flags:**

 =========================== ==================================================== ================================================
 Option                      Description                                          Default
 =========================== ==================================================== ================================================
  --timestamp <on|off>        Turn on/off gpu kernel dispatch timestamps           **OFF**
  --basenames <on|off>        Turn on/off turncating gpu kernel names such         **OFF**
                              as template parameters and arguments types
  -o <output csv file>        Save direct counter data to a specified file name    <input file base>.csv
  -d <directory>              Send profiling data to a specified directory         /tmp
  -t <temporary directory>    Change the directory where tmp files are created     /tmp
                              allowing you to save these files
 =========================== ==================================================== ================================================
        
**Profiling Flags:**

 =========================== ==============================================================================================
 Option                      Description                                  
 =========================== ==============================================================================================
  -i input<.txt|.html>        Specify an input file (note: all output files will be named input.\*)
  --stats                     Generate a file of kernel execution stats called <output file>.stats.csv
  --hsa-trace                 Trace GPU kernels, host HSA events, and HIP memory copies. Generates JSON file that is 
                              compatible with chrome-tracing
  --hip-trace                 Trace HIP API calls. Generates JSON file that is compatible with chrome-tracing
 =========================== ==============================================================================================



