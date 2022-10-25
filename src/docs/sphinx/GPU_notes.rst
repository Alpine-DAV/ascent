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
.. list-table::
   :header-rows: 1
   * Help Flag
   
   * /rocm/bin/rocprof -h
   
Helpful Flags
^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * Housekeeping Flags
   
   * --timestamp <on|off>: turn on/off gpu kernel dispatch timestamps
        
   * --basenames <on|off>: turn on/off turncating gpu kernel names such as template parameters and arguments types
      
   * -o <output csv file>: save direct counter data to a specified file name
        
   * -d <directory>: send profiling data to a specified directory
        
   * -t <temporary directory>: change the directory where files are created from /tmp to temporary directory, allowing you to save these files
        
Helpful Flags
^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1
   
   * Profiling Flags
   
   * -i input<.txt|.html>: specify an input file (note: all output files will be named input.\*)
        
   * --stats: generate a file of kernel execution stats called <output file>.stats.csv
        
   * --hsa-trace: trace GPU kernels, host HSA events, and HIP memory copies. Generates JSON file that is compatible with chrome-tracing
        
   * --hip-trace: trace HIP API calls. Generates JSON file that is compatible with chrome-tracing

