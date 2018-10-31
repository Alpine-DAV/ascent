.. ############################################################################
.. # Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Ascent.
.. #
.. # For details, see: http://ascent.readthedocs.io/.
.. #
.. # Please also read ascent/LICENSE
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


Extracts
========
Extracts are an abstraction that enables the user to specify how they want to capture their data.
In terms of Ascent, data capture sends data outside the Ascent infrastructure.
Examples include writing out the raw simulation data to the file system, creating HDF5 files, or sending the data off node (e.g., ADIOS).

Currently supported extracts include:
    
    * Python : use a python script with NumPy to analyze mesh data
    * Relay : leverages Conduit's Relay library to do parallel I/O 
    * ADIOS : use ADIOS to send data to a separate resource


Python
------
Python extracts can execute arbitrary python code. Python code uses Conduit's python interface
to interrogate and retrieve mesh data. Code is executed on each MPI rank, and mpi4py can be 
used for collective communication.

.. code-block:: c++

  conduit::Node extracts;
  extracts["e1/type"]  = "python";
  extracts["e1/params/source"] = py_script;


Python source code is loaded into Ascent via a string that could be loaded from the file system

.. code-block:: c++

  import numpy as np
  from mpi4py import MPI
  
  # obtain a mpi4py mpi comm object
  comm = MPI.Comm.f2py(ascent_mpi_comm_id())
  
  # get this MPI task's published blueprint data
  mesh_data = ascent_data().child(0)
  
  # fetch the numpy array for the energy field values
  e_vals = mesh_data["fields/energy/values"]
  
  # find the data extents of the energy field using mpi
  
  # first get local extents
  e_min, e_max = e_vals.min(), e_vals.max()
  
  # declare vars for reduce results
  e_min_all = np.zeros(1)
  e_max_all = np.zeros(1)
  
  # reduce to get global extents
  comm.Allreduce(e_min, e_min_all, op=MPI.MIN)
  comm.Allreduce(e_max, e_max_all, op=MPI.MAX)
  
  # compute bins on global extents 
  bins = np.linspace(e_min_all, e_max_all)
  
  # get histogram counts for local data
  hist, bin_edges = np.histogram(e_vals, bins = bins)
  
  # declare var for reduce results
  hist_all = np.zeros_like(hist)
  
  # sum histogram counts with MPI to get final histogram
  comm.Allreduce(hist, hist_all, op=MPI.SUM)

The example above shows how a python script could be used to create a distributed-memory
histogram of a mesh variable that has been published by a simulation.


.. code-block:: python 

  import conduit
  import ascent.mpi
  # we treat everything as a multi_domain in ascent so grab child 0
  n_mesh = ascent_data().child(0)
  ascent_opts = conduit.Node()
  ascent_opts['mpi_comm'].set(ascent_mpi_comm_id())
  a = ascent.mpi.Ascent()
  a.open(ascent_opts)
  a.publish(n_mesh)
  actions = conduit.Node()
  scenes  = conduit.Node()
  scenes['s1/plots/p1/type'] = 'pseudocolor'
  scenes['s1/plots/p1/params/field'] = 'radial_vert'
  scenes['s1/image_prefix'] = 'tout_python_mpi_extract_inception'
  add_act =actions.append()
  add_act['action'] = 'add_scenes'
  add_act['scenes'] = scenes
  actions.append()['action'] = 'execute'
  a.execute(actions)
  a.close()

In addition to performing custom python analysis, your can create new data sets and plot them
through a new instance of Ascent. We call this technique Inception. For 

Relay
-----
Relay extracts saves data to the file system. Currently, Relay supports saving files in two Blueprint formats: HDF5 and json (default).
By default, Relay saves the published mesh data to the file system, but is a pipeline is specified, then the result of the
pipeline is saved. Relay extracts can be opened by post-hoc tools such as VisIt.

.. code-block:: c++ 

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "radial_vert";
    contour_params["iso_values"] = 250.;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";

    extracts["e1/params/path"] = output_file;

In this example, a contour of a field is saved to the file system in json form. 
To save the files in HDF5 format:

.. code-block:: c++ 

    extracts["e1/params/protocol"] = "blueprint/mesh/hd5f";

ADIOS
-----
The current ADIOS extract is experimental and this section is under construction.
