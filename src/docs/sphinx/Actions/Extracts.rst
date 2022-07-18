.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _extracts:

Extracts
========
Extracts are an abstraction that enables the user to specify how they want to capture their data.
In terms of Ascent, data capture sends data outside the Ascent infrastructure.
Examples include writing out the raw simulation data to the file system, creating HDF5 files, or sending the data off node (e.g., ADIOS).

Currently supported extracts include:

    * Python : use a python script with NumPy to analyze mesh data
    * Relay : leverages Conduit's Relay library to do parallel I/O
    * HTG : write a VTK HTG (HyperTreeGrid) file
    * ADIOS : use ADIOS to send data to a separate resource

.. _extracts_python:

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
through a new instance of Ascent. We call this technique Inception.


.. _extracts_relay:

Relay
-----
Relay extracts save data to the file system. Currently, Relay supports saving data to Blueprint HDF5, YAML, or JSON files.
By default, Relay saves the published mesh data to the file system, but if a pipeline is specified, then the result of the
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

    extracts["e1/params/protocol"] = "hdf5";

``yaml`` and ``json`` are also valid ``protocol`` options.


By default, the relay extract creates one file per mesh domain saved. You can control
the number of files written (aggregating multiple domains per file) using the
``num_files`` parameter:

.. code-block:: c++

    extracts["e1/params/num_files"] = 2;


Additionally, Relay supports saving out only a subset of the data. The ``fields`` parameters is a list of
strings that indicate which fields should be saved.

.. code-block:: c++

    extracts["e1/params/fields"].append("density");
    extracts["e1/params/fields"].append("pressure");

.. _extracts_htg:

HTG
---
HTG extracts save data to the file system as a VTK HyperTreeGrid.
HyperTreeGrid is a tree based uniform grid for element based data.
The current implementation writes out binary trees from uniform grids.
As such there are a number of limitations on the type of data it writes out.
These include the following:

    * The mesh must be a uniform grid.
    * The mesh must have a pwer of 2 number of elements in each direction.
    * The mesh dimensions must be the same in each direction.
    * The fields must be element based.

The extract also takes a ``blank_value`` parameter that specifies a field value that indicates that the cell is empty.

.. code-block:: c++

    conduit::Node data;
    conduit::blueprint::mesh::examples::basic("uniform", 33, 33, 33, data);

    conduit::Node extracts;
    extracts["e1/type"]  = "htg";

    extracts["e1/params/path"] = "basic_mesh33x33x33";
    extracts["e1/params/blank_value"] = -10000.;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    //
    // Run Ascent
    //
    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

In this example, the field is saved to the file system in ``basic_mesh33x33x33.htg``.

Additionally, HTG supports saving out only a subset of the data.
The fields parameters is a list of strings that indicate which fields should be saved.

.. code-block:: c++

    extracts["e1/params/fields"].append("density");
    extracts["e1/params/fields"].append("pressure");


.. _extracts_flatten:

Flatten
-------
Flatten extracts save data to the file system. Currently, Flatten supports saving data to Blueprint HDF5, YAML, CSV (default), or JSON files.
By default, Flatten saves the published mesh data to the file system, but if a pipeline is specified, then the result of the
pipeline is saved. 
Flatten transforms the data from Blueprint Meshes to Blueprint Tables. 
This extract generates two files: one for vertex data and one for element data. 

This extract requires a ``path`` for the location of the resulting files. 
Optional parameters include ``protocol`` for the type of output file (default is CSV), and ``fields``, which specifies the fields to be included in the files (default is all present fields). 

ADIOS
-----
The current ADIOS extract is experimental and this section is under construction.
