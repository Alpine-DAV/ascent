.. ############################################################################
.. # Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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

Ascent API
============
The top level API for ascent consists of five calls:

  - open(condiut::Node)
  - publish(conduit::Node)
  - execute(conduit::Node)
  - info(conduit::Node)
  - close()

.. _ascent_api_open:

open
----
Open provides the initial setup of Ascent from a Conduit Node.
Options include runtime type (e.g., ascent, flow, or empty) and associated backend if available.
If running in parallel (i.e., MPI), then a MPI comm handle must be supplied.
Ascent will always check the file system for a file called ``ascent_options.json`` that will override compiled in options, and for obvious reasons, a MPI communicator cannot be specified in the file.
Here is a file that would set the runtime to the main ascent runtime using a OpenMP backend (inside VTK-m):


.. code-block:: json

  {
    "runtime/type"    : "ascent",
    "runtine/vtkm/backend" : "openmp"
  }

Example Options
"""""""""""""""
A typical integration will include the following code:

.. code-block:: c++

  Ascent ascent;
  conduit::Node ascent_options;

  #if USE_MPI
  ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  #endif
  ascent_options["runtime/type"] = "ascent";
  ascent_options["runtime/backend"] = "openmp";

  ascent.Open(ascent_options);


Default Directory
"""""""""""""""""
By default, Ascent will output files in the current working directory.
This can be overrided by specifying the ``default_dir``. This directory
must be a valid directory, i.e., Ascent will not create this director for
you. Many Ascent filters have parameters that specify output files, and Ascent
will only place files that do not have an absolue path specified.
For example, the ``my_image`` would be written to the default directory, but
``/some/other/path/my_image`` would be written in the directory
``/some/other/path/``.

.. code-block:: json

  {
    "default_dir" : "/path/to/output/dir"
  }

High Order Mesh Refinement
""""""""""""""""""""""""""
If MFEM is enabled, one additional options argument, ``refinement_level`` can be specified.
High-order meshes variable are continuous polynomial functions that cannot be captured
by linear low-order meshes. In order to approximate the functions with less error,
high-order elements are discretized into many linear elements. The minimum value for refinement
is ``1``, i.e., no refinement. There is a memory-accuracy trade-off when using refinement.
The higher the value,
the more accurate the low-order representation is, but more discretization means more memory
usage and more time tp process the additional elements.

.. code-block:: json

  {
    "refinement_level" : 4
  }

Runtime Options
"""""""""""""""
Valid runtimes include:

  - ``ascent``

  - ``flow``

  - ``empty``


Logging Options
"""""""""""""""
There are a few other options that control behaviors common to all runtimes:

 * ``messages``

   Controls if logged info messages are printed or muted.

   Supported values:

    - ``quiet`` (default if omitted) Logged info messages are muted

    - ``verbose``  Logged info messages are printed

Exception Handling
""""""""""""""""""
If ascent is not behaving as expected, a good first step is to enable verbose messaging.
There are often warnings and other information that can indicate potential issues.

 * ``exceptions``

   Controls if Ascent traps or forwards C++ exceptions that are thrown.

   Supported values:

    - ``forward`` (default if omitted) Exceptions thrown will propagate to the calling code

    -  ``catch`` Catches conduit::Error exceptions at the Ascent interface and prints info about the error to standard out.
       This case this provides an easy way to prevent host program crashes when something goes wrong in Ascent.

By default, Ascent looks for a file called ``ascent_actions.json`` that can append additional actions at runtime.
This default file name can be overridden in the Ascent options:

.. code-block:: c++

    ascent_opts["actions_file"] = custom_ascent_actions_file;

When running on the GPU, Ascent will automatically choose which GPU to run code on if there are
multiple available, unless told otherwise. In the default configuration, it is important to
launch one MPI task per GPU. This default behavior can be overridden with the following option:

.. code-block:: c++

    ascent_opts["cuda/init"] = "false";

By disabling CUDA GPU initialization, an application is free to set the active device.

Filter Timings
""""""""""""""
Ascent has internal timings for filters. The timings output is one csv file
per MPI rank.

.. code-block:: json

  {
    "timings" : "true"
  }


Field Filtering
"""""""""""""""
By default, Ascent passes all of the published data to. Some simulations
have just a few variables that they publish, but other simulations an
publish 100s of variables to Ascent. In this case, its undesirable to
use all fields when the actions only need a single variable. This reduces
the memory overhead Ascent uses.

Field filtering scans the user's actions to identify what fields are required,
only passing the required fields into Ascent. However, there are several
actions where the required fields cannot be resolved. For example, saving simulation
data to the file system saves all fields, and in this case, it is not possible to resolve
the required fields. If field filtering encounters this case, then an error is generated.
Alternatively, if the actions specify which fields to save, then this field filtering
can resolve the fields.

.. code-block:: json

  {
    "field_filtering" : "true"
  }



publish
-------
This call publishes data to Ascent through `Conduit Blueprint <http://llnl-conduit.readthedocs.io/en/latest/blueprint.html>`_ mesh descriptions.
In the Lulesh proxy-app, data is already in a form that is compatible with the blueprint conventions and the code to create the Conduit Node is straight-forward:

.. code-block:: c++

      // provide state information
      mesh_data["state/time"].set_external(&m_time);
      mesh_data["state/cycle"].set_external(&m_cycle);
      mesh_data["state/domain_id"] = myRank;

      // coordinate system data
      mesh_data["coordsets/coords/type"] = "explicit";
      mesh_data["coordsets/coords/x"].set_external(m_x);
      mesh_data["coordsets/coords/y"].set_external(m_y);
      mesh_data["coordsets/coords/z"].set_external(m_z);

      // topology data
      mesh_data["topologies/mesh/type"] = "unstructured";
      mesh_data["topologies/mesh/coordset"] = "coords";
      mesh_data["topologies/mesh/elements/shape"] = "hexs";
      mesh_data["topologies/mesh/elements/connectivity"].set_external(m_nodelist);

      // one or more scalar fields
      mesh_data["fields/p/type"]        = "scalar";
      mesh_data["fields/p/topology"]    = "mesh";
      mesh_data["fields/p/association"] = "element";
      mesh_data["fields/p/values"].set_external(m_p);

If the data does not match the blueprint mesh conventions, then you must transform the data into a compatible format.

You can check if a node confirms to the mesh blueprint using the verify function provided by conduit.

.. code-block:: c++

    #include <conduit_blueprint.hpp>

    Node verify_info;
    if(!conduit::blueprint::mesh::verify(mesh_data,verify_info))
    {
        // verify failed, print error message
        ASCENT_INFO("Error: Mesh Blueprint Verify Failed!");
        // show details of what went awry
        verify_info.print();
    }

Once the Conduit Node has been populated with data conforming to the mesh blueprint, simply publish the data using the Publish call:

.. code-block:: c++

  ascent.Publish(mesh_data);

Publish is called each cycle where Ascent is used.

execute
-------
Execute applies some number of actions to published data.
Each action is described inside of a Conduit Node and passed to the Execute call.
For a full description of supported actions see :ref:`ascent-actions`.

Here is a simple example of adding a plot using the C++ API:

.. code-block:: c++

      // In the main simulation loop
      conduit::Node actions;

      // create a one scene with one plot
      conduit::Node scenes;
      scenes["s1/plots/p1/type"] = "pseudocolor";
      scenes["s1/plots/p1/params/field"] = "braid";

      // add the scenes and execute
      conduit::Node &add_plots = actions.append();
      add_plots["action"] = "add_scenes";
      add_plots["scenes"] = scenes;
      conduit::Node &execute = actions.append();
      execute["action"] = "execute";

      ascent.Publish(mesh_data);
      ascent.Execute(actions);

info
----
Info populates a conduit Node with infomation about Ascent including runtime execution and outputted results.
This information can be used to return data back to the simulation and for debugging purposes.

.. code-block:: c++

  conduit::Node info;
  ascent.info(info);
  info.print();

The data populated inside the info node is as follows:

  - ``runtime``: the default runtime that Ascent used. Unless a custom runtime was used, this value will be ``ascent``.
  - ``registered_filter_types``: a list of filters that have been registered with the Ascent runtime.
  - ``flow_graph``: description of the data flow network that was run with the last ``Execute`` call.
  - ``actions``: the last set of input actions Ascent ran with the last ``Execute`` call.
  - ``images``: a list of image file names and camera parameters that were create in the last call to ``Execute``.
  - ``expressions``: a set of query results from all calls to ``Execute``.

close
-----
Close informs Ascent that all actions are complete, and the call performs the appropriate clean-up.

.. code-block:: c++

  ascent.close();


Error Handling
---------------

  Ascent uses Conduit's error handling machinery. By default when errors occur
  C++ exceptions are thrown, but you can rewire Conduit's handlers with your own callbacks. For more info
  see the `Conduit Error Handling Tutorial <http://llnl-conduit.readthedocs.io/en/latest/tutorial_cpp_errors.html>`_.
  You can also stop exceptions at the Ascent interface using the ``exceptions`` option for :ref:`Ascent::open<ascent_api_open>` .

