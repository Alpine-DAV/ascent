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


Flow Filter Anatomy
===================
The full interface to a Flow filter can be found in the
`Flow filter header file <https://github.com/Alpine-DAV/ascent/blob/develop/src/flow/flow_filter.hpp>`_.
Here is a summary of the functions relevant to a filter developer:

.. code-block:: c++

  public:
    Filter();
    virtual ~Filter();

    // override and fill i with the info about the filter's interface
    virtual void          declare_interface(conduit::Node &i) = 0;

    // override to imp filter's work
    virtual void          execute() = 0;

    // optionally override to allow filter to verify custom params
    // (used as a guard when a filter instance is created in a graph)
    virtual bool          verify_params(const conduit::Node &params,
                                        conduit::Node &info);


A derived filter must minimally implement the ``declare_interface`` and ``execute``
methods, but it is highly encouraged that a new filter implement ``verify_params``
as well. ``verify_params`` alerts uses to errors and unexpected parameters.

.. note::

    Developing a flow filter will require a working knowledge of the Conduit API.
    In the :ref:`tutorials` section under ``Conduit Examples``, there are several
    examples of basic Conduit usage. More Conduit tutorial resources can be found in the
    `Conduit documentation <https://llnl-conduit.readthedocs.io/en/latest/tutorial_cpp.html>`_.

Flow filter implementations are located in the ``src/ascent/runtimes/flow_filters`` directory.

Implementing A New Filter
-------------------------
As a convenience, we have created the
`VTKHNoOp <https://github.com/Alpine-DAV/ascent/blob/develop/src/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.cpp>`_
filter as staring point and reference. Although the NoOp filter demonstrates how to use a
VTK-h filter, the implementation is relevant to anyone developing flow filters in Ascent
regardless of whether VTK-h or VTK-m is used.

Interface Declaration
"""""""""""""""""""""
.. code-block:: c++

    void
    VTKHNoOp::declare_interface(conduit::Node &i)
    {
        i["type_name"]   = "vtkh_no_op";
        i["port_names"].append() = "in";
        i["output_port"] = "true";
    }


* ``type_name``: declares the name of the filter to flow, and the only requirement is that this name be unique.
* ``port_names``: declares a list input port names.
* ``output_port``: declares if this filter has an output of not. Valid values are ``true`` and ``false``.

The ``port_names`` parameter is a list of input port names that can be referenced by name or index
when creating the filter within the runtime. The typical number of inputs is one, but there is no
restriction on the input count. To add additional inputs, additional ``append()`` calls will add
more inputs to the port list, and the input port names must be unique.

.. code-block:: c++

   i["port_names"].append() = "in1";
   i["port_names"].append() = "in2";


For the majority of developers, a transform (i.e., a filter that can be part of a pipeline)
filter will have one input (e.g., the data set) and one output. If creating an extract,
the ``output_port`` should be declared ``false`` indicating that this filter is a sink.

Parameter Verification
""""""""""""""""""""""
Parameters are passed through Ascent and passed to the filters. For detailed
examples of filter in Ascent see the :ref:`pipelines` section.


How Are Parameters Passed
^^^^^^^^^^^^^^^^^^^^^^^^^
The parameters are passed to the Ascent API through Conduit nodes. A simple filters
interface looks like this in c++:

.. code-block:: c++

    conduit::Node filter;
    filter["type"] = "filter_name";
    filter["params/string_param"] = "string";
    filter["params/double_param"] = 2.0;


or equivalently in json:

.. code-block:: json

    {
      "type"   : "filter_name",
      "params":
      {
        "string_param" : "string",
        "double_param" : 2.0
      }
    }

The Ascent runtime looks for the ``params`` node and passes it to the filter
upon creation. Parameter are verified at the time of the filter execution.

Filter Parameter Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``verify_params`` method allow the filter creator to verify the expected parameters
and parameter types before the filter is executed. If the verification fails, error messages
are shown to the users. The method has two parameters: the parameters of the filter and a
conduit node that is populated with error information that flow will show if the result
of the verification is false (error state).

.. code-block:: c++
    :caption: Example parameter verification
    :name: verify
    :linenos:

    bool
    VTKHNoOp::verify_params(const conduit::Node &params,
                            conduit::Node &info)
    {
        info.reset();

        bool res = check_string("field",params, info, true);

        std::vector<std::string> valid_paths;
        valid_paths.push_back("field");

        std::string surprises = surprise_check(valid_paths, params);

        if(surprises != "")
        {
          res = false;
          info["errors"].append() = surprises;
        }

        return res;
    }

Check Parameters
++++++++++++++++
While you can use the Conduit API to check for expected paths and types of values, we
provide a number of methods to streamline common checks. These
`parameter checking helpers <https://github.com/Alpine-DAV/ascent/blob/develop/src/ascent/runtimes/flow_filters/ascent_runtime_param_check.hpp>`_
provide two basic checking mechanisms:
* ``check_string``: checks for the presence of a string parameter
* ``check_numeric``: checks for the presence of a numeric parameter

Both functions have the same signature:

.. code-block:: c++

    bool check_numeric(const std::string path,
                       const conduit::Node &params,
                       conduit::Node &info,
                       bool required);

* ``path``: the expected path to the parameter in the Conduit node
* ``params``: the parameters pass into verify
* ``info``: the info node passed into verify
* ``required``: indication that the parameter is required or optional

These helper functions return ``false`` if the parameter check fails.

Surprises
+++++++++
A common user error is to set the path to a parameter in the wrong node path.
For example the filter expects a parameter ``field`` but the user
adds the path ``field_name``, the verification will fail and complain about a
missing parameter. In order to provide a better error message, we provide
a surprise parameter checking mechanism that reports unknown paths.
Lines 9-18 in :ref:`verify` shows how to use the surprise_check function to
declare a set of known parameters and check for the existence of surprises.
``surpise_check`` also provides a means to ignore certain paths, which enables
filters to perform hierarchical surprise checking.

Execute
-------

Using MPI Inside Ascent
-----------------------

VTK-h and Ascent both create two separate libraries for MPI and non-MPI (i.e., serial).
In order to maintain the same interface for both version of the library, ``MPI_Comm`` handles
are represented by integers and are converted to the MPI implementations underlying representation
by using the ``MPI_Comm_f2c`` function.

Code containing calls to MPI are protected by the define ``VTKH_PARALLEL`` and calls to MPI API calls
must be guarded inside the code. Here is an example.

.. code-block:: c++

    #ifdef VTKH_PARALLEL
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      int rank;
      MPI_Comm_rank(comm, &rank);
    #endif

.. note::
    ``vtkh::GetMPICommHandle()`` will throw an exception if called outside of a ``VTKH_PARALLEL``
    block.


