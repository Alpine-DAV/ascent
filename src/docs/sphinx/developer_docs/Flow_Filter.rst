.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _flow_filter:

Flow Filter Anatomy
===================
Flow filters are the basic unit of execution inside of Ascent, and all functionality
is implemented as a Flow filter. The full interface to a Flow filter can be found in the
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
as well. ``verify_params`` alerts users to input errors and unexpected parameters.

.. note::

    Developing a flow filter requires a working knowledge of the Conduit API.
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
* ``port_names``: declares a list of input port names.
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
Parameters are passed through Ascent and then to filters. For detailed
examples of filter in Ascent see the :ref:`pipelines` section.


How Are Parameters Passed?
^^^^^^^^^^^^^^^^^^^^^^^^^^
The parameters are passed to the Ascent API through Conduit nodes. A simple filter
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
upon creation. Parameters are verified when the filter is created during execution.

Filter Parameter Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``verify_params`` method allow the filter creator to verify the expected parameters
and parameter types before the filter is executed. If the verification fails, error messages
are shown to the user. The method has two parameters: a Conduit node holding the parameters
of the filter and a Conduit node that is populated with error information that flow will
show if the result of the verification is false (error state).

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
* ``params``: the parameters passed into verify
* ``info``: the info node passed into verify
* ``required``: indication that the parameter is required or optional

These helper functions return ``false`` if the parameter check fails.

Surprises
+++++++++
A common user error is to set a parameter at the wrong path.
For example the filter expects a parameter ``field`` but the user
adds the path ``field_name``, the verification will fail and complain about a
missing parameter. In order to provide a better error message, we provide
a surprise parameter checking mechanism that reports unknown paths.
Lines 9-18 in :ref:`verify` show how to use the surprise_check function to
declare a set of known parameters and check for the existence of surprises.
``surpise_check`` also allows you to ignore certain paths, which enables
hierarchical surprise checking.

Execute
"""""""
The `execute()` method does the real work. In our example, we are wrapping the
``VTKHNoOp`` filter which is a `transform`, i.e., a filter that can be called
inside of a pipeline. Be default, `transforms` are passed VTK-h data sets and
`extracts` are called with either Conduit Blueprint data sets (i.e., the data
published by the simulation) or VTK-h data sets, when the `extract` consumes
the result of a pipeline. The data type can be checked by the filter and converted
by one of Ascent's data adapters located in the ``src/ascent/runtimes`` directory.

.. code-block:: c++
    :caption: An example execute method
    :linenos:

    void
    VTKHNoOp::execute()
    {

        if(!input(0).check_type<vtkh::DataSet>())
        {
            ASCENT_ERROR("vtkh_no_op input must be a vtk-h dataset");
        }

        std::string field_name = params()["field"].as_string();

        vtkh::DataSet *data = input<vtkh::DataSet>(0);
        vtkh::NoOp noop;

        noop.SetInput(data);
        noop.SetField(field_name);

        noop.Update();

        vtkh::DataSet *noop_output = noop.GetOutput();

        set_output<vtkh::DataSet>(noop_output);
    }


Filter Inputs
^^^^^^^^^^^^^

Inputs to filters are always pointers.
Lines 5-8 demonstrate how to check the type of data to the filter.
``input(0).check_type<SomeType>()`` returns true if the input pointer
is of the same type as the template parameter. Alternatively, we could
reference the input port by its declared interface name:
``input("in").check_type<SomeType>()``.

.. warning::
    If you perform input data type conversion, the temporary converted
    data must be deleted before exiting the execute method.

Once the filter input type is known it is safe to call ``input<KnownType>(0)``
to retrieve the pointer to the input (line 12).

Flow filters have a member function ``params()`` that returns a reference
to the Conduit node containing the filter parameters that were previously
verified. Since we already verified the existence of the string parameter
``field``, it is safe to grab that parameter without checking the type or
path.


For optional parameters, care should be used when accessing node paths.
Conduit nodes paths can be checked with ``params().has_path("some_path")``
Other methods exist to verify or convert their underlying types such as
``node["path"].is_numeric()``. If you are expecting an integer the semantics
between these two calls are very different:

* ``node["path"].as_int32()``: I am positive this is an int32 and I alone
  accept the consequences if it is not
* ``node["path"].to_int32()``: I am expecting an int32 and please convert if for me
  assuming whatever type it is can be converted to what I am expecting

Filter Output
^^^^^^^^^^^^^
A filter's output is a pointer to a data sets. In the case of `tranforms` this type is
expected to be a VTK-h data set. Output pointers are reference counted by Flow's registry
and will be deleted when no downstream filter needs the output of the current filter.

In the case of an `extract`, no output needs to be set.

Registering Filters With Ascent
"""""""""""""""""""""""""""""""
Newly created filters need to be registered with the Ascent runtime.
The file
`ascent_runtime_filters.cpp <https://github.com/Alpine-DAV/ascent/blob/develop/src/ascent/runtimes/flow_filters/ascent_runtime_filters.cpp>`_
is where all builtin filter are registered. Following the NoOp example:

.. code-block:: c++
    :caption: Ascent runtime filter registration

    AscentRuntime::register_filter_type<VTKHNoOp>("transforms","noop");

Filter registration is templated on the filter type and takes two arguments.

* arg1: the type of the fitler. Valid values are ``transforms`` and ``extracts``
* arg2: the front-facing API name of the filter. This is what a user would declare in an actions file.

Accessing Metadata
------------------
We currently populate a limited set of metadata that is accessable to flow filters.
We place a Conduit node containing the metadata inside the registry which can be
accessed in the following manner:

.. code-block:: c++
    :caption: Accessing the regsitry metadata inside a flow filter

    conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");
    int cycle = -1;
    float time = -1.f;
    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].to_int32();
    }
    if(meta->has_path("time"))
    {
       time = (*meta)["time"].to_int32();
    }

The above code is conservative, checking to see if the paths exist. The current metadata values
Ascent populates are:

* cycle: simulation cycle
* time: simulation time
* refinement_level: number of times a high-order mesh is refined

If these values are not provided by the simulation, then defaults are used.

Using the Registry (state)
--------------------------
Filters are created and destroyed every time the graph is executed. Filters might
want to keep state associated with a particular execution of the filter. A conduit node
is a convenient container for arbitrary data, but there is no restriction on the type
of data that can go inside the registry.

.. code-block:: c++
    :caption: Accessing the registry metadata inside a flow filter

    conduit::Node *my_state_data = new conduit::Node();
    // insert some data to the node

    // adding the  node to the registry
    graph().workspace().registry().add<conduit::Node>("my_state", my_state_data, 1);

    // check for existence and retrieve
    if(graph().workspace().registry().has_entry("my_state"))
    {
      conduit::Node *data = graph().workspace().registry().fetch<conduit::Node>("my_state"))
      // do more stuff
    }

Data kept in the registry will be destroyed when Ascent is torn down, but will persist otherwise.
A problem that arises is how to tell different invocations of the same filter apart, since
a filter can be called an arbitrary number of times every time ascent is executed. The Ascent
runtime gives unique names to filters that can be accessed by a filter member function
``this->detailed_name()``. One possible solution is to use this name to differentiate
filter invocations. This approach is reasonable if the actions remain the same throughout
the simulation, but if they might change, all bets are o ff.

.. note::
    Advanced support of registry and workspace usage is only supported through
    the Ascent developers platinum support contract, which can be purchased with
    baby unicorn tears. Alternatively, you are encouraged to look at the flow
    source code, unit tests, and ask questions.

Using MPI Inside Ascent
-----------------------

Ascent creates two separate libraries for MPI and non-MPI (i.e., serial).
In order to maintain the same interface for both versions of the library, ``MPI_Comm`` handles
are represented by integers and are converted to the MPI implementations underlying representation
by using the ``MPI_Comm_f2c`` function.

Code containing calls to MPI are protected by the define ``ASCENT_MPI_ENABLED`` and calls to MPI API calls
must be guarded inside the code. In Ascent, the MPI comm handle is stored in and can be
retrieved from the ``flow::Workspace`` which is accessible from inside a flow filter.

.. code-block:: c++
    :caption: Example of code inside a filter that retrieves the MPI comm handle from the workspace

    #ifdef ASCENT_MPI_ENABLED
      int comm_id = flow::Workspace::default_mpi_comm();
      MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
      int rank;
      MPI_Comm_rank(comm, &rank);
    #endif


