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
`Flow filter header file <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/flow/flow_filter.hpp>`_.
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
    In the :ref:`tutorial_intro` section under ``Conduit Examples``, there are several
    examples of basic Conduit usage. More Conduit tutorial resources can be found in the
    `Conduit documentation <https://llnl-conduit.readthedocs.io/en/latest/tutorial_cpp.html>`_.

Flow filter implementations are located in the ``src/libs/ascent/runtimes/flow_filters`` directory.

Implementing A New Filter
-------------------------
As a convenience, we have created the
`VTKHNoOp <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_vtkh_filters.cpp>`_
filter as staring point and reference. Although the NoOp filter demonstrates how to use a
VTK-h filter, the implementation is relevant to anyone developing flow filters in Ascent
regardless of whether VTK-h or Viskores is used.

Interface Declaration
"""""""""""""""""""""
.. code-block:: c++

    void
    VTKHNoOp::declare_interface(conduit::Node &i)
    {
        i["type_name"]   = "vtkh_no_op";
        i["port_names"].append() = "in";
        i["output_port"] = "true";
        i["param_schema"] = conduit::Node();
    }


* ``type_name``: declares the name of the filter to flow, and the only requirement is that this name be unique.
* ``port_names``: declares a list of input port names.
* ``output_port``: declares if this filter has an output of not. Valid values are ``true`` and ``false``.
* ``param_schema``: defines a parameter schema as a conduit node expressing the required and optional parameters and their types.

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


or equivalently in yaml:

.. code-block:: yaml

  type: "filter_name"
  params:
    string_param: "string"
    double_param: 2.0


The Ascent runtime looks for the ``params`` node and passes it to the filter
upon creation. Parameters are verified when the filter is created during execution.

Filter Parameter Verification Schemas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter parameters are validated and checked for any surprises using :ref:`param_schema`.
In practice, a filter's interface ``param_schema`` describes the expected structure of the ``params``
node that users provide when creating a filter.

.. note::
    In prior versions of Ascent, the ``verify_params`` method was used to allow the filter creator
    to define parameter verification and surprise checking for each individual flow filter.
    The base implementation of ``verify_params`` now uses the ``param_schema`` defined in the filter interface
    to validate parameters and check for surprises against the expected schema
    using :ref:`param_schema`.

Ascent Parameter Schema Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To assist in constructing parameter schemas, Ascent has a number of  small schema
builder helpers defined in:

- ``src/libs/ascent/runtimes/flow_filters/ascent_runtime_param_check.hpp``
- ``src/libs/ascent/runtimes/flow_filters/ascent_runtime_param_check.cpp``

Each helper returns a Conduit node that represents a schema fragment (see :ref:`param_schema`).
These fragments are typically inserted under ``param_schema["properties/<param_name>"]``.

Example pattern:

.. code-block:: c++

    using namespace ascent::runtime::filters;

    void MyFilter::declare_interface(conduit::Node &i)
    {
        i["type_name"] = "my_filter";
        i["port_names"].append() = "in";
        i["output_port"] = "true";

        conduit::Node &param_schema = i["param_schema"];
        param_schema["type"] = "object";
        param_schema["additionalProperties"] = false;

        string_schema(param_schema["properties/field"], 1); // non-empty string
        param_schema["required"].append() = "field";
    }

string_schema
#############

Builds a string schema with optional length bounds.

.. code-block:: c++

    conduit::Node &string_schema(conduit::Node &schema_node,
                                 std::size_t minLength = 0,
                                 std::size_t maxLength = std::numeric_limits<std::size_t>::max());

.. raw:: html

  <details><summary><strong>Example: Using ``string_schema`` (with length bounds)</strong></summary>

.. code-block:: c++

    // Example: require a non-empty string with a max length
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"], 1, 64);
    param_schema["required"].append() = "field";

Resulting schema fragment:

.. code-block:: yaml

  type: string
  minLength: 1
  maxLength: 64

Example input accepted by this schema:

.. code-block:: yaml

  params:
    field: "pressure"

.. raw:: html
  
  </details>
  <br>


string_enum_schema
##################

Builds a string schema restricted to an enumerated set of allowed values.

.. code-block:: c++

    conduit::Node &string_enum_schema(conduit::Node &schema_node, 
                                      const std::vector<std::string> &options);

.. raw:: html

  <details><summary><strong>Example: Using ``string_enum_schema`` (string enum)</strong></summary>

.. code-block:: c++

    // Example: restrict a string parameter to a fixed set of values
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_enum_schema(param_schema["properties/interpolation"], {"nearest", "linear"});
    param_schema["required"].append() = "interpolation";

Resulting schema fragment:

.. code-block:: yaml

  type: string
  enum: ["nearest", "linear"]

Example input accepted by this schema:

.. code-block:: yaml

  params:
    interpolation: "nearest"

.. raw:: html

  </details>
  <br>

bool_schema
###########

Builds a boolean-like schema represented as a *string* enum: ``"true"`` or ``"false"``.

.. code-block:: c++

    conduit::Node &bool_schema(conduit::Node &schema_node);

.. raw:: html

  <details><summary><strong>Example: Using ``bool_schema`` ("true"/"false" string enum)</strong></summary>

.. code-block:: c++

    // Example: model a boolean-like parameter as "true"/"false" strings
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    bool_schema(param_schema["properties/enabled"]);
    param_schema["required"].append() = "enabled";

Resulting schema fragment:

.. code-block:: yaml

  type: string
  enum: ["true", "false"]

Example input accepted by this schema:

.. code-block:: yaml

  # Note: the values must be strings, so quote them in YAML.
  params:
    enabled: "true"

.. raw:: html

  </details>
  <br>

number_schema
#############

Builds a numeric schema, optionally with bounds. If ``supports_expressions`` is ``true``,
the schema accepts either a number *or* an expression string (via ``oneOf``).

.. code-block:: c++

    conduit::Node &number_schema(conduit::Node &schema_node,
                                 bool supports_expressions = false,
                                 int minimum = std::numeric_limits<int>::lowest(),
                                 int maximum = std::numeric_limits<int>::max(),
                                 int exclusiveMinimum = std::numeric_limits<int>::lowest(),
                                 int exclusiveMaximum = std::numeric_limits<int>::max());

.. raw:: html

  <details><summary><strong>Example: Using ``number_schema`` (bounded number)</strong></summary>

.. code-block:: c++

    // Example: accept a number in [0, 10]
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    number_schema(param_schema["properties/threshold"], false, 0, 10);
    param_schema["required"].append() = "threshold";

Resulting schema fragment:

.. code-block:: yaml

  # Example: bounded number (no expressions)
  type: number
  minimum: 0
  maximum: 10

Example input accepted by this schema:

.. code-block:: yaml

  params:
    threshold: 3.5

.. raw:: html

  </details>
  <br>

.. raw:: html

  <details><summary><strong>Example: Using ``number_schema`` (number or expression)</strong></summary>

.. code-block:: c++

    // Example: accept either a numeric value or an expression string
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    number_schema(param_schema["properties/threshold"], true, 0, 10);
    param_schema["required"].append() = "threshold";

Resulting schema fragment:

.. code-block:: yaml

  # Example: number OR expression string
  oneOf:
    - {type: number, minimum: 0, maximum: 10}
    - {type: string, format: expression}

Example input accepted by this schema:

.. code-block:: yaml

  params:
    threshold: "1 + 2"

.. raw:: html

  </details>
  <br>

integer_schema
##############

Builds an integer schema, optionally with bounds. If ``supports_expressions`` is ``true``,
the schema accepts either an integer *or* an expression string (via ``oneOf``).

.. code-block:: c++

    conduit::Node &integer_schema(conduit::Node &schema_node,
                                  bool supports_expressions = false,
                                  int minimum = std::numeric_limits<int>::lowest(),
                                  int maximum = std::numeric_limits<int>::max(),
                                  int exclusiveMinimum = std::numeric_limits<int>::lowest(),
                                  int exclusiveMaximum = std::numeric_limits<int>::max());

.. raw:: html

  <details><summary><strong>Example: Using ``integer_schema`` (bounded integer)</strong></summary>

.. code-block:: c++

    // Example: accept an integer >= 1
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    integer_schema(param_schema["properties/levels"], false, 1);
    param_schema["required"].append() = "levels";

Resulting schema fragment:

.. code-block:: yaml

  type: integer
  minimum: 1

Example input accepted by this schema:

.. code-block:: yaml

  params:
    levels: 8

.. raw:: html

  </details>
  <br>

.. raw:: html

  <details><summary><strong>Example: Using ``integer_schema`` (integer or expression)</strong></summary>

.. code-block:: c++

    // Example: accept either an integer value or an expression string
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    integer_schema(param_schema["properties/levels"], true, 1);
    param_schema["required"].append() = "levels";

Resulting schema fragment:

.. code-block:: yaml

  oneOf:
    - {type: integer, minimum: 1}
    - {type: string, format: expression}

Example input accepted by this schema:

.. code-block:: yaml

  params:
    levels: "2 + 2"

.. raw:: html

  </details>
  <br>

vec3_schema
###########

Builds an object schema that requires three numeric components. The default component
names are ``x``, ``y``, and ``z``.

.. code-block:: c++

    conduit::Node &vec3_schema(conduit::Node &schema_node,
                               bool supports_expressions = false);

    conduit::Node &vec3_schema(conduit::Node &schema_node,
                               const std::string var1,
                               const std::string var2,
                               const std::string var3,
                               bool supports_expressions = false);

.. raw:: html

  <details><summary><strong>Example: Using ``vec3_schema`` (required x/y/z object)</strong></summary>

.. code-block:: c++

    // Example: require all 3 components: x, y, z
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    vec3_schema(param_schema["properties/offset"]);
    param_schema["required"].append() = "offset";

Resulting schema fragment:

.. code-block:: yaml

  type: object
  additionalProperties: false
  required: ["x", "y", "z"]
  properties:
    x: {type: number}
    y: {type: number}
    z: {type: number}

Example input accepted by this schema:

.. code-block:: yaml

  params:
    offset: {x: 0.0, y: 1.0, z: 2.0}

.. raw:: html

  </details>
  <br>

.. raw:: html

  <details><summary><strong>Example: Using ``vec3_schema`` with custom param names</strong></summary>

.. code-block:: c++

    // Example: require all 3 components: x, y, z
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    vec3_schema(param_schema["properties/offset"], "var_i", "var_j", "var_k");
    param_schema["required"].append() = "offset";

Resulting schema fragment:

.. code-block:: yaml

  type: object
  additionalProperties: false
  required: ["var_i", "var_j", "var_k"]
  properties:
    var_i: {type: number}
    var_j: {type: number}
    var_k: {type: number}

Example input accepted by this schema:

.. code-block:: yaml

  params:
    offset: {var_i: 0.0, var_j: 1.0, var_k: 2.0}

.. raw:: html

  </details>
  <br>

vec3_schema_anyOf
#################

Builds an object schema that allows any subset of three numeric components, but requires
that *at least one* of them is present (via ``anyOf``). The default component names are
``x``, ``y``, and ``z``.

.. code-block:: c++

    conduit::Node &vec3_schema_anyOf(conduit::Node &schema_node,
                                     bool supports_expressions = false);

    conduit::Node &vec3_schema_anyOf(conduit::Node &schema_node,
                                     const std::string var1,
                                     const std::string var2,
                                     const std::string var3,
                                     bool supports_expressions = false);

.. raw:: html

  <details><summary><strong>Example: Using ``vec3_schema_anyOf`` (x/y/z object, at least one required)</strong></summary>

.. code-block:: c++

    // Example: allow x/y/z, but require at least one component to be present
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    vec3_schema_anyOf(param_schema["properties/origin"]);
    param_schema["required"].append() = "origin";

Resulting schema fragment:

.. code-block:: yaml

  type: object
  additionalProperties: false
  properties:
    x: {type: number}
    y: {type: number}
    z: {type: number}
  anyOf:
    - {type: object, required: ["x"]}
    - {type: object, required: ["y"]}
    - {type: object, required: ["z"]}

Example input accepted by this schema:

.. code-block:: yaml

  params:
    origin: {x: 0.0}  # or {y: 1.0} / {z: 2.0} / {x: 0.0, y: 1.0}, etc.

.. raw:: html

  </details>
  <br>

array_schema
############

Builds an array schema. With ``item_schema``, each element is validated against the
provided schema.

.. code-block:: c++

    conduit::Node &array_schema(conduit::Node &schema_node);

    conduit::Node &array_schema(conduit::Node &schema_node,
                                const conduit::Node &item_schema);

.. raw:: html

  <details><summary><strong>Example: Using ``array_schema`` (array of numbers)</strong></summary>

.. code-block:: c++

    // Example: require an array where each item is a number
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    array_schema(param_schema["properties/iso_values"], number_schema());
    param_schema["required"].append() = "iso_values";

Resulting schema fragment:

.. code-block:: yaml

  # Array of numbers
  type: array
  items: {type: number}

Example input accepted by this schema:

.. code-block:: yaml

  params:
    iso_values: [0.1, 0.2, 0.3]

.. raw:: html

  </details>
  <br>

ignore_schema
#############

Builds an object schema that is explicitly skipped during validation (via ``constraints/skip``).
This is useful when a parameter subtree is accepted but not yet formally specified, or when
validation is handled elsewhere.

.. code-block:: c++

    conduit::Node &ignore_schema(conduit::Node &schema_node);

.. raw:: html

  <details><summary><strong>Example: Using ``ignore_schema`` (skip validation for a subtree)</strong></summary>

.. code-block:: c++

    // Example: accept a subtree but skip validating it
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    ignore_schema(param_schema["properties/options"]);
    param_schema["required"].append() = "options";

Resulting schema fragment:

.. code-block:: yaml

  type: object
  constraints:
    skip: true

Example input accepted by this schema:

.. code-block:: yaml

  params:
    options:
      any_subtree: "is accepted"
      nested: {a: 1, b: [1, 2, 3]}

.. raw:: html

  </details>
  <br>

Execute
"""""""
The `execute()` method does the real work. In our example, we are wrapping the
``VTKHNoOp`` filter which is a `transform`, i.e., a filter that can be called
inside of a pipeline. Be default, `transforms` are passed VTK-h data sets and
`extracts` are called with either Conduit Blueprint data sets (i.e., the data
published by the simulation) or VTK-h data sets, when the `extract` consumes
the result of a pipeline. The data type can be checked by the filter and converted
by one of Ascent's data adapters located in the ``src/libs/ascent/runtimes`` directory.

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
`ascent_runtime_filters.cpp <https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/ascent/runtimes/flow_filters/ascent_runtime_filters.cpp>`_
is where all builtin filter are registered. Following the NoOp example:

.. code-block:: c++
    :caption: Ascent runtime filter registration

    AscentRuntime::register_filter_type<VTKHNoOp>("transforms","noop");

Filter registration is templated on the filter type and takes two arguments.

* arg1: the type of the filter. Valid values are ``transforms`` and ``extracts``
* arg2: the front-facing API name of the filter. This is what a user would declare in an actions file.

Accessing Metadata
------------------
We currently populate a limited set of metadata that is accessible to flow filters.
We place a Conduit node containing the metadata inside the registry which can be
accessed in the following manner:

.. code-block:: c++
    :caption: Accessing the registry metadata inside a flow filter

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
