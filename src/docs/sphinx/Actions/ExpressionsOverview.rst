.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _ExpressionsOverview:

Expressions Overview
====================
Expressions are a powerful analysis tool that can both answer questions
about mesh data and calculate derived quatities. Ascent uses a
python-like domain specific language (DSL) to execute generic compuations.
Broadly speaking, the expressions system allows users to mix and match
two types of computation:

    - Queries: data summarization (e.g a max reduction via ``max(field('energy'))``)
    - Derived Fields (e.g. ``field('energy') + 1``)


.. _queries:

Queries
-------
Queries are expressions that perform some sort of data reduction from
problem sized data. The results of queries can be single values,
such as the maximum value of a field, or could be multi-valued, such
as a histogram.

Queries are executed via an action with two parameters: an expression
and a name.  Below is an example of a simple query.

.. code-block:: yaml

    -
      action: "add_queries"
      queries:
        q1:
          params:
            expression: "cycle()"
            name: "my_cycle"

In the above example, we are asking about the state of the simulation, but
A more useful query will ask questions of the mesh or variables on the mesh.
A common query is to find the maximum value of field over time.

.. code-block:: yaml

    -
      action: "add_queries"
      queries:
        q1:
          params:
            expression: "max(field('density'))"
            name: "max_density"

Combining Queries
^^^^^^^^^^^^^^^^^
Queries are executed in the same order they are declared, and since the result
of each query stored, each query can be thought of as an assignment statement in
a program, with one query building on the previous queries.

.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
         params:
           expression: "1+1"
           name: "two"
       q2:
         params:
           expression: "two + 1"
           name: "result"

In the above example, ``q1`` is evaluated and the result is stored in the
identifier ``two``.
In ``q2``, the identifier is referenced and the expression evaluates to ``3``.

Query Results
^^^^^^^^^^^^^
Every time a query is executed the results are stored and can be accessed
in three ways:

    - Inside expressions: values can be referenced by name by other expressions
    - Inside the simulation: query results can be programatically accessed
    - As a post process: results are stored inside a file called `ascent_session.yaml`

The session file provides a way to acccess and plot the results of queries.
Session files are commonly processed by python scripts.
Here is an example of a single time step of the max query:

.. code-block:: yaml

    max_density:
      10:
        type: "value_position"
        attrs:
          value:
            value: 1.99993896548026
            type: "double"
          position:
            value: [4.921875, 4.921875, 10.078125]
            type: "vector"
          element:
            rank: 4
            domain_index: 4
            index: 285284
            assoc: "element"
        time: 0.313751488924026

Query results are index by name and then cycle.
Additionally, the simulation time for each query is always available.
In this example, the max query returns both the value and position
of the element contaning the maximum value.

Query History
-------------
Since the results of queries are stored, we can access values from previous
executions.
The ``history`` function allows expressions to have a temporal component, which
is a powerful tool
when tracking simulation state and adaptively responding to user defined events.
The history function can be called on any existing query.

The history of a query can be indexed in two ways:

   - ``relative_index``: a positive value that indicates how far back in history
     to access. If the index exceeds the current history, the value is clamped
     to the last index. An index of 0 is equivalent to the current time value
     and an index of 1 is the value of the identifier on the last execution.
   - ``absolute_index``: the index of the value to access. 0 is the first query
     result.

Here is an example of a use case for the history function:

.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
         params:
           # get the maximum value of a field
           expression: "max(field('pressure'))"
           name: "max_pressure"
       q2:
         params:
           expression: "max_pressure - history(max_pressure, relative_index = 1)
           > 100"
           name: "result"

In the above example, `q2` will evaluate to true if the maximum value of
pressure jumps over 100 units since the last in invocation, possibly indicating
that an interesting event inside the simulation occurred.

Derived Fields
--------------
Derived fields allow users to create new fields on the mesh as a
result of some arbitrary computation. A simple example of a derived field
is calculating mass based on cell volume and density, when mass is not
directly available. Once created, the new field can be manipuated via
filters or plotted

Derived expressions are just-in-time(JIT) compiled at runtime. That is,
code is generatated on the fly and the compiler is invoked during to create
the binary code that is executed. While this can be expensived the first time
the expression is run, the binary is cached and the cost is amortized over the
entire simulation run. Additionally, the binary is cached in Ascent's default
directory (which defaults to the current working directory), so the compile
time cost can also be amortized over multiple simulation invocations.
Supported backends include serial, OpenMP, and CUDA.

Derived generation is triggered by using either the `field` function used
in conjuction with math operations or the `topo` function.
The expressions filter provides a way to create a derived field
that is mapped back onto the mesh. Since derived fields transfrom data,
expressions filters are part of pipeline in Ascent. Here is a examle
of creating a simple derived field on the mesh:

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        pl1:
          f1:
            type: "expression"
            params:
              expression: "field('density') + 1"
              name: "density_plus_1"

Subsequent pipeline filters will have access the the variable
`density_plus_1`.

The `topo` function allows users to access information about the mesh topologies.
There are several topological attributes accessable through the `topo` function
including area (if 2d) and volume (if 3d). Here is an example of creating a
new field on the mesh that has the volume of each cell:

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        pl1:
          f1:
            type: "expression"
            params:
              expression: "topo('mesh').cell.volume"
              name: "cell_volume"

Using both fields and topological information inside a derived field can help
calculate quantities such as mass:

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        pl1:
          f1:
            type: "expression"
            params:
              expression: "topo('mesh').cell.volume * field('density')"
              name: "mass"


Combining Queries and Derived Fields
------------------------------------
Queries and derived fields can be used together. For example,
if we want to keep track of the total mesh volume over time

.. code-block:: yaml

    -
      action: "add_queries"
      queries:
        q1:
          params:
            expression: "sum(topo('mesh').cell.volume)"
            name: "total_volume"

In queries, the only restriction is that the result must be
a single value or object (i.e., a data reduction) so it can
be stored for access. However, there is no restriction on the results
of expressions filters and they can be either derived fields or queires.

Queries on Pipeline Results
---------------------------
Normally, queries execute on the mesh published to Ascent by the simulation,
but queries can also consume the results of pipelines.
The example below demonstrates the use of an expression to find the total area
of a contour mesh which is output by a pipeline.

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        pl1:
          f1:
            type: "contour"
            params:
              field: "energy"
              levels: 3
    -
      action: "add_queries"
      queries:
        q1:
          pipeline: "pl1"
          params:
            expression: "sum(topo('mesh').cell.area)"
            name: "total_iso_area"

Queries like the one above will act on the data published to Ascent. Queries
are also capable of acting on the results of pipelines.

Using Queries in Filter Parameters
----------------------------------
When running in situ, its often the case that you know what you are interested
in (e.g., I want to see the top 10% of density), but not know exactly what
the value range is. To help with that, Ascent can use expressions within filter
parameters. The following example creates an isovolume of the top 10% of density.

.. code-block:: yaml

    -
      action: "add_queries"
      queries:
        q1:
          params:
            expression: "max(field('density')).value"
            name: "max_density"
        q2:
          params:
            expression: "max_density - min(field('density')).value"
            name: "d_length"
    -
      action: "add_pipelines"
      pipelines:
        pl1:
          f1:
            type: "isovolume"
            params:
              field: "density"
              min_value: "max_density - 0.1 * d_length"
              max_value: "max_density"

Note: not all filter parameters support using expressions.


Derived Fields
^^^^^^^^^^^^^^
Derived field expressions can be used to create new fields on the mesh.

Functions, Objects, and Binary Operations are all capable of returning derived
fields.

The language's type system will determine when an expression needs to generate
a derived field for a Binary Operation. For example, if the expression is
``field('density') + 1`` the language will see that a scalar value is being
added to a field so the result must be a derived field where ``1`` is added to
each element of ``field('density')``.

Certain functions and object attributes will also generate derived fields. The
return type of such functions (in :ref:`ExpressionFunctions`)  and
objects (in :ref:`ExpresssionsObjects`) is ``jitable``. For example, there
is an overload of the ``max`` function with a return type of ``jitable``
which can be used to take the maximum of two fields via
``max(field('energy1'), field('energy2'))``.

.. note::
    For an expression like ``max(field('energy1'), field('energy2'))`` to work,
    the fields ``field('energy1')`` and ``field('energy2')`` must have the same
    number of elements.

Objects such as the ``topo`` object have attributes of type ``jitable`` which
will also generate derived fields. For example, a field of cell volumes for a
certain topology can be generated via ``topo('mesh').cell.volume``.

Derived fields or ``jitable`` types can be used in place of a regular field. For
example, we can combine a derived field and a reduction by writing
``sum(topo('mesh').cell.volume)`` to find the total volume of the mesh.

The if-then-else construct can also be used to generate derived fields when one
or more its operands (i.e. condition, if-branch, else-branch) are fields. If we
wanted to zero-out energy values below a certain threshold we can write
``if(field('energy') < 10) then 0 else field('energy')``.

Expressions that output a derived field will result in a mesh field with the same
name as the expression (see more about names below) which can be retrieved in
later expressions via ``field('name')``.

.. note::
   For performance reasons, derived expressions dealing with vectors should
   prefer using ``field('velocity', 'x')`` over ``field('velocity').x`` to get
   a component. Using ``.x`` will be necessary in the case that the field is a
   derived field. See the :ref:`Curl Example`.

Function Examples
~~~~~~~~~~~~~~~~~
   - ``cycle()``: returns the current simulation cycle
   - ``field('braid')``: returns a field object for the simulation field
     specified
   - ``histogram(field("braid"), num_bins=128)``: returns a histogram of the
     ``braid`` field
   - ``entropy(histogram(field("braid"), num_bins=128))``: returns the entropy
     of the histogram of the ``braid`` field
   - ``curl(field('velocity'))``: generates a derived vector field which is
     the curl of the ``velocity`` field (i.e. the vorticity)
   - ``curl(field('velocity'))``: generates a derived vector field which is
     the curl of the ``velocity`` field (i.e. the vorticity)


Assignments
^^^^^^^^^^^
As expressions start to become more complex the user may wish to store certain
temporary values. The language allows the use of 0 or more assignments
preceding the final expression.

Here is an expression which takes advantage of assignments to calculate the
curl (which is also a builtin function) using the gradient.

.. _Curl Example:
.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
        params:
          expression: |
                  du = gradient(field('velocity', 'u'))
                  dv = gradient(field('velocity', 'v'))
                  dw = gradient(field('velocity', 'w'))
                  w_x = dw.y - dv.z
                  w_y = du.z - dw.x
                  w_z = dv.x - du.y
                  vector(w_x, w_y, w_z)
          name: "velocity_vorticity"

The assignments and the final expression must be separated by newlines or
semicolons or both. The above example shows newline separation using multi-line
strings in YAML. Backslashes (``\``) may be used at the end of a line to split up long lines. Lines can also be split without the need for a backslash if there are unclosed parenthesis or brackets.

When resolving identifiers, the language will give precedence to identifiers
defined in the same expression (as shown in this example) before falling back
to the names of previously defined expressions (see below).



The Name
--------
The result of the expression is `stored` internally and can be accessed in two
ways.

 - Through the ``ascent.Info(info_node)`` call. This can be used as a way to
   feed information back the simulation.
 - As an identifier in a subsequent expression.

Combining Queries
^^^^^^^^^^^^^^^^^
Queries are executed in the same order they are declared, and since the result
of each query stored, each query can be thought of as an assignment statement in
a program, with one query building on the previous queries.

.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
         params:
           expression: "1+1"
           name: "two"
       q2:
         params:
           expression: "two + 1"
           name: "result"

In the above example, ``q1`` is evaluated and the result is stored in the
identifier ``two``.
In ``q2``, the identifier is referenced and the expression evaluates to ``3``.

Query History
-------------
Since the results of queries are stored, we can access values from previous
executions.
The ``history`` function allows expressions to have a temporal component, which
is a powerful tool
when tracking simulation state and adaptively responding to user defined events.
The history function can be called on any existing query.

The history of a query can be indexed in two ways:

   - ``relative_index``: a positive value that indicates how far back in history
     to access. If the index exceeds the current history, the value is clamped
     to the last index. An index of 0 is equivalent to the current time value
     and an index of 1 is the value of the identifier on the last execution.
   - ``absolute_index``: the index of the value to access. 0 is the first query
     result.

Here is an example of a use case for the history function:

.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
         params:
           # get the maximum value of a field
           expression: "max(field('pressure'))"
           name: "max_pressure"
       q2:
         params:
           expression: "max_pressure - history(max_pressure, relative_index = 1)
           > 100"
           name: "result"

In the above example, `q2` will evaluate to true if the maximum value of
pressure jumps over 100 units since the last in invocation, possibly indicating
that an interesting event inside the simulation occurred.

Session File
------------
Ascent saves the results of all queries into a file called `ascent_session.yaml`
when the simulation exits. This file is convenient for creating plotting scripts
that consume the results of queries. The session file is capable of surviving
simulation restarts, and it will continue adding to the file from the last time.
If the restart occurs at a cycle in the past (i.e., if the session was saved at cycle
200 and the simulation was restarted at cycle 150), all newer entries will be removed.

Default Session Name
^^^^^^^^^^^^^^^^^^^^
The default session file name is `ascent_session`, and you can change the session
file name with an entry in the `ascent_options.yaml` file.

.. code-block:: yaml

   session_name : my_session_name


If the simulation crashes, there is no promise that the session file will successfully
written out, so Ascent provides an explicit action to save the session file. Its
important to note that this involves IO, so its a good idea to only use this actions
periodically.

Save Session Action
^^^^^^^^^^^^^^^^^^^
If the simulation crashes, there is no promise that the session file will successfully
written out, so Ascent provides an explicit action to save the session file. Its
important to note that this involves IO, so its a good idea to only use this actions
periodically. The save session action always executes after all other actions have finished.

.. code-block:: yaml

   -
     action: "save_session"


Additionally, you can explicitly override the default session name by using the
`file_name` parameter:

.. code-block:: yaml

   -
     action: "save_session"
     file_name: "saved_by_name_with_selection"


Finally, you can save only a subset of the expressions using a list:

.. code-block:: yaml

  -
    action: "add_queries"
    queries:
      q1:
        params:
          expression: "max(field('p'))"
          name: "max_pressure"
      q2:
        params:
          expression: "10 + 1"
          name: "garbage"
  -
    action: "save_session"
    file_name: "saved_by_name_with_selection"
    expression_list:
      - max_pressure

In this example, there are two queries. The save session action specifies that only
the `max_pressure` expression should be saved inside the file named
`saved_by_name_with_selection`.
.. code-block:: yaml

   -
     action: "save_session"
