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

.. _Expressions:

Expressions
===========
Expressions are a way to ask questions and get answers. The output of an
expression can be categorized into one of the following categories.

    - Single-Value (e.g a sum reduction via ``sum(field('energy'))``)
    - Multi-Value (e.g. a data binning via ``binning('braid','max', [axis('x',
      num_bins=10)])``)
    - Derived Fields (e.g. ``field('energy') + 1``)

Each expression has two required parameters: an expression and a name.

Below is an example of a simple query.

.. code-block:: yaml

    -
      action: "add_queries"
      queries:
        q1:
          params:
            expression: "cycle()"
            name: "my_cycle"

Queries like the one above will act on the data published to Ascent. Queries
are also capable of acting on the results of pipelines.

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


Finally queries can themselves be part of a pipeline via the "expression"
filter. The example below creates a new vector field on the mesh using a
pipeline. The result of the pipeline then extracted to an HDF5 file.

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        pl1:
          f1:
            type: "expression"
            params:
              expression: "curl(field('velocity'))"
              name: "velocity_vorticity"
    -
      action: "add_extracts"
      extracts:
        e1:
          pipeline: "pl1"
          type: "relay"
          params:
            path: "vorticity_out"
            protocol: "blueprint/mesh/hdf5"


The results of queries can be accessed by the simulation and serve as a way to
compose
complex triggers, i.e., taking actions as a result of a condition.
Query support is in beta, meaning its a part of Ascent currently under
development.

Expressions Overview
--------------------
Expressions are based on a simple python-like language that supports math
evaluation and function calls.

Basic Data Types
^^^^^^^^^^^^^^^^
There are three main integral types in the language:

   - int: ``0``, ``-1``, ``1000``
   - double: ``1.0``, ``-1.0``, ``2.13e10``
   - string: ``'this is a string'``
   - bool: ``True``, ``False``

Operators
^^^^^^^^^
The supported math operators follow the standard operator precedence order:

   - ``()``: grouping
   - ``f(args...)``: function call
   - ``a.attribute``: attribute reference
   - ``-``: negation
   - ``*, /, %``: multiplication, division, modulus (remainder)
   - ``+, -``: addition, subtraction
   - ``not, or, and, <, <=, >, >=, !=, ==``: comparisons

Control Flow
^^^^^^^^^^^^
The expression language currently supports simple if-then-else semantics.

.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
         params:
           expression: "if cycle() > 100 then 1 else 0"
           name: "cycle_bigger_than_100"

.. code-block:: yaml

   -
     action: "add_queries"
     queries:
       q1:
         params:
           expression: |
                   energy_max = max(field('energy'))
                   if energy_max.value > 10 then \
                        energy_max.position \
                   else \
                        vector(0, 0, 0)
           name: "max_energy_position_above_10"

.. code-block:: yaml

     action: "add_queries"
     queries:
       q1:
         params:
           expression: |
             accept_prob = binning('cnt', 'cdf', [axis(field('braid'), num_bins=10)])
             if(binning_value(accept_prob) < rand()) then 0 else field('braid')
           name: "cdf_random_sampling"

.. note::
   Both branches of the if-then-else will be execute.

Functions
^^^^^^^^^
The expression system supports functions with both required and optional (named)
parameters.
Functions can both accept and return complex objects like histograms or arrays.
Additionally, function overloading is supported and the overload type is
resolved by the function parameters.

Objects
^^^^^^^
Objects have attributes that can be accessed using the ``.`` operator. For
example, the ``value_position`` object returned by the ``min`` function has two
attributes, the ``value`` which holds a double containing the minimum value and
the ``position`` which holds a vector describing the position of the minimum
value on the mesh. They would be accessed via ``min(field('braid')).value`` and
``min(field('braid')).position`` respectively.

The attributes of the various objects are specified in the :ref:`Ascent
Objects Documentation` section.

.. note::
   Not all attributes listed in the documentation are available at runtime. For
   example, the ``topo`` object has attributes ``x``, ``y``, and ``z``, however
   at runtime something like ``topo('mesh').z`` may throw an error if the
   topology named 'mesh' has only 2 dimensions.

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
return type of such functions and objects is ``jitable`` in the :ref:`Ascent
Expressions Documentation`. For example, there is an overload of the ``max``
function with a return type of ``jitable`` which can be used to take the
maximum of two fields via ``max(field('energy1'), field('energy2'))``.

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

The if-then-else construct can also be used to generate derived fields. If we
wanted to zero-out energy values below a certain threshold we can write
``if(field('energy') < 10) then 0 else field('energy')``.

Expressions that output a derived field will result in a mesh field with the same
name as the expression (see more about names below) which can be retrieved in
later expressions via ``field('name')``.


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
