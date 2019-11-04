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

.. _queries:

Queries
========
Queries are a way to ask questions and get answers.
Each query has to required parameters: an expression and a name.

Below is an example of a simple query.

.. code-block:: yaml

   actions:
     -
       action: "add_queries"
       queries:
         q1:
           params:
             expression: "cycle()"
             name: "my_cycle"

.. note::
    Queries currently only support inspecting the data published to Ascent, and queries
    on the results of pipelines is **not** currently supported..


The results of queries can be accessed by the simulation and serve as a way to compose
complex triggers, i.e., taking actions as a result of a condition.
Query support is in beta, meaning its a part of Ascent currently under development.

Expressions Overview
--------------------
Expressions are based on a simple python-like language that supports math evaluation and function calls.

Basic Data Types
^^^^^^^^^^^^^^^^
There are three main integral types in the language:

   - integers: ``0``, ``-1``, ``1000``
   - floating point: ``1.0``, ``-1.0``, ``2.13e10``
   - strings: ``'this is a string'``
   - boolean: only created as the result of a comparison

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

   actions:
     -
       action: "add_queries"
       queries:
         q1:
           params:
             expression: "if cycle() > 100 then 1 else 0"
             name: "cycle_bigger_than_100"

.. note::
   Both branches of the if-then-else will be execute.

Functions
^^^^^^^^^
The expression system supports functions both required and optional (named) parameters.
Functions can both accept and return complex objects like histograms or arrays.
Additionally, function overloading is supported and the overload type is resolved by the
function parameters.


Function Examples
~~~~~~~~~~~~~~~~~
   - ``cycle()``: returns the current simulation cycle
   - ``field("braid")``: returns a field object for the simulation field specified
   - ``histogram(field("braid"), num_bins=128)``: returns a histogram of the ``braid`` field
   - ``entropy(histogram(field("braid"), num_bins=128))``: returns a the entropy of the histogram of the ``braid`` field

The Name
--------
The result of the expression is `stored` internally and can be accessed in two ways.

 - Through the ``ascent.Info(info_node)`` call. This can be used as a way to feed information back the simulation.
 - As an identifier in a subsequent expression.

Combining Queries
^^^^^^^^^^^^^^^^^
Queries are executed in the same order they are declared, and since the result of each query stored,
each query can be thought of as an assignment statement in a program, with one query building on the previous queries.

.. code-block:: yaml

   actions:
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

In the above example, ``q1`` is evaluated and the result is stored in the identifier ``two``.
In ``q2``, the identifier is referenced and the expression evaluates to ``3``.

Query History
-------------
Since the results of queries are stored, we can access values from previous executions.
The ``history`` function allows expressions to have a temporal component, which is a powerful tool
when tracking simulation state and adaptively responding to user defined events.
The history function can be called on any existing query.

The history of a query can be indexed in two ways:

   - ``relative_index``: a positive value that indicates how far back in history to access. If the index exceeds the current history, the value is clamped to the last index. An index of 0 is equivalent to the current time value and and index of 1 is the value of the identifier on the last execution.
   - ``absolute_index``: the index of the value to access. 0 is the first query result.

Here is an example of a use case for the history function:

.. code-block:: yaml

   actions:
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
             expression: "max_pressure - history(max_pressure, relative_index = 1) > 100"
             name: "result"

In the above example, `q2` will evaluate to true if the maximum value of pressure jumps over 100 units
since the last in invocation, possibly indicating that an interesting event inside the simulation occurred.
