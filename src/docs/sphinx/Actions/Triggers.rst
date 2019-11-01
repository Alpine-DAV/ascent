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


Triggers
========
Triggers allow the user to specify a set of actions that are `triggered` by
the result of a boolean expression. Triggers leverage the query and expression
system to enable a set of primitives to create custom tailored triggers for
simulations. There are several motivations that make triggers useful.
Examples include cost reduction (time) and debugging (correctness).

Cost Reduction
--------------
Visualization and analysis operation are not free, i.e., performing them takes time.
One example is that rendering an image every simulation cycle might be a relatively high cost.
Another example is performing a potentially time consuming operation like flow field analysis
may might need to take place at a lower frequency than each time ``ascent.execute(actions)`` is
called or may need to execute only when a specific simulation event occurs.

Debugging
---------
Triggers can be a useful tool in detecting anomalous conditions while a simulation is
running. For example, triggers can detect of the minimum or maximum values of a field
are outside the expected ranges and dump out the mesh and related field for inspection
inside of VisIt.

Simple Example
--------------
A trigger requires a trigger condition and a actions file:

.. code-block:: yaml

   actions:
     -
       action: "add_triggers"
       triggers:
         t1:
           params:
             condition: "cycle() % 100 == 0"
             actions_file : "my_trigger_actions.yaml"


In this example, the trigger will fire when the current cycle is divisible by 100.

Queries and Triggers
--------------------
Triggers can leverage the query system, and combining both queries and triggers
creates a rich set of potential options for triggers.


Debugging
^^^^^^^^^
The following code example shows how queries and triggers can be used for debugging:

.. code-block:: yaml

   actions:
     -
       action: "add_queries"
       queries:
         q1:
           params:
             # get the maximum value of a field
             expression: "max(field(\"pressure\"))"
             name: "max_pressure"
         q2:
           params:
             expression: "1e300"
             name: "crazy_value"
     -
       action: "add_triggers"
       triggers:
         t1:
           params:
             condition: "max_pressure > crazy_value"
             actions_file : "my_trigger_actions.yaml"

In the above example, the trigger will fire if the maximum value of pressure exceeds ``crazy_value``.
A useful actions file might dump the mesh to a hdf5 file for later inspection.

Event Detection
^^^^^^^^^^^^^^^
The following code example shows how queries and triggers can be used for to detect a jump in
entropy that could represent a simulation event that the user wants act on:

.. code-block:: yaml

   actions:
     -
       action: "add_queries"
       queries:
         q1:
           params:
             expression: "entropy(histogram(field(\"pressure\"))"
             name: "entropy"
     -
       action: "add_triggers"
       triggers:
         t1:
           params:
             condition: "history(entropy, relative_index = 1) - entropy > 10"
             actions_file : "my_trigger_actions.yaml"

In the above example, the trigger will fire if the change in entropy changes by more than
10 in positive direction.

