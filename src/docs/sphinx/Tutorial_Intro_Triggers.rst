.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _tutorial_triggers:

Adapting visualization with Triggers
--------------------------------------

Triggers allow the user to specify a set of actions that are triggered by the result of a boolean expression.
They provide flexibility to adapt what analysis and visualization actions are taken in situ. Triggers leverage Ascent's Query and Expression infrastructure. See Ascent's :ref:`triggers` docs for deeper details on Triggers.

Using triggers to render when events occur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_trigger_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_trigger_example1.cpp
   :language: cpp
   :lines: 50-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_trigger_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_trigger_example1.py
   :language: python
   :lines: 45-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_trigger_example1.txt
