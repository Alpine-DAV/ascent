.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _tutorial_queries:

Asking and answering questions with Queries
---------------------------------------------

Queries are a way to ask summarization questions about meshes. Queries results can be used with Triggers to adapt analysis and visualization actions.  This section shows how to execute queries with Ascent and access query results. See :ref:`queries` docs for deeper details on Queries.


Extracting mesh cycle and entropy of a time varying mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_query_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_query_example1.cpp
   :language: cpp
   :lines: 13-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_query_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_query_example1.py
   :language: python
   :lines: 7-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_query_example1.txt


