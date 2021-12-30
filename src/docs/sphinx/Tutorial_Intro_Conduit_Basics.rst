.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _tutorial_conduit_basics:

Conduit Basics
----------------

Ascent's API is based on `Conduit <http://software.llnl.gov/conduit/>`_. Both mesh data and action descriptions are passed to Ascent as Conduit trees.  The Conduit C++ and Python interfaces are very similar, with the C++ interface heavily influenced by the ease of use of Python. These examples provide basic knowledge about creating Conduit Nodes to use with Ascent. You can also find more introductory Conduit examples in Conduit's `Tutorial Docs <https://llnl-conduit.readthedocs.io/en/latest/conduit.html>`_ .

Creating key-value entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example1.cpp
   :language: cpp
   :lines: 50-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_cpp_conduit_example1.txt
   :language: yaml
   
:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example1.py
   :language: python
   :lines: 45-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_conduit_example1.txt
   :language: yaml


Creating a path hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example2.cpp
   :language: cpp
   :lines: 50-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_cpp_conduit_example2.txt
   :language: json

   
:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example2.py
   :language: python
   :lines: 45-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_conduit_example2.txt
   :language: yaml


Setting array data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example3.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example3.cpp
   :language: cpp
   :lines: 50-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_cpp_conduit_example3.txt
   :language: json

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example3.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example3.py
   :language: python
   :lines: 45-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_conduit_example3.txt
   :language: yaml


Zero-copy vs deep copy of array data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example4.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example4.cpp
   :language: cpp
   :lines: 50-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_cpp_conduit_example4.txt
   :language: json

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example4.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example4.py
   :language: python
   :lines: 45-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_conduit_example4.txt
   :language: yaml