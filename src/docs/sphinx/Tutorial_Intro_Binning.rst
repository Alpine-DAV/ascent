..  Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
..  Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
..  other details. No copyright assignment is required to contribute to Ascent.

.. _tutorial_binning:

Data Binning 
--------------------------------------

Ascent provides a multi-dimensional data binning capability that allows you
 o calculate spatial distributions, find extreme values, etc. With the right
approach, you can implement mesh agnostic analysis that can be used across
simulation codes. You can also map the binned result back onto the original
mesh topologyto enable further analysis, like deviations from an average.
These examples show how to define and execute binning operations using Ascent's
query interface. See Ascent's :ref:`Binning` docs for deeper details about Data Binning.

Binning data spatially
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_binning_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_binning_example1.cpp
   :language: cpp
   :lines: 50-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_binning_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_binning_example1.py
   :language: python
   :lines: 4-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_binning_example1.txt
