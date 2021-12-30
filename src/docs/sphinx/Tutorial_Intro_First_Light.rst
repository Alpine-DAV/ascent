.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _tutorial_first_light:

First Light
--------------

Render a sample dataset using Ascent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start, we run a basic "First Light" example to generate an image. This example renders the an example dataset using ray casting to create a pseudocolor plot. The dataset is one of the built-in Conduit Mesh Blueprint examples, in this case an unstructured mesh composed of hexagons.


:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/ascent_first_light_example.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_first_light_example.cpp
   :language: cpp
   :lines: 50-

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/ascent_first_light_example.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_first_light_example.py
   :language: python
   :lines: 53-

..  figure:: Tutorial_Output/out_first_light_render_3d.png
    :scale: 50 %
    :align: center

    First Light Example Result

