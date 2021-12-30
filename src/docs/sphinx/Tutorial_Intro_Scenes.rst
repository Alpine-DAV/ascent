.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _tutorial_scenes:

Rendering images with Scenes
-------------------------------

Scenes are the construct used to render pictures of meshes in Ascent. A scene description encapsulates all the information required to generate one or more images. Scenes can render mesh data published to Ascent or the result of an Ascent Pipeline.  This section of the tutorial provides a few simple examples demonstrating how to describe and render scenes. See :ref:`scenes` docs for deeper details on Scenes.


Using multiple scenes to render different variables 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example1.cpp
   :language: cpp
   :lines: 13-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example1.py
   :language: python
   :lines: 7-

..  figure:: Tutorial_Output/out_scene_ex1_render_var1.png
    :scale: 50 %
    :align: center

    Render of tets var1 field

..  figure:: Tutorial_Output/out_scene_ex1_render_var2.png
    :scale: 50 %
    :align: center

    Render of tets var2 field


Rendering multiple plots to a single image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example2.cpp
   :language: cpp
   :lines: 13-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example2.py
   :language: python
   :lines: 7-
   
..  figure:: Tutorial_Output/out_scene_ex2_render_two_plots.png
    :scale: 50 %
    :align: center

    Render of two plots to a single image

Adjusting camera parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related docs: :ref:`scenes_renders`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example3.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example3.cpp
   :language: cpp
   :lines: 13-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example3.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example3.py
   :language: python
   :lines: 7-

..  figure:: Tutorial_Output/out_scene_ex3_view1.png
    :scale: 50 %
    :align: center

    Render with an azimuth change


..  figure:: Tutorial_Output/out_scene_ex3_view2.png
    :scale: 50 %
    :align: center

    Render with a zoom


Changing the color tables
~~~~~~~~~~~~~~~~~~~~~~~~~~

Related docs: :ref:`scenes_color_tables`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example4.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_scene_example4.cpp
   :language: cpp
   :lines: 13-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example4.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_scene_example4.py
   :language: python
   :lines: 7-

..  figure:: Tutorial_Output/out_scene_ex4_render_viridis.png
    :scale: 50 %
    :align: center

    Render with Viridis Color table


..  figure:: Tutorial_Output/out_scene_ex4_render_inferno.png
    :scale: 50 %
    :align: center

    Render with Inferno Color table