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

.. _tutorial_conduit_mesh_blueprint:

Conduit Blueprint Mesh Examples
---------------------------------

Simulation mesh data is passed to Ascent using a shared set of conventions called the
`Mesh Blueprint <https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html>`_.

Simply described, these conventions outline a structure to follow to create Conduit trees 
that can represent a wide range of simulation mesh types (uniform grids, unstructured meshes, etc).
Conduit's dynamic tree and zero-copy support make it easy to adapt existing data to conform to the Mesh Blueprint for use in tools like Ascent.

These examples outline how to create Conduit Nodes that describe simple single domain meshes and review
some of Conduits built-in mesh examples. More Mesh Blueprint examples are also detailed in Conduit's `Mesh Blueprint Examples Docs <https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#examples>`_ .

Creating a uniform grid with a single field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/blueprint_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/blueprint_example1.cpp
   :language: cpp
   :lines: 50-

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/blueprint_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/blueprint_example1.py
   :language: python
   :lines: 45-

..  figure:: Tutorial_Output/out_ascent_render_uniform.png
    :scale: 50 %
    :align: center

    Example Uniform Mesh Rendered



Creating an unstructured tet mesh with fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/blueprint_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/blueprint_example2.cpp
   :language: cpp
   :lines: 50-

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/blueprint_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/blueprint_example2.py
   :language: python
   :lines: 45-

..  figure:: Tutorial_Output/out_ascent_render_tets.png
    :scale: 50 %
    :align: center

    Example Tet Mesh Rendered

Experimenting with the built-in braid example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related docs: `Braid <https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#braid>`_ .

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/blueprint_example3.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/blueprint_example3.cpp
   :language: cpp
   :lines: 50-

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/blueprint_example3.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/blueprint_example3.py
   :language: python
   :lines: 45-

..  figure:: Tutorial_Output/out_ascent_render_braid_tv_0019.png
    :scale: 50 %
    :align: center

    Final Braid Time-varying Result Rendered

