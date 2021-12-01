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

.. _tutorial_extracts:

Capturing data with Extracts
-----------------------------------

Extracts are the construct that allows users to capture and process data outside Ascent's pipeline infrastructure. Extract use cases include: Saving mesh data to HDF5 files, creating Cinema databases, and running custom Python analysis scripts. These examples outline how to use several of Ascent's extracts. See Ascent's :ref:`extracts`  docs for deeper details on Extracts.


Exporting input mesh data to Blueprint HDF5 files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related docs: :ref:`extracts_relay`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example1.cpp
   :language: cpp
   :lines: 50-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example1.py
   :language: python
   :lines: 45-

How can you use these files?

You can use them as input to Ascent :ref:`utils_replay`, which allows you to run Ascent outside of a simulation code.

You can also visualize meshes from these files using VisIt 2.13 or newer.

..  figure:: Tutorial_Output/out_extract_visit_braid_all.png
    :scale: 50 %
    :align: center
    
    Exported Mesh Rendered in VisIt


VisIt can also export Blueprint HDF5 files, which can be used as input to Ascent Replay.

Blueprint files help bridge in situ and post hoc workflows.


Exporting selected fields from input mesh data to Blueprint HDF5 files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related docs: :ref:`extracts_relay`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example2.cpp
   :language: cpp
   :lines: 50-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example2.py
   :language: python
   :lines: 45-

..  figure:: Tutorial_Output/out_extract_visit_braid_one.png
    :scale: 50 %
    :align: center
    
    Exported Mesh Rendered in VisIt

Exporting the result of a pipeline to Blueprint HDF5 files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Related docs: :ref:`extracts_relay`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example3.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example3.cpp
   :language: cpp
   :lines: 50-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example3.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example3.py
   :language: python
   :lines: 45-

..  figure:: Tutorial_Output/out_extract_visit_braid_contour.png
    :scale: 50 %
    :align: center
    
    Exported Mesh Rendered in VisIt

Creating a Cinema image database for post-hoc exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Related docs: :ref:`actions_cinema`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example4.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example4.cpp
   :language: cpp
   :lines: 50-

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example4.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example4.py
   :language: python
   :lines: 45-

..  figure:: Tutorial_Output/out_extracts_cinema_snap.png
    :scale: 50 %
    :align: center

    Snapshot of Cinema Database Result


Using a Python Extract to execute custom Python analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related docs: :ref:`extracts_python`

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example5.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example5.cpp
   :language: cpp
   :lines: 50-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_cpp_extract_example5.txt

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example5.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example5.py
   :language: python
   :lines: 45-

**Output**

.. literalinclude:: Tutorial_Output/out_txt_py_extract_example5.txt

