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

SHORT BLURB ABOUT EXTRACTS, link to docs etc

Exporting input mesh data to Blueprint HDF5 files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_pipeline_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_pipeline_example1.cpp
   :language: cpp

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_pipeline_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_pipeline_example1.py
   :language: python



VisIt 2.13 or newer, when built with Conduit support, can load and visualize these files.


Exporting the result of a pipeline to Blueprint HDF5 files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_pipeline_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_pipeline_example2.cpp
   :language: cpp

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_pipeline_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_pipeline_example2.py
   :language: python


Creating a Cinema image database for post-hoc exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example2.cpp
   :language: cpp

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example2.py
   :language: python


If you are already in Python land this may seem strange, but for C++ and FORTRAN codes, the Python extract gives you access to a distributed-memory python environment with in-situ access to mesh data


Using a Python Extract to execute custom Python analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ <../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example3.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/ascent_extract_example3.cpp
   :language: cpp

:download:`Python <../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example3.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/ascent_extract_example3.py
   :language: python


If you are already in Python land this may seem strange, but for C++ and FORTRAN codes, the Python extract gives you access to a distributed-memory python environment with in-situ access to mesh data
