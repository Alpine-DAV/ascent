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

.. _tutorial_conduit_basics:

Conduit Basics
----------------

Ascent's API is based on `Conduit <http://software.llnl.gov/conduit/>`_. Both mesh data and action descriptions are passed to Ascent as Conduit trees.  The Conduit C++ and Python interfaces are very similar, with the C++ interface heavily influenced by the ease of use of Python. These examples provide basic knowledge about creating Conduit Nodes to use with Ascent. You can also find more introductory Conduit examples in Conduit's `Tutorial Docs <https://llnl-conduit.readthedocs.io/en/latest/conduit.html>`_ .

Creating key-value entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example1.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example1.cpp
   :language: cpp

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example1.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example1.py
   :language: python


Creating a path hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example2.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example2.cpp
   :language: cpp

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example2.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example2.py
   :language: python


Setting array data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example3.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example3.cpp
   :language: cpp

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example3.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example3.py
   :language: python


Zero-copy vs deep copy of array data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`C++ Source <../../../src/examples/tutorial/ascent_intro/cpp/conduit_example4.cpp>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/cpp/conduit_example4.cpp
   :language: cpp

:download:`Python Source <../../../src/examples/tutorial/ascent_intro/python/conduit_example4.py>`

.. literalinclude:: ../../../src/examples/tutorial/ascent_intro/python/conduit_example4.py
   :language: python


