.. ############################################################################
.. # Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Ascent.
.. #
.. # For details, see: http://software.llnl.gov/ascent/.
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

Tutorial Demos
=================

.. note::
   
   WIP

Demo 1: First Light
-----------------------

``examples/python/ascent_python_render_example.py``

Demo 2: Using Pipelines and Scenes
----------------------------------------

* cloverleaf3d: volume rendering
* cloverleaf3d: isosurface + pseudocolor
* cloverleaf3d: isosurface + pseudocolor & volume rendering

Demo 3: Creating Cinema Extracts
----------------------------------

* cloverleaf3d: cinema spec a


Demo 4: Custom Python Extracts
-----------------------------------

* cloverleaf3d: custom python histogram

Ascent's `python` extract provides a simple path to run python scripts for
custom analysis. 

For this demo we use numpy and mpi4py to compute a histogram of Cloverleaf's energy
field. 

First, since we will use the Cloverleaf3d ascent integration, make sure you are in 
``examples/proxies/cloverleaf3d`` directory of your Ascent install.

Edit the ``ascent_actions.json`` file to define a single python extract that runs a script file:

.. literalinclude:: ../../../src/examples/tutorial/demo_4/ascent_actions.json
   :language: json

This adds a `python` extract that will use an embedded python to execute 
``ascent_tutorial_demo_4_histogram.py``, specified using the ``file`` parameter. 
The `python` extract also supports a ``source`` parameter that allows you to pass 
a python script as a string. 


Create ``ascent_tutorial_demo_4_histogram.py``


.. literalinclude:: ../../../src/examples/tutorial/demo_4/ascent_tutorial_demo_4_histogram.py 
   :language: python
   :lines: 43-



There are only two functions provided by ascent

* ``ascent_mpi_comm_id()``  - returns the Fortran style MPI handle (an integer) of the MPI Communicator Ascent is using

* ``ascent_data()`` - returns a Conduit tree with the data publish by this MPI Task

   
   

