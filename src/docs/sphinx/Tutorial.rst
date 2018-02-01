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


Demo 1: First Light
-----------------------

*Render a sample dataset using Ascent from C++ and Python*

For this demo, we run some of the "First Light" examples are installed with Ascent to enable users to quickly test ascent in their build system.

C++ Example:

.. literalinclude:: ../../../src/examples/using-with-make/ascent_render_example.cpp
   :language: cpp
   :lines: 45-

Python Example:

.. literalinclude:: ../../../src/examples/python/ascent_python_render_example.py
   :language: python
   :lines: 45-


These examples render the same example data set using ray casting to create a pseudocolor plot. 
The data set is one of the built-in Conduit Mesh Blueprint examples, in this case an unstructured mesh composed of hexagons. The Conduit C++ and Python interfaces are very similar, with the C++ interface heavily influenced by the ease of use of Python. 


Demo 2: Using Pipelines and Scenes
----------------------------------------

*Use Ascent's pipelines in Cloverleaf3D to transform data and render scenes in situ*

* cloverleaf3d: volume rendering
* cloverleaf3d: isosurface + pseudocolor
* cloverleaf3d: isosurface + pseudocolor & volume rendering

Demo 3: Creating Cinema Extracts
----------------------------------

*Use Ascent to create a Cinema database from Cloverleaf3D that can be explored after the simulation finishes*

* cloverleaf3d: cinema spec a


Demo 4: Custom Python Extracts
-----------------------------------

*Use Ascent to run a Python script which computes a histogram of Cloverleaf3D's energy field in situ*

Ascent's python extract provides a simple path to run python scripts for
custom analysis. Ascent provides the python environment, so python extracts can 
for any host code (even those without a Python interface).
 


For this demo we use numpy and mpi4py to compute a histogram of Cloverleaf3D's 
energy field. 

First, since we will use the Cloverleaf3D Ascent integration, make sure you are in 
``examples/proxies/cloverleaf3d`` directory of your Ascent install. Then edit the ``ascent_actions.json`` 
file to define a single python extract that runs a script file:

.. literalinclude:: ../../../src/examples/tutorial/demo_4/ascent_actions.json
   :language: json

This requests a python extract that will use an embedded python interpreter to execute 
``ascent_tutorial_demo_4_histogram.py``, which is specified using the ``file`` parameter.
The python extract also supports a ``source`` parameter that allows you to pass 
a python script as a string. 


Next, create our analysis script ``ascent_tutorial_demo_4_histogram.py``:


.. literalinclude:: ../../../src/examples/tutorial/demo_4/ascent_tutorial_demo_4_histogram.py 
   :language: python
   :lines: 43-

This script computes a basic histogram counting the number of energy field elements that
fall into a set of uniform bins.  It uses numpy's histogram function and mpi4py to handle 
distributed-memory coordination. 

Note, there are only two functions provided by ascent:

* ``ascent_data()``
 
   Returns a Conduit tree with the data published to this MPI Task. 

   Conduit's Python API mirrors its C++ API, but with leaves returned as numpy.ndarrays.
   For examples of how to use Conduit's Python API, see the 
   `Conduit Python Tutorial <http://llnl-conduit.readthedocs.io/en/latest/tutorial_python.html>`_.
   In this script, we simply fetch a ndarray that points to the values of a known field, energy. 
   

* ``ascent_mpi_comm_id()``

   Returns the Fortran style MPI handle (an integer) of the MPI Communicator Ascent is using.

   In this script, we use this handle and ``mpi4py.MPI.Comm.f2py()`` to obtain a mpi4py 
   Communicator.



Finally, run Cloverleaf3D:

.. code::

   mpiexec -n 2 ./cloverleaf3d_par 


With the default ``clover.in`` settings, Ascent execute the python script every 10th cycle. 
The script computes the histogram of the energy field and prints a summary like the following: 

.. code::

  Energy extents: 1.0 2.93874088025

  Histogram of Energy:

  Counts:
  [159308   4041   1763   2441   2044   1516   1780   1712   1804   1299
     1366   1959   1668   2176   1287   1066    962   2218   1282   1006
     1606   2236   1115   1420   1185   1293   2495   1255   1191   1062
     1435   1329   2371   1619   1067   2513   3066   2124   2755   3779
     3955   4933   2666   3279   3318   3854   3123   4798   2604]

  Bin Edges:
  [ 1.          1.03956614  1.07913228  1.11869842  1.15826456  1.1978307
    1.23739684  1.27696298  1.31652912  1.35609526  1.3956614   1.43522754
    1.47479368  1.51435983  1.55392597  1.59349211  1.63305825  1.67262439
    1.71219053  1.75175667  1.79132281  1.83088895  1.87045509  1.91002123
    1.94958737  1.98915351  2.02871965  2.06828579  2.10785193  2.14741807
    2.18698421  2.22655035  2.26611649  2.30568263  2.34524877  2.38481491
    2.42438105  2.4639472   2.50351334  2.54307948  2.58264562  2.62221176
    2.6617779   2.70134404  2.74091018  2.78047632  2.82004246  2.8596086
    2.89917474  2.93874088]


