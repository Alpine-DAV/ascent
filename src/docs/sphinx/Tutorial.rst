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

Tutorial
=================

This page outlines how to run several demos included with Ascent.

These demos assume you have an Ascent install. If you have access to Docker, the easiest way to test the waters is via the ``alpinedav/ascent`` Docker image. For more details about using this image see :ref:`demos_using_docker`. You can also build Ascent with `Spack <https://spack.io/>`_. For more details see :ref:`building_with_spack`.



.. _tutorial_demo_1:

Demo 1: First Light
-----------------------

*Render a sample dataset using Ascent from C++ and Python*

For this demo, we run some of the "First Light" examples are installed with Ascent to enable users to quickly test ascent in their build system.

C++ Example (`examples/ascent/using-with-make/ascent_render_example.cpp`):

.. literalinclude:: ../../../src/examples/using-with-make/ascent_render_example.cpp
   :language: cpp
   :lines: 45-

Python Example (`examples/ascent/python/ascent_python_render_example.py`):

.. literalinclude:: ../../../src/examples/python/ascent_python_render_example.py
   :language: python
   :lines: 45-


These examples render the same example data set using ray casting to create a pseudocolor plot. 
The data set is one of the built-in Conduit Mesh Blueprint examples, in this case an unstructured mesh composed of hexagons. The Conduit C++ and Python interfaces are very similar, with the C++ interface heavily influenced by the ease of use of Python. 


Demo 2: Using Pipelines and Scenes
----------------------------------------

*Use Ascent's pipelines in Cloverleaf3D to transform data and render in situ*

For this demo, we will use Ascent to process data from :ref:`Cloverleaf3D  <cloverleaf3d_integration>` in situ. 

To begin, make sure you are in the ``examples/ascent/proxies/cloverleaf3d`` directory of your Ascent install.

The default integration example for Cloverleaf3D sets up Ascent to volume render the ``energy`` field. Here is a snippet of the related Fortran code that specifies the default actions: 


.. literalinclude:: ../../../src/examples/proxies/cloverleaf3d-ref/visit.F90
   :language: fortran
   :lines: 267-282


Ascent also allows you to override compiled in actions with a  ``ascent_actions.json`` file. In this case, the file we provide with Cloverleaf3D mirrors the compiled in actions:

Cloverleaf3D default  ``ascent_actions.json`` file (`examples/ascent/proxies/cloverleaf3d-ref/ascent_actions.json`):

.. literalinclude:: ../../../src/examples/proxies/cloverleaf3d-ref/ascent_actions.json
   :language: json

We will override the default actions to compute contours of the input data and render the result. To do this we use a pipeline. Ascent's pipelines allow you add transforms that modify the mesh data published to Ascent. Pipeline results can be rendered in a scene or used as input to extracts. 

Edit the ``ascent_actions.json`` to create a pipeline that computes contours and renders them using a pseudocolor plot:

.. literalinclude:: ../../../src/examples/tutorial/demo_2/contour.json
   :language: json
  
(Also available in install directory: `examples/ascent/tutorial/demo_2/contour.json`)

You can also compose more complex scenes that render both pipeline results and the published data. 
To demonstrate this, we combine the pseudocolor rendering of the contour results with a volume rendering of the entire mesh:

.. literalinclude:: ../../../src/examples/tutorial/demo_2/volume_contour.json
   :language: json
  

(Also available in install directory: `examples/ascent/tutorial/demo_2/volume_contour.json`)

Demo 3: Creating Cinema Extracts
----------------------------------

*Use Ascent to create a Cinema database from Cloverleaf3D that can be explored after the simulation finishes*


In this demo, we use Ascent to render out a :ref:`Cinema database  <actions_cinema>` of a plot in Cloverleaf3D.

Make sure you are in ``examples/ascent/proxies/cloverleaf3d`` directory of your Ascent install. Edit the 
Cloverleaf3D ``ascent_actions.json`` file to direct Ascent to render out a scene using the Cinema Astaire
specification (Spec-A):

.. literalinclude:: ../../../src/examples/tutorial/demo_3/ascent_actions.json
   :language: json
  
(Also available in install directory: `examples/ascent/tutorial/demo_3/ascent_actions.json`)

Run Cloverleaf3D with this setup and it will render several viewpoints and construct Cinema database
in the current directory. 
You can then open this database with a Cinema viewer and interactively explore views of data set 
after the simulation finishes. 


Demo 4: Custom Python Extracts
-----------------------------------

*Use Ascent to run a Python script which computes a histogram of Cloverleaf3D's energy field in situ*

Ascent's Python extract provides a simple path to run Python scripts for
custom analysis. Ascent provides the Python environment, so Python extracts can 
for any host code (even those without a Python interface).
 


For this demo we use numpy and mpi4py to compute a histogram of Cloverleaf3D's 
energy field. 

Again, since we will use the Cloverleaf3D Ascent integration, make sure you are in 
``examples/ascent/proxies/cloverleaf3d`` directory of your Ascent install. Then edit the ``ascent_actions.json`` 
file to define a single python extract that runs a script file:



.. literalinclude:: ../../../src/examples/tutorial/demo_4/ascent_actions.json
   :language: json

(Also available in install directory: `examples/ascent/tutorial/demo_4/ascent_actions.json`)

This requests a python extract that will use an embedded python interpreter to execute 
``ascent_tutorial_demo_4_histogram.py``, which is specified using the ``file`` parameter.
The python extract also supports a ``source`` parameter that allows you to pass 
a python script as a string. 


Next, create our analysis script ``ascent_tutorial_demo_4_histogram.py``:

.. literalinclude:: ../../../src/examples/tutorial/demo_4/ascent_tutorial_demo_4_histogram.py 
   :language: python
   :lines: 43-

(Also available in install directory: `examples/ascent/tutorial/demo_4/ascent_tutorial_demo_4_histogram.py`)


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

.. _demos_using_docker:

Running Demos using Docker
-----------------------------------

If you have Docker installed you can obtain a Docker image with a ready-to-use ascent install from `Docker Hub <https://hub.docker.com/r/alpinedav/ascent/>`_.

Fetch the latest Ascent image:

.. code::

    docker pull alpinedav/ascent

After the download completes, create and run a container using this image:

.. code::

    docker run -p 8000:8000 -p 10000:10000 -t -i alpinedav/ascent

(The ``-p`` is used to forward ports between the container and your host machine, we use these ports to allow web servers on the container to serve data to the host.)


You will now be at a bash prompt in you container. 

To add the proper paths to Python and MPI to your environment run:

.. code::

    source ascent_docker_setup.sh

The ascent source code is at ``/ascent/src/``, and the install is at ``/ascent/install-debug``.

Next, try running the python example mentioned in :ref:`tutorial_demo_1`:

.. code::

    cd /ascent/install-debug/examples/ascent/python
    python ascent_python_render_example.py

You should see some verbose output and ``out_ascent_render_3d.png`` will be created. 


To view output files you can use a simple Python web server to expose files from the container to your host machine:

.. code::

    python -m SimpleHTTPServer 10000


With this server running, open up a web browser on your host machine and view localhost:10000. You should be able to click on ``out_ascent_render_3d.png`` and view the rendered result in your web browser. 

You should now be ready to run the other demos, remember to use the Python web server to browse results.



