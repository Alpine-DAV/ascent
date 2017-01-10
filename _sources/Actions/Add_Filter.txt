.. ############################################################################
.. # Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-716457
.. #
.. # All rights reserved.
.. #
.. # This file is part of Conduit.
.. #
.. # For details, see: http://software.llnl.gov/strawman/.
.. #
.. # Please also read strawman/LICENSE
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

.. danger::
    should we include this yet?

add_filter
==========

Filters apply and operation to the input data set to create a new data set.
Currently, filters are only supported in the EAVL pipeline, but filter will be coming to the VTK-m pipeline in coming versions.
The EAVL pipeline has has several filters that are supported, although some are serial.
Currently, filter in EAVL are only applied to the first plot added.

  - Box
  - Isosurface
  - Cell to node
  - Threshold
  - External faces

Box
---
The box filter will clip all cells that are outside the provided range.
Range is a array of doubles in the format (x_min, x_max, y_min, y_max, z_min, z_max).

.. code-block:: json

  {
    "action" : "add_filter",
    "type"   : "box_filter",
    "range"  : [0.0,0.0,0.0,10.0,10.0,10.0]
  }

Isosurface
----------
The isosurface filter evalates a node-centered scalar field for all points at a given iso-value.
This results in a surface if the the iso-value is within the scalar field.

.. code-block:: json

  {
    "action"    : "add_filter",
    "type"      : "isosurface_filter",
    "iso_value" : 1.0
  }

.. danger::
    this should be called "recenter"

Cell To Node
------------
The cell to node filter re-centers a scalar field from cell-centered quantities (i.e., zones) to point-centered quantities (i.e., zones).
This filter takes no input and operates on the active plot variable.

.. code-block:: json

  {
    "action" : "add_filter",
    "type"   : "cell_to_node_filter"
  }

Threshold
---------
The threshold filter removes cells that are not contained within a specified scalar range.


.. code-block:: json

  {
    "action"    : "add_filter",
    "type"      : "threshold_filter",
    "min_value" : 0.5,
    "max_value" : 1.5 
  }

External Faces
--------------
External faces extract all the faces of a cell set that are note shared between cells.
When rendering surfaces, internal faces are never seen and are removed when calling this filter.
External faces takes no arguments.

.. code-block:: json

  {
    "action" : "add_filter",
    "type"   : "external_faces_filter"
  }

