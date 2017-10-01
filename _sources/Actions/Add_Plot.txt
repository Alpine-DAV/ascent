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
.. _add_plot-label:

add_plot
=========

The only required entry is the name of mesh field to plot.

.. code-block:: json
   :emphasize-lines: 3,3

   {
    "action" : "add_plot",
    "field_name"    : "p"
   }

Rendering Options
^^^^^^^^^^^^^^^^^
When only selecting the variable, all default rendering options are used. 
The default camera angle will always place the data set in the center of the image, and the default rendering type is a pseudocolor plot of the surface of the data set.
The top-level rendering option allow you to specify the type of renderer, image dimensions.
Additional parameters allow you to specific camera and color map options.

- ``file_name`` if specified, the image will be saved to the file system. Otherwise, images will be streamed to the web server.
- ``width`` image width in pixels
- ``height`` image height in pixels
- ``renderer`` The VTK-m pipeline includes a renderer. Valid options are ``raytracer`` and ``volume``. 
- ``color_map`` specifies a the color map to use
- ``camera`` specifies the camera parameters to use

Color Map
"""""""""
The color map translates normalized scalars to color values.
Minimally, a color map name needs to be specified, but the ``color_map`` nodes allows you to specify RGB and Alpha (opacity) control points for complete customization of color maps. 
Alpha control points are used when rendering volumes.
Color map names names can be found in the `VTK-m repository <https://gitlab.kitware.com/vtk/vtk-m/blob/master/vtkm/rendering/ColorTable.cxx>`_.
Colors are three double precision values between 0 and 1.
Alphas and positions  are a single double precision values between 0 and 1.

Here is an example of the color map node that define three RGB control points and two alpha control points that defines a custom color map for volume rendering:

.. code-block:: json

   {
     "color_map": 
     {
        "control_points": 
        [
          {
            "type": "rgb",
            "position": 0.0,
            "color": [1.0, 0.0, 0.0]
          },
      
          {
            "type": "rgb",
            "position": 0.5,
            "color": [0.0, 1.0, 0.0]
          },
      
          {
            "type": "rgb",
            "position": 0.5,
            "color": [0.0, 0.0, 1.0]
          },
      
          {
            "type": "alpha",
            "position": 0.0,
            "alpha": 0.0
          },
      
          {
            "type": "alpha",
            "position": 1.0,
            "alpha": 1.0
          }
        ]
     }
   }

The equivalent c++ code is:

.. code-block:: c++

    add_plot["action"] = "add_plot";
    add_plot["field_name"] = "p";
    add_plot["render_options/file_name"] = outFileName;
    add_plot["render_options/renderer"] = "volume";
    
    conduit::Node control_points;
    
    conduit::Node &point1 = control_points.append();
    point1["type"] = "rgb";
    point1["position"] = 0.;
    double color[3] = {1., 0., 0.};
    point1["color"].set_float64_ptr(color, 3);
    
    conduit::Node &point2 = control_points.append();
    point2["type"] = "rgb";
    point2["position"] = 0.5;
    color[0] = 0;
    color[1] = 1.;
    point2["color"].set_float64_ptr(color, 3);
    
    conduit::Node &point3 = control_points.append();
    point3["type"] = "rgb";
    point3["position"] = 1.0;
    color[1] = 0;
    color[2] = 1.;
    point3["color"].set_float64_ptr(color, 3);
    
    conduit::Node &point4 = control_points.append();
    point4["type"] = "alpha";
    point4["position"] = 0.;
    point4["alpha"] = 0.;
    
    conduit::Node &point5 = control_points.append();
    point5["type"] = "alpha";
    point5["position"] = 1.0;
    point5["alpha"] = 1.;
     
    add_plot["render_options/color_map/control_points"] = control_points;

It is also possible to combine existing color maps defined by name and combine it with custom alpha control points. 
In the example below, we specify a thermal color map and add two alpha control points.
The opacity is linearly interpolated from 0 (fully transparent) to 1 (fully opaque) across the the color map.

.. code-block:: json

   {
     "color_map": 
     {  
        "name" : "thermal",
        "control_points": 
        [
           {
            "type": "alpha",
            "position": 0.0,
            "alpha": 0.0
          },

          {
            "type": "alpha",
            "position": 1.0,
            "alpha": 1.0
          }
        ]
     }
   }

Camera Parameters
"""""""""""""""""
Camera parameters can also be controlled through a Conduit Node and are all expected to be double precision values. The supported parameters are:

- ``look_at`` an array of 3 values that specifies the point the camera is looking at
- ``position`` an array of 3 values that specifies the camera position
- ``up`` an array of 3 values that specifies the camera up vector
- ``fov`` 1 value that specifies the field of view in degrees
- ``xpan`` 1 value that specifies the distance in the x direction to pan the camera
- ``ypan`` 1 value that specifies the distance in the y direction to pan the camera
- ``zpan`` 1 value that specifies the distance in the z direction to pan the camera
- ``zoom`` 1 value that specifies the amount of camera zoom
- ``nearplane`` 1 value that specifies the distance to the near plane of the camera
- ``farplane`` 1 value that specifies the distance to the far plane of the camera

Ascent always creates default parameters camera based on the spatial extents of the data set, and all or a few of the camera parameters can be modified.
Like all the other action parameters, each can be specified in the actions json file or can be specified programmatically:

.. code-block:: json

   {
     "camera": 
     {
       "position": [1.4, 1.4, 1.4],
       "look_at": [0.6, 0.6, 0.6],
       "fov": 45.0
     }
   }


.. code-block:: c++
  
   // Create the camera node 
   conduit::Node camera;
   // Set the camera position
   double position[3] = {1.4, 1.4, 1.4};
   camera["position"].set_float64_ptr(position,3);
   double look_at[3] = {.6, .6, .6};
   // Point the camera to the data set
   camera["look_at"].set_float64_ptr(look_at,3);
   // Set the field of view to 45 degrees
   camera["fov"] = 45.0;
   // Add the camera parameters to the plot
   add_plot["render_options/camera"] = camera;
