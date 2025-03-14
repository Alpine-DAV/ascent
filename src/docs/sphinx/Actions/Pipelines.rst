.. ############################################################################
.. # Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
.. # Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
.. # other details. No copyright assignment is required to contribute to Ascent.
.. ############################################################################

.. _Pipelines:

Pipelines
=========
Pipelines allow users to compose filters that transform the published input data into new meshes.
This is where users specify typical geometric transforms (e.g., clipping and slicing), field based transforms (e.g., threshold and contour), etc.
The resulting data from each Pipeline can be used as input to Scenes or Extracts.
Each pipeline contains one or more filters that transform the published mesh data.
When more than one filter is specified, each successive filter consumes the result of the previous filter, and filters are executed in the order in which they are declared.

The code below shows how to declare two pipelines, and generate images of the pipeline results.
The first applies a contour filter to extract two isosurfaces of the scalar field ``noise``.
The second pipeline applies a threshold filter to screen the ``noise`` field, and then a clip
filter to extract the intersection of what remains from the threshold with a sphere.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "contour";
  // filter parameters
  conduit::Node contour_params;
  contour_params["field"] = "noise";
  constexpr int num_iso_values = 2;
  double iso_values[num_iso_values] = {0.0, 0.5};
  contour_params["iso_values"].set_external(iso_values, num_iso_values);
  pipelines["pl1/f1/params"] = contour_params;

  // pipeline 2
  pipelines["pl2/f1/type"] = "threshold";
  // filter parameters
  conduit::Node thresh_params;
  thresh_params["field"]  = "noise";
  thresh_params["min_value"] = 0.0;
  thresh_params["max_value"] = 0.5;
  pipelines["pl2/f1/params"] = thresh_params;

  pipelines["pl2/f2/type"]   = "clip";
  // filter parameters
  conduit::Node clip_params;
  clip_params["topology"] = "mesh";
  clip_params["sphere/center/x"] = 0.0;
  clip_params["sphere/center/y"] = 0.0;
  clip_params["sphere/center/z"] = 0.0;
  clip_params["sphere/radius"]   = .1;
  pipelines["pl2/f2/params/"] = clip_params;

  // make some imaages of the data
  conduit::Node scenes;
  // add a plot of pipeline 1
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p1/field"] = "noise";

  // add a plot of pipeline 2
  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/pipeline"] = "pl2";
  scenes["s2/plots/p1/field"] = "noise";

  // setup actions
  conduit::Node actions;

  conduit::Node add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;

  conduit::Node add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  actions.append()["action"] = "execute";

  Ascent ascent;
  ascent.open();
  ascent.publish(mesh); // mesh not shown
  ascent.execute(actions);
  ascent.close();


Ascent is under active development and features are being added rapidly.
In the following section we provide brief descriptions and code examples of the supported filters.
For complete code examples, please consult the unit tests located in ``src/tests/ascent``..


Filters
-------
Our filter API consists of the type of filter and the parameters associated with the filter in the general form:

.. code-block:: yaml

  type: "filter_name"
  params:
      string_param: "string"
      double_param: 2.0

In c++, the equivalent declarations would be as follows:

.. code-block:: c++

  conduit::Node filter;
  filter["type"] = "filter_name";
  filter["params/string_param"] = "string";
  filter["params/double_param"] = 2.0;

Contour
~~~~~~~
The contour filter evaluates a node-centered scalar field for all points at a given iso-value.
This results in a surface if the iso-value is within the scalar field.
``iso_vals`` can contain a single double or an array of doubles.
Additionally, instead of specifying exact iso-values, a number of 'levels' can be entered.
In this case, iso-values will be created evenly spaced through the scalar range. For example,
if the scalar range is `[0.0, 1.0]` and 'levels' is set to `3`, then the iso-values `(0.25, 0.5, 0.75)`
will be created.
The code below provides examples creating a pipeline using all three methods:

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "contour";
  // filter knobs
  conduit::Node &contour_params = pipelines["pl1/f1/params"];
  contour_params["field"] = "braid";
  contour_params["iso_values"] = -0.4;

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "contour";
  // filter knobs
  conduit::Node &contour_params = pipelines["pl1/f1/params"];
  contour_params["field"] = "braid";
  constexpr int num_iso_values = 3;
  double iso_vals[num_iso_values] = {-0.4, 0.2, 0.4};
  contour_params["iso_values"].set_external(iso_vals, num_iso_values);

.. _contourfig:

..  figure:: ../images/contour.png
    :scale: 50 %
    :align: center

    An example image of multiple contours produced using the previous code sample.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "contour";
  // filter knobs
  conduit::Node &contour_params = pipelines["pl1/f1/params"];
  contour_params["field"] = "braid";
  contour_params["levels"] = 5;

.. _contourlevelsfig:

..  figure:: ../images/contour_levels.png
    :scale: 50 %
    :align: center

    An example of creating five evenly spaced iso-values through a scalar field.

:numref:`Figure %s <contourfig>` shows an image produced from multiple contours.
All contour examples are  located in the test in the file `contour test <https://github.com/Alpine-DAV/ascent/blob/develop/src/tests/ascent/t_ascent_contour.cpp>`_.

Threshold
~~~~~~~~~
The threshold filter removes cells that are not contained within a specified scalar range.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "threshold";
  // filter knobs
  conduit::Node &thresh_params = pipelines["pl1/f1/params"];
  thresh_params["field"] = "braid";
  thresh_params["min_value"] = -0.2;
  thresh_params["max_value"] = 0.2;

.. _thresholdfig:

..  figure:: ../images/threshold.png
    :scale: 50 %
    :align: center

    An example image of the threshold filter using the previous code sample.

:numref:`Figure %s <thresholdfig>` shows an image produced from a threshold filter.
The full example is located in the file `threshold test <https://github.com/Alpine-DAV/ascent/blob/develop/src/tests/ascent/t_ascent_threshold.cpp>`_.

Slice
~~~~~
The slice filter extracts a 2d plane from a 3d data set.
The plane is defined by a point (on the plane) and a normal vector (not required to be normalized).

.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "slice";
  // filter knobs
  conduit::Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["point/x"] = 0.f;
  slice_params["point/y"] = 0.f;
  slice_params["point/z"] = 0.f;

  slice_params["normal/x"] = 0.f;
  slice_params["normal/y"] = 0.f;
  slice_params["normal/z"] = 1.f;

.. _slicefig:

..  figure:: ../images/slice.png
    :scale: 50 %
    :align: center

    An example image of the slice filter on a element-centered variable using the previous code sample.

:numref:`Figure %s <slicefig>` shows an image produced from the slice filter.
The full example is located in the file `slice test <https://github.com/Alpine-DAV/ascent/blob/develop/src/tests/ascent/t_ascent_slice.cpp>`_.

Three Slice
~~~~~~~~~~~
The three slice filter slices 3d data sets using three axis-aligned slice planes and
leaves the resulting planes in 3d where they can all be viewed at the same time.
Three slice is meant primarily for quick visual exploration of 3D data where the
internal features cannot be readily observed from the outside.

The slice planes will automatically placed at the center of the data sets spatial extents.
Optionally, offsets for each plane can be specified. Offsets for each axis are specified
by a floating point value in the range ``[-1.0, 1.0]``, where ``-1.0`` places the plane at the
minimum spatial extent on the axis, ``1.0`` places the plane at the maximum spatial extent
on the axis, and ``0.0`` places the plane at the center of the spatial extent. By default,
all three offsets are ``0.0``.

.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "3slice";

.. _threeslicefig:

..  figure:: ../images/three_slice.png
    :scale: 50 %
    :align: center

    An example image of the three slice filter on a element-centered variable using the
    previous code sample with automatic slice plane placement.

.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "3slice";

  // filter knobs (all these are optional)
  conduit::Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["x_offset"] = 1.f;   // largest value on the x-axis
  slice_params["y_offset"] = 0.f;   // middle of the y-axis
  slice_params["z_offset"] = -1.f;  // smalles value of the z-axis

.. _threeslice2fig:

..  figure:: ../images/three_slice2.png
    :scale: 50 %
    :align: center

    An example image of the three slice filter on a element-centered variable using the
    previous code sample with user specified offsets for each axis.

:numref:`Figures %s <threeslicefig>` and :numref:`%s <threeslice2fig>` show an images produced from the three slice filter.
The full example is located in the file `slice test <https://github.com/Alpine-DAV/ascent/blob/develop/src/tests/ascent/t_ascent_slice.cpp>`_.

Automatic Slice
~~~~~~~~~~~~~~~
The automatic slice filter extracts a 2d plane from a 3d data set by slicing the data set in a user-specified direction a user-specified number of times, and then selects the slice that has the highest entropy for the user-specified field.
The slicing direction of the data set is determined by a normal (not required to be normalized), and the number of slices evaluated is specified by the number of levels, which will equally space the slices in the normal direction.
Automatic slice is meant primarily for quick visual exploration of 3D data where the
internal features cannot be readily observed from the outside.

The slice planes will be automatically placed based on the ``normal`` provided and the number of ``levels`` specified.
The final output slice will be the slice that has the highest entropy for the specified ``field``.

Depending on the normal provided, the rendering camera may need to be adjusted in order to view the chosen slice. 
By default, the camera is pointed down the z-axis, so a normal of ``(0,0,1)`` does not need any adjusting. 
In contrast, if the normal is ``(1,0,0)``, the camera needs to be adjusted to point down the x-axis, this can be done by adjusting the ``azimuth`` to rotate the camera horizontally around data set. 
Additionally, if the normal is ``(0,1,0)`` and the camera needs to point down the y-axis, this can be achieved by using the ``elevation`` camera parameter to rotate the camera vertically around the data set.

.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "auto_slice";

  // filter knobs (not optional)
  conduit::Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["normal/x"] = 0.f;   
  slice_params["normal/y"] = 0.f;   
  slice_params["normal/z"] = 1.f; 
  slice_params["field"] = "braid";
  slice_params["levels"] = 10;

.. _autoslicefig:

..  figure:: ../images/auto_slice_z_axis.png
    :scale: 50 %
    :align: center

    An example image of the automatic slice filter using the previous code sample. 
    This example uses a normal that points down the z-axis, the same as the default camera.

.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "auto_slice";

  // filter knobs (not optional)
  conduit::Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["normal/x"] = 1.f;   
  slice_params["normal/y"] = 0.f;   
  slice_params["normal/z"] = 0.f; 
  slice_params["field"] = "braid";
  slice_params["levels"] = 10;

  conduit::Node scenes;
  // add a plot of pipeline 1
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p1/field"] = "braid";
  //Need to turn camera 90 degrees horizontally
  //in order to point down x-axis
  scenes["s1/renders/r1/camera/azimuth"] = 90.0;
  scenes["s1/renders/r1/image_prefix"]     = output_file;

.. _autoslice2fig:

..  figure:: ../images/auto_slice_x_axis.png
    :scale: 50 %
    :align: center

    An example image of the automatic slice filter using the previous code sample. 
    This example uses a normal that points down the x-axis, meaning the angle camera needs to be adjusted using the azimuth.

.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "auto_slice";

  // filter knobs (not optional)
  conduit::Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["normal/x"] = 0.f;   
  slice_params["normal/y"] = 1.f;   
  slice_params["normal/z"] = 0.f; 
  slice_params["field"] = "braid";
  slice_params["levels"] = 10;

  conduit::Node scenes;
  // add a plot of pipeline 1
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p1/field"] = "braid";
  //Need to turn camera 90 degrees vertically
  //in order to point down y-axis
  scenes["s1/renders/r1/camera/elevation"] = 90.0;
  scenes["s1/renders/r1/image_prefix"]     = output_file;

.. _autoslice3fig:

..  figure:: ../images/auto_slice_y_axis.png
    :scale: 50 %
    :align: center

    An example image of the automatic slice filter using the previous code sample. 
    This example uses a normal that points down the y-axis, meaning the angle camera needs to be adjusted using the elevation.
    
.. code-block:: c++

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "auto_slice";

  // filter knobs (not optional)
  conduit::Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["normal/x"] = 1.f;   
  slice_params["normal/y"] = 1.f;   
  slice_params["normal/z"] = 0.f; 
  slice_params["field"] = "braid";
  slice_params["levels"] = 10;

  conduit::Node scenes;
  // add a plot of pipeline 1
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p1/field"] = "braid";
  //Need to turn camera 
  //90 degrees horizontally
  //and 45 degrees vertically
  //based on normal
  scenes["s1/renders/r1/camera/azimuth"] = 90.0;
  scenes["s1/renders/r1/camera/elevation"] = 45.0;
  scenes["s1/renders/r1/image_prefix"]     = output_file;

.. _autoslice4fig:

..  figure:: ../images/auto_slice_xy_axis.png
    :scale: 50 %
    :align: center

    An example image of the automatic slice filter using the previous code sample. 
    This example uses a normal that points in the xy-direction, meaning the angle camera needs to be adjusted using both the azimuth and elevation.

:numref:`Figures %s <autoslicefig>`, :numref:`%s <autoslice2fig>` , :numref:`%s <autoslice3fig>` , and :numref:`%s <autoslice4fig>` show images produced from the automatic slice filter.
The full example is located in the file `slice test <https://github.com/Alpine-DAV/ascent/blob/develop/src/tests/ascent/t_ascent_slice.cpp>`_.

Clip
~~~~
The clip filter removes cells from the specified topology using implicit functions.
By default, only the area outside of the implicit function remains, but the clip
can be inverted. There are three implicit functions that clip can use: sphere, box,
and plane.

.. code-block:: c++

  // define a clip by a sphere
  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "clip";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["topology"] = "mesh";
  clip_params["sphere/radius"] = 11.;
  clip_params["sphere/center/x"] = 0.;
  clip_params["sphere/center/y"] = 0.;
  clip_params["sphere/center/z"] = 0.;

.. _clipspherefig:

..  figure:: ../images/clip.png
    :scale: 50 %
    :align: center

    An example image of the clip filter using the previous code sample.
    The data set is a cube with extents from (-10, -10, -10) to (10, 10, 10), and the code removes a sphere centered at the origin with a radius of 11.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "clip";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["topology"] = "mesh";
  clip_params["invert"] = "true";
  clip_params["sphere/radius"] = 11.;
  clip_params["sphere/center/x"] = 0.;
  clip_params["sphere/center/y"] = 0.;
  clip_params["sphere/center/z"] = 0.;

.. _clipsphereInvertedfig:

..  figure:: ../images/clip_inverted.png
    :scale: 50 %
    :align: center

    An example of the same sphere clip, but in this case, the clip is inverted.

.. code-block:: c++

  // define a clip by a box
  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "clip";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["topology"] = "mesh";
  clip_params["box/min/x"] = 0.;
  clip_params["box/min/y"] = 0.;
  clip_params["box/min/z"] = 0.;
  clip_params["box/max/x"] = 10.01; // <=
  clip_params["box/max/y"] = 10.01;
  clip_params["box/max/z"] = 10.01;

.. _clipboxfig:

..  figure:: ../images/box_clip.png
    :scale: 50 %
    :align: center

    A box clip of the same data set that removes the octant on the positive x, y, and z axes.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "clip";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["topology"] = "mesh";
  clip_params["plane/point/x"] = 0.;
  clip_params["plane/point/y"] = 0.;
  clip_params["plane/point/z"] = 0.;
  clip_params["plane/normal/x"] = 1.;
  clip_params["plane/normal/y"] = 0.;
  clip_params["plane/normal/z"] = 0;

.. _clipplanefig:

..  figure:: ../images/clip_plane.png
    :scale: 50 %
    :align: center

    Clipping by a plane defined by a point on the plane and the plane normal.

:numref:`Figures %s <clipspherefig>`,
:numref:`%s <clipsphereInvertedfig>`,
:numref:`%s <clipboxfig>`, and
:numref:`%s <clipplanefig>` show an images produced from the clip filter.
All of the clip  examples are located in the file `clip test <https://github.com/Alpine-DAV/ascent/blob/develop/src/tests/ascent/t_ascent_clip.cpp>`_.

Clip By Field
~~~~~~~~~~~~~
The clip by field filter removes cells from the specified topology using the values in a scalar field.
By default, all values below the clip value are removed from the data set. As with clip by implicit function,
the clip can be inverted.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "clip_with_field";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["field"] = "braid";
  clip_params["clip_value"] = 0.;

.. _clipfieldfig:

..  figure:: ../images/clip_field.png
    :scale: 50 %
    :align: center

    An example of clipping all values below 0 in a data set.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "clip_with_field";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["field"] = "braid";
  clip_params["invert"] = "true";
  clip_params["clip_value"] = 0.;

.. _clipfieldinvertedfig:

..  figure:: ../images/clip_field_inverted.png
    :scale: 50 %
    :align: center

    An example of clipping all values above 0 in a data set.

IsoVolume
~~~~~~~~~
IsoVolume is a filter that clips a data set based on a minimum
and maximum value in a scalar field. All value outside of the minimum and maximum
values are removed from the data set.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "iso_volume";
  // filter knobs
  conduit::Node &clip_params = pipelines["pl1/f1/params"];
  clip_params["field"] = "braid";
  clip_params["min_value"] = 5.;
  clip_params["max_value"] = 10.;

.. _isovolume:

..  figure:: ../images/iso_volume.png
    :scale: 50 %
    :align: center

    An example of creating a iso-volume of values between 5.0 and 10.0.

Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The transform filter allows you to scale, translate, reflect, and rotate your mesh.
It also accepts a general 4x4 transformation matrix.

Translate Example:

.. code-block:: yaml

  -
    action: "add_pipelines"
    pipelines:
      pl1:
        f1:
          type: "transform"
          params:
            translate:
              x: 23.0
              y: 15.0

Scale Example:

.. code-block:: yaml

  -
    action: "add_pipelines"
    pipelines:
      pl1:
        f1:
          type: "transform"
          params:
            scale:
              x: 2.0
              y: 0.5
              z: 2.0

Reflect Example:

.. code-block:: yaml

  -
    action: "add_pipelines"
    pipelines:
      pl1:
        f1:
          type: "transform"
          params:
            reflect:
              x: 1.0

Rotate Example:

.. code-block:: yaml

  -
    action: "add_pipelines"
    pipelines:
      pl1:
        f1:
          type: "transform"
          params:
            rotate:
              angle: 45.0
              axis:
                y: 1.0


Matrix Example:

.. code-block:: yaml

  -
    action: "add_pipelines"
    pipelines:
      pl1:
        f1:
          type: "transform"
          params:
            matrix: [2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 50.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]


Particle Advection
~~~~~~~~~~~~~~~~~~~~
The particle advection filter distributes some number of weightless particles over a user-specified vector field (``field``) and, given some advection distance (``step_size``), advects them for some number of advection steps (``num_steps``).

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "particle_advection";
  //required params
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";                 // name of the vector field
  params["step_size"] = 0.01;              // advection step size
  params["num_steps"] = 100;               // number of advection steps

Users also need to specify how to generate seed placements (``seeds``). 
The seed placements can be an individual point (``point``), a list of points (``point_list``), a line (``line``), or a box (``box``). 
The seed placement type will determine the necessary parameters:

  - ``point`` requires a ``location`` as an [x,y,z] list of doubles.
  - ``point_list`` requires a ``location`` as an [x0,y0,z0,...,xn,yn,zn] list of doubles.
  - ``line`` requires a ``start`` and ``end`` as [x,y,z] lists of doubles, the number of seeds (``num_seeds``) to place on the line as well as defining the spacing between seeds (``sampling_type``) as either ``uniform`` or ``random``.
  - ``box`` requires the sampling space (``sampling_space``) to be defined (``boundary`` or ``interior``), the sampling type (``sampling_type``) to be defined (``random`` or ``uniform``). By default the boundary of the entire dataset is used, but user can define a new boundary (``x_extents``, ``y_extents``, and ``z_extents``).


At this time, Ascent can only save the output of the particle advection filter as an extract. For rendering, consider using the streamline filter. 

Streamlines
~~~~~~~~~~~~
The streamline filter behaves similarly to the particle advection filter, but as the particles are advected, the path of the particle is is collected as a streamline that can be rendered or saved as an extract. 
The streamlines are rendered using tubes, which transform the streamline data into a 3D surface. 
Tubes are on by default but they can be disabled, though this would also disable rendering capabilities. 

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] =  "streamline";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";                 // name of the vector field
  params["num_steps"] = 1;               // number of advection steps
  params["step_size"] = 0.01;              // advection step size
  params["seeds/type"] = "point";
  params["seeds/location"] = [-0.826997,-5.62082,3.52779]; 
  //all tubing params are optional
  params["enable_tubes"] = "true";         //default: true
  params["tube_size"] = 0.4;               //default: based on bounds
  params["tube_sides"] = 4;                //default: 3
  params["tube_val"] = 1.0;                //default: 0.0
  params["tube_capping"] = "true";         //default: true
  params["output_field"] = "lines";        //name of streamline tubes for rendering
                                           //default: "field" + "_streamlines" 
                                           //e.g "vel_streamlines"

..  figure:: ../images/tout_render_streamlines_point100.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of streamline seed placements using ``point``.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] =  "streamline";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";                 // name of the vector field
  params["num_steps"] = 1;               // number of advection steps
  params["step_size"] = 0.01;              // advection step size
  params["seeds/type"] = "point_list";
  params["seeds/location"] = [-9,-9,-9,1,1,1]; // two points
  //all tubing params are optional
  params["enable_tubes"] = "true";         //default: true
  params["tube_size"] = 0.4;               //default: based on bounds
  params["tube_sides"] = 4;                //default: 3
  params["tube_val"] = 1.0;                //default: 0.0
  params["tube_capping"] = "true";         //default: true
  params["output_field"] = "lines";        //name of streamline tubes for rendering
                                           //default: "field" + "_streamlines" 
                                           //e.g "vel_streamlines"

..  figure:: ../images/tout_render_streamlines_point_list100.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of streamline seed placements using ``point_list``.
.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] =  "streamline";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";                 // name of the vector field
  params["num_steps"] = 1;               // number of advection steps
  params["step_size"] = 0.01;              // advection step size
  params["seeds/type"] = "line";
  //required: how to space the seeds on the line
  params["seeds/sampling_type"] = "uniform"; //or "random"
  params["seeds/start"] = [-9,-9,-9]; // required: start of line
  params["seeds/end"] = [9,9,9];      // required: end of line
  params["seeds/num_seeds"] = 10;     // required: number of seeds
  //all tubing params are optional
  params["enable_tubes"] = "true";         //default: true
  params["tube_size"] = 0.1;               //default: based on bounds
  params["tube_sides"] = 4;                //default: 3
  params["tube_val"] = 1.0;                //default: 0.0
  params["tube_capping"] = "true";         //default: true
  params["output_field"] = "lines";        //name of streamline tubes for rendering
                                           //default: "field" + "_streamlines" 
                                           //e.g "vel_streamlines"

..  figure:: ../images/tout_render_streamlines_line100.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of streamline seed placements using ``line``.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] =  "streamline";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";                 // name of the vector field
  params["step_size"] = 0.01;              // advection step size
  params["num_steps"] = 1;               // number of advection steps
  //seed parameters
  params["seeds/type"] = "box";
  params["seeds/sampling_type"] = "uniform"; //or "random"
  params["seeds/sampling_space"] = "interior"; //or "boundary"
  //default is using the boundary of the entire dataset
  params["seeds/x_extents"] = [-9,9]; //optional: define the boundary
  params["seeds/y_extents"] = [-9,9]; //for the distribution
  params["seeds/z_extents"] = [-9,9]; //of the particles
  //all tubing params are optional
  params["enable_tubes"] = "true";         //default: true
  params["tube_size"] = 0.1;               //default: based on bounds
  params["tube_sides"] = 4;                //default: 3
  params["tube_val"] = 1.0;                //default: 0.0
  params["tube_capping"] = "true";         //default: true
  params["output_field"] = "lines";        //name of streamline tubes for rendering
                                           //default: "field" + "_streamlines" 
                                           //e.g "vel_streamlines"

..  figure:: ../images/tout_render_streamlines_box100.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of streamline seed placements using ``box``.

Streamlines with Charged Particles (WarpX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The streamlines with charged particles filter behaves similarly to the streamline filter, but instead utilizes charged particles, which are particles with physical attributes (``charge``, ``mass``, ``momentum``, ``weighting``), that are advected using magnetic (``b_field``) and electric (``e_field``) vector fields.
The resulting streamlines are rendered using tubes, which transform the streamline data into a 3D surface. 
Note: the tube functionality is not behaving correctly, currently this functionality is OFF by default. 
Otherwise, the resulting streamlines can be saved via an extract.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] =  "warpx_streamline";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  //vector fields
  params["b_field"] = "magnetic_field"; //default: B
  params["e_field"] = "electric_field"; //default: E
  //charged particle params
  params["charge_field"] = "charge_field";       //default: Charge
  params["mass_field"] = "mass_field";           //default: Mass
  params["momentum_field"] = "momentum_field";   //default: Momentum
  params["weighting_field"] = "weighting_field"; //default: Weighting
  //tubing params
  params["enable_tubes"] = "true";  //default: false
  params["tube_size"] = 0.2;        //default: based on bounds
  params["tube_sides"] = 4;         //default: 3
  params["tube_val"] = 1.0;         //default: 0.0
  params["tube_capping"] = "true";  //default: true
  params["output_field"] = "lines"; //name of streamline tubes for rendering
                                    //default: "b_field" + "e_field" + "_streamlines" 
                                    //e.g "B_E_streamlines"

Vector Magnitude
~~~~~~~~~~~~~~~~
Vector magnitude creates a new field on the data set representing the magitude
of a vector variable. The only parameters are the input vector field name
and the name of the new field.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "vector_magnitude";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";         // name of the vector field
  params["output_name"] = "mag";   // name of the output field

.. _vecmag:

..  figure:: ../images/vector_magnitude.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of vector magnitude


Vector Component
~~~~~~~~~~~~~~~~
Vector component creates a new scalar field on the data set by
extracting a component of a vector field. There are three required
parameters: the input field, the output field name, and the index
of the component to extract.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "vector_component";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "vel";         // name of the vector field
  params["output_name"] = "vel_x"; // name of the output field
  params["component"] = 0; // index of the component

Composite Vector
~~~~~~~~~~~~~~~~
Composite Vector creates a new vector field on the data set
by combining two or three scalar fields into a vector.
The first two fields are required and the presense of the
third field dictates whether a 2D or 3D vector is created.
Input fields can be different types (e.g., int32 and float32),
and the resulting vector field will be a float64.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "composite_vector";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field1"] = "pressure";      // (required)
  params["field2"] = "temperature";   // (required)
  params["field3"] = "bananas";       // (optional, 2D vector if not present)
  params["output_name"] = "my_vec";   // (required) name of the output field
  params["component"] = 0; // (required) index of the component


Recenter
~~~~~~~~
Recenter changes the association of a field. Fields associated with either `element` or `vertex` can
be interchanged by averaging the surrounding values. When recentering to a element associated field, all vertex
values incident to a element are averaged, and similarly when rencentering to a vertex associated field,
all element values incident to the vertex are averaged. If a field is already of the desired associated, then
the nothing is done and the field is simply passed through the filter. Note: ghost zones must be available when
the data set has more than one domain. Without ghost, the averaging will not be smooth across domain boundaries.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "recenter";
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "braid";         // name of the vector field
  params["association"] = "vertex";   // output field association
  // or params["association"] = "element";   // output field association

Uniform Grid
~~~~~~~~~~~~~~~~~~~~~
Uniform Grid filter changes the coordinate system of the input mesh to that of the user-specified regular mesh. Input fields are transferred by sampling the data at the vertex locations of the output geometry. For the output geometry, users must specify the field (`field`) to be sampled, and have the option to specify the origin (`origin`), the number of points along each axis (`dims`) from the origin, and the spacing between these points (`spacing`). 

For distributed data, the final output of this filter is composited on the root process, and ties for sampled points are handled by taking the average of all valid values.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "sample_grid";
  //params optional
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "dist";   //required
  params["origin/x"] = 0.0;   //default: minimum point in x dim
  params["origin/y"] = 0.0;   //default: minimum point in y dim
  params["origin/z"] = 0.0;   //default: minimum point in z dim
  params["dims/i"] = 10.0;    //default: x extents
  params["dims/j"] = 10.0;    //default: y extents
  params["dims/k"] = 10.0;    //default: z extents
  params["spacing/dx"] = 1.0; //default: 1.0
  params["spacing/dy"] = 1.0; //default: 1.0
  params["spacing/dz"] = 1.0; //default: 1.0
  //field value for sampled points outside of input mesh
  params["invalid_value"] = -100.0; //default: 0.0

Gradient
~~~~~~~~
Computes the gradient of a vertex-centered input field for every element
in the input data set. Fields will be automaticall recentered if they
are elemenet-centered.
The gradient computation can either generate cell center based gradients,
which are fast but less accurate, or more accurate but slower
point based gradients (default).

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "gradient";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "velocity";          // (required)
  params["output_name"] = "my_grad";     // (required) name of the output field
  params["use_cell_gradient"] = "false"; // (optional)


Vorticity
~~~~~~~~~
Computes the vorticity of a vertex-centered input field for every element
in the input data set. Fields will be automaticall recentered if they
are elemenet-centered.
The vorticity computation (based on the gradient) can either generate
cell center based gradients, which are fast but less accurate, or more
accurate but slower point based gradients (default).

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "vorticity";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "velocity";          // (required)
  params["output_name"] = "my_vorticity";// (required) name of the output field
  params["use_cell_gradient"] = "false"; // (optional)

Q-Criterion
~~~~~~~~~~~
Computes the qcriterion of a vertex-centered input field for every element
in the input data set. Fields will be automaticall recentered if they
are elemenet-centered.
The qcriterion computation (based on the gradient) can either generate
cell center based gradients, which are fast but less accurate, or more
accurate but slower point based gradients (default).

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "qcriterion";
  // filter knobs (all these are optional)
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["field"] = "velocity";          // (required)
  params["output_name"] = "my_q";        // (required) name of the output field
  params["use_cell_gradient"] = "false"; // (optional)

Material Interface Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Material Interface Reconstruction (MIR) filter can be used to represent and track the interfaces between different materials or phases in a multiphase flow. 
The user must specify the name of the material set (`matset`) they wish to use.  
Optional parameters include error scaling (`error_scaling`), scaling decay (`scaling_decay`), maximum iterations (`iterations`), and maximum error percentage (`max_error`).
The output field of the MIR Filter will be the name of the material set and can be used further in scenes or pipelines. 

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "mir";
  //params optional
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["matset"] = "matset";   //required
  params["error_scaling"] = 0.2; //default: 0.0
  params["scaling_decay"] = 1.0; //default: 0.0
  params["iterations"] = 8;      //default: 0
  params["max_error"] = 0.00001; //default: 0.00001

Add MPI Ranks as Field Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ascent provides a filter to add MPI rank information to a mesh. 
If the input data has multiple topolgies, the user must specify which topology to add the new field to.
The user also has the option of specifying the output name for the new field.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "add_mpi_ranks";
  //params optional
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["output"] = "ranks";//default: "mpi_ranks"
  params["topology"] = "topo";   //required if data has multiple topologies

.. _addmpiranks:

..  figure:: ../images/add_mpi_ranks.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of MPI ranks. 

Add Domain IDs as Field Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ascent provides a filter to add domain ID information to a mesh. 
If the input data has multiple topolgies, the user must specify which topology to add the new field to.  
The user also has the option of specifying the output name for the new field.

.. code-block:: c++

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "add_domain_ids";
  //params optional
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["output"] = "domain_ids";//default: "domain_ids"
  params["topology"] = "topo";   //required if data has multiple topologies

.. _adddomainids:

..  figure:: ../images/add_domain_ids.png
    :scale: 50 %
    :align: center

    An example of creating a pseudocolor plot of domain IDs. 

Partitioning
~~~~~~~~~~~~
Partitioning meshes is commonly needed in order to evenly distribute work
among many simulation ranks. Ascent utilizes the ``partition()`` functions provided from Conduit::Blueprint. Blueprint provides two ``partition()`` functions
that can be used to split or recombine Blueprint meshes in serial or parallel.
Full M:N repartioning is supported. The ``partition()`` functions are in the
serial and parallel Blueprint libraries, respectively.
Funtionality and further descriptions of optional parameters can be found in the Conduit::Blueprint `documentation <https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh_partition.html#partitioning>`_.

.. code:: cpp

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "partition";
  //params optional
  pipelines["pl1/f1/params/target"] = 1;
  pipelines["pl1/f1/params/fields"].append() = "pink";
  pipelines["pl1/f1/params/fields"].append() = "pony";
  pipelines["pl1/f1/params/fields"].append() = "club";
  pipelines["pl1/f1/params/merge_tolerance"] = 0.000001;
  pipelines["pl1/f1/params/mapping"] = 0; //turns off; on by default
  pipelines["pl1/f1/params/build_adjsets"] = 1;

   


In parallel, the Partition filter will make an effort to redistribute data across MPI
ranks to attempt to balance how data are assigned. Domains produced from selections
are assigned round-robin across ranks from rank 0 through rank N-1 until all
domains have been assigned. This assignment is carried out after extracting
selections locally so they can be restributed among ranks
before being combined into the target number of domains.


.. figure:: ../images/partition.png
    :width: 800px
    :align: center

    Partition used to re-partition a 7 domain mesh (left) to different target numbers of domains and to isolate logical subsets.


Options
+++++++
The Partition filter accepts optional parameters. 
If no optional parameters are given, each input mesh
domain will be fully selected. 
If no ``selections`` are specifed as ``params`` then the partitioner
will create selections of an appropriate type that selects all elements in each
input domain.

The ``target`` parameter is useful for setting the target number of domains in the
final output mesh. If the target value is larger than the number of input domains
or selections then the mesh will be split to achieve that target number of domains.
This may require further subdividing selections. Alternatively, if the target is
smaller than the number of selections then the selections will be combined to
yield the target number of domains. The combining is done such that smaller element
count domains are combined first. 
Additionally, Ascent provides an optional boolean parameter, ``distributed``, which dictates if the number 
of chosen target domains is applied across ranks (``true``, default), or to each rank individually (``false``).

.. tabularcolumns:: |p{1.5cm}|p{4cm}|L|

+---------------------+-----------------------------------------+------------------------------------------+
| **Option**          | **Description**                         | **Example**                              |
+---------------------+-----------------------------------------+------------------------------------------+
| selections          | A list of selection objects that        | .. code:: yaml                           |
|                     | identify regions of interest from the   |                                          |
|                     | input domains. Selections can be        |    selections:                           |
|                     | different on each MPI rank.             |      type: "logical"                     |
|                     |                                         |      start: [0,0,0]                      |
|                     |                                         |      end: [9,9,9]                        |
|                     |                                         |      domain_id: 10                       |
+---------------------+-----------------------------------------+------------------------------------------+
| target              | An optional integer that determines the | .. code:: yaml                           |
|                     | fields containing original domains and  |                                          |
|                     | number of domains in the output. If     |    target: 4                             |
|                     | given, the value must be greater than 0.|                                          |
|                     | Values larger than the number of        |                                          |
|                     | selections cause domains to be split.   |                                          |
|                     | Values smaller than the number of       |                                          |
|                     | selections cause domains to be combined.|                                          |
|                     | Invalid values are ignored.             |                                          |
|                     |                                         |                                          |
|                     | If not given, the output will contain   |                                          |
|                     | the number of selections. In parallel,  |                                          |
|                     | the largest target value from the ranks |                                          |
|                     | will be used for all ranks.             |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| fields              | An list of strings that indicate the    | .. code:: yaml                           |
|                     | names of the fields to extract in the   |                                          |
|                     | output. If this option is not provided, |    fields: ["dist", "pressure"]          |
|                     | all fields will be extracted.           |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| mapping             | An integer that determines whether      | .. code:: yaml                           |
|                     | fields containing original domains and  |                                          |
|                     | ids will be added in the output. These  |    mapping: 0                            |
|                     | fields enable one to know where each    |                                          |
|                     | vertex and element came from originally.|                                          |
|                     | Mapping is on by default. A non-zero    |                                          |
|                     | value turns it on and a zero value turns|                                          |
|                     | it off.                                 |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| merge_tolerance     | A double value that indicates the max   | .. code:: yaml                           |
|                     | allowable distance between 2 points     |                                          |
|                     | before they are considered to be        |    merge_tolerance: 0.000001             |
|                     | separate. 2 points spaced smaller than  |                                          |
|                     | this distance will be merged when       |                                          |
|                     | explicit coordsets are combined.        |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| distributed         | An optional boolean value for parallel  | .. code:: yaml                           |
|                     | execution. If true, the chosen number   |                                          |
|                     | of target domains will be applied       |    distributed: "false"                  |
|                     | across all ranks. If false, the chosen  |                                          |
|                     | number of target domains will be        |                                          |
|                     | applied to each rank individually.      |                                          |
|                     |                                         |                                          |
|                     | If not given, the default is true.      |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| build_adjsets       | An integer that determines whether      | .. code:: yaml                           |
|                     | the partitioner should build adjsets,   |                                          |
|                     | if they are present in the selected     |    build_adjsets: 1                      |
|                     | topology.                               |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| original_element_ids| A string value that provides desired    | .. code::yaml                            |
|                     | field name used to contain original     |                                          |
|                     | element ids created from partitioning.  |    original_element_ids: "elem_name"     |
|                     | The default value is                    |                                          |
|                     | original_element_ids.                   |                                          |
+---------------------+-----------------------------------------+------------------------------------------+
| original_vertex_ids | A string value that provides desired    | .. code::yaml                            |
|                     | field name used to contain original     |                                          |
|                     | vertex ids created from partitioning.   |    original_vertex_ids: "vert_name"      |
|                     | The default value is                    |                                          |
|                     | original_vertex_ids.                    |                                          |
+---------------------+-----------------------------------------+------------------------------------------+


Selections
++++++++++
Selections can be specified in the options for the Partition Filter to
select regions of interest that will participate in mesh partitioning. If
selections are not used then all elements from the input meshes will be
selected to partitipate in the partitioning process. Selections can be further
subdivided if needed to arrive at the target number of domains. Selections can
target specific domains and topologies as well. If a selection does not apply
to the input mesh domains then no geometry is produced in the output for that
selection.

The Partition filter supports 4 types of selections:

.. tabularcolumns:: |p{1.5cm}|p{2cm}|L|

=============== =============================== =============================================
Selection Type  Topologies                      Description
=============== =============================== =============================================
logical          uniform,rectilinear,structured Identifies start and end logical IJK ranges to select sub-bricks of uniform, rectilinear, or structured topologies. This selection is not compatible with other topologies.
explicit         all                            Identifies an explicit list of element ids and it works with all topologies.
ranges           all                            Identifies ranges of element ids, provided as pairs so the user can select multiple contiguous blocks of elements. This selection works with all topologies
field            all                            Uses a specified field to indicate destination domain for each element.
=============== =============================== =============================================

By default, a selection does not apply to any specific domain_id. A list of
selections applied to a single input mesh will extract multiple new domains from
that original input mesh. Since meshes are composed of many domains in practice,
selections can also be associated with certain domain_id values. Selections that
provide a domain_id value will only match domains that either have a matching
state/domain_id value or match its index in the input node's list of children
(if state/domain_id is not present).

Selections can apply to certain topology names as well. By default, the first
topology is used but if the ``topology`` name is provided then the selection will
operate on the specified topology only.


+------------------+-----------------------------------------+------------------------------------------+
| **Option**       | **Description**                         | **Example**                              |
+------------------+-----------------------------------------+------------------------------------------+
| type             | The selection type                      | .. code:: yaml                           |
|                  |                                         |                                          |
|                  |                                         |    selections:                           |
|                  |                                         |       type: "logical"                    |
+------------------+-----------------------------------------+------------------------------------------+
| domain_id        | The domain_id to which the selection    | .. code:: yaml                           |
|                  | will apply. This is almost always an    |                                          |
|                  | unsigned integer value.                 |    selections:                           |
|                  |                                         |       type: "logical"                    |
|                  |                                         |       domain_id: 10                      |
|                  |                                         |                                          |
|                  |                                         | .. code:: yaml                           |
|                  |                                         |                                          |
|                  | For field selections, domain_id is      |    selections:                           |
|                  | allowed to be a string "any" so a single|       type: "logical"                    |
|                  | selection can apply to many domains.    |       domain_id = "any"                  |
|                  |                                         |                                          |
|                  |                                         |                                          |
+------------------+-----------------------------------------+------------------------------------------+
| topology         | The topology to which the selection     | .. code:: yaml                           |
|                  | will apply.                             |                                          |
|                  |                                         |    selections:                           |
|                  |                                         |       type: "logical"                    |
|                  |                                         |       domain_id: 10                      |
|                  |                                         |       topology: "mesh"                   |
|                  |                                         |                                          |
+------------------+-----------------------------------------+------------------------------------------+

Logical Selection
*****************
The logical selection allows the partitioner to extract a logical IJK subset from uniform, rectilinear,
or structured topologies. The selection is given as IJK start and end values. If the end values extend
beyond the actual mesh's logical extents, they will be clipped. The partitioner may
automatically subdivide logical selections into smaller logical selections, if needed,
preserving the logical structure of the input topology into the output.

.. code:: cpp

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "partition";
  //params optional
  pipelines["pl1/f1/params/selections/type"] = "logical";
  const float start[3] = {0,0,0};
  const float end[3] = {10,10,10};
  pipelines["pl1/f1/params/selections/start"].set(start,3); 
  pipelines["pl1/f1/params/selections/end"].set(end,3); 

Explicit Selection
******************
The explicit selection allows the partitioner to extract a list of elements.
This is used when the user wants to target a specific set of elements.
The output will result in an explicit topology.

.. code:: cpp

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "partition";
  //params optional
  pipelines["pl1/f1/params/selections/type"] = "explicit";
  const int elements[6] = [0,1,2,3,100,101,102]; 
  pipelines["pl1/f1/params/selections/elements"].set(elements,6); 


Ranges Selection
****************
The ranges selection is similar to the explicit selection except that it identifies
ranges of elements using pairs of numbers. The list of ranges must be a multiple of
2 in length. The output will result in an explicit topology.

.. code:: cpp

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "partition";
  //params optional
  pipelines["pl1/f1/params/selections/type"] = "ranges";
  const int elements[4] = [0,3,100,102]; 
  pipelines["pl1/f1/params/selections/elements"].set(elements,4); 


Field Selection
***************
The field selection enables the partitioner to use partitions done by other tools
using a field on the mesh as the source of the final domain number for each element.
The field must be associated with the mesh elements. When using a field selection,
the partitioner will make a best attempt to use the domain numbers to extract
mesh pieces and reassemble them into domains with those numberings. If a larger
target value is specified, then field selections can sometimes be partitioned further
as explicit partitions. The field selection is unique in that its ``domain_id`` value
can be set to "any" if it is desired that the field selection will be applied to
all domains in the input mesh. The domain_id value can still be set to specific
integer values to limit the set of domains over which the selection will be applied.

+------------------+-----------------------------------------+------------------------------------------+
| **Option**       | **Description**                         | **Example**                              |
+------------------+-----------------------------------------+------------------------------------------+
| field            | The name of the element field that will | .. code:: yaml                           |
|                  | be used for partitioning. The field     |                                          |
|                  | shall contain non-negative domain       |    selections:                           |
|                  | numbers.                                |       type: "field"                      |
|                  |                                         |       domain_id: "any"                   |
|                  |                                         |                                          |
+------------------+-----------------------------------------+------------------------------------------+
| destination_ranks| An optional list of integers            | .. code:: yaml                           |
|                  | representing the MPI rank where the     |                                          |
|                  | domain will be sent after partitioning. |    selections:                           |
|                  | This option can help ensure domains for |       type: "field"                      |
|                  | topologies partitioned via multiple     |       field: "albatraoz"                 |
|                  | calls to partition() end up together on |       domain_id: "any"                   |
|                  | a target MPI rank. The example shows    |       destination_ranks: [0,1,2,3]       |
|                  | domain 0 going to MPI rank 0 and so on. |                                          |
|                  |                                         |                                          |
+------------------+-----------------------------------------+------------------------------------------+

.. code:: cpp

  conduit::Node pipelines;
  // pipeline 1
  pipelines["pl1/f1/type"] = "partition";
  //params optional
  pipelines["pl1/f1/params/selections/type"] = "field";
  pipelines["pl1/f1/params/selections/domain_id"] = "any";
  pipelines["pl1/f1/params/selections/field"] = "padam_padam"; 

