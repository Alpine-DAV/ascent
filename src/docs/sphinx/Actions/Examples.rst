.. _yaml-examples:

Ascent Actions Examples
=======================

An example of the contour filter with a single iso value.
----------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_single_contour_3d100.yaml

Resulting image:

.. image:: examples/tout_single_contour_3d100.png

An example of rendering a point field with constant radius.
------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_points_const_radius100.yaml

Resulting image:

.. image:: examples/tout_render_3d_points_const_radius100.png

An example of creating a mesh plot.
------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_mesh100.yaml

Resulting image:

.. image:: examples/tout_render_3d_mesh100.png

An example of using the volume (unstructured grid) extract.
------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_rover_volume100.yaml

Resulting image:

.. image:: examples/tout_rover_volume100.png

An example if using the vector component filter  to extract a scalar component of a vector field.
--------------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_vector_component100.yaml

Resulting image:

.. image:: examples/tout_vector_component100.png

An example of creating a render, specifying all camera parameters.
-------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/render_0100.yaml

Resulting image:

.. image:: examples/render_0100.png

An example rendering a 2d field.
---------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_2d_default_runtime100.yaml

Resulting image:

.. image:: examples/tout_render_2d_default_runtime100.png

An example of using the log filter.
------------------------------------

YAML actions:

.. literalinclude:: examples/tout_log_field100.yaml

Resulting image:

.. image:: examples/tout_log_field100.png

An example of rendering with no background (alpha channel = 0)
---------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_no_bg100.yaml

Resulting image:

.. image:: examples/tout_render_3d_no_bg100.png

An example of changing the azimuth of the camera.
--------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_azimuth100.yaml

Resulting image:

.. image:: examples/tout_render_3d_azimuth100.png

An example of the contour filter with a number of evenly spaced levels.
------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_multi_contour_levels100.yaml

Resulting image:

.. image:: examples/tout_multi_contour_levels100.png

An example an inverted sphere clip using a center and radius
-------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_clip_inverted_sphere100.yaml

Resulting image:

.. image:: examples/tout_clip_inverted_sphere100.png

An example of creating a transfer function for volume rendering.
-----------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_multi_default_runtime100.yaml

Resulting image:

.. image:: examples/tout_render_3d_multi_default_runtime100.png

An example of the interconnecting pipelines.
---------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_pipelines100.yaml

Resulting image:

.. image:: examples/tout_pipelines100.png

An example of using the gradient filter and plotting the magnitude.
--------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_vorticity_vel100.yaml

Resulting image:

.. image:: examples/tout_vorticity_vel100.png

An example of the three slice filter.
--------------------------------------

YAML actions:

.. literalinclude:: examples/tout_3slice_3d100.yaml

Resulting image:

.. image:: examples/tout_3slice_3d100.png

An example of using the volume (unstructured grid) extract with min and max values.
------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_rover_volume_min_max100.yaml

Resulting image:

.. image:: examples/tout_rover_volume_min_max100.png

An example of using the gradient filter on a element centered fieldand plotting the magnitude.
-----------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_gradient_mag_radial100.yaml

Resulting image:

.. image:: examples/tout_gradient_mag_radial100.png

An example of the contour filter with a multiple iso values.
-------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_multi_contour_3d100.yaml

Resulting image:

.. image:: examples/tout_multi_contour_3d100.png

An example of the slice filter with a single plane.
----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_slice_3d100.yaml

Resulting image:

.. image:: examples/tout_slice_3d100.png

An example of creating a mesh plot of a contour.
-------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_multi_mesh100.yaml

Resulting image:

.. image:: examples/tout_render_3d_multi_mesh100.png

An example of using inverted clip with field.
----------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_clip_with_field_inverted100.yaml

Resulting image:

.. image:: examples/tout_clip_with_field_inverted100.png

An example if using the vector magnitude filter.
-------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_vec_mag100.yaml

Resulting image:

.. image:: examples/tout_vec_mag100.png

An example of using the log filter and clamping the min value. This can help when there are negative values present.
---------------------------------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_log_field_clamp100.yaml

Resulting image:

.. image:: examples/tout_log_field_clamp100.png

An example of the slice filter with a single plane (off-axis).
---------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_slice_3d_off_axis100.yaml

Resulting image:

.. image:: examples/tout_slice_3d_off_axis100.png

An example of the slice filter with a single plane.
----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_exaslice_3d100.yaml

Resulting image:

.. image:: examples/tout_exaslice_3d100.png

An example of creating a plot specifying the min and max values of the scalar range.
-------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_ascent_min_max100.yaml

Resulting image:

.. image:: examples/tout_render_3d_ascent_min_max100.png

An example a plane clip defined with a point and a normal
----------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_clip_plane100.yaml

Resulting image:

.. image:: examples/tout_clip_plane100.png

An example of using clip with field.
-------------------------------------

YAML actions:

.. literalinclude:: examples/tout_clip_with_field100.yaml

Resulting image:

.. image:: examples/tout_clip_with_field100.png

An example if using the re-center filter (to vertex).
------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_recenter_element100.yaml

Resulting image:

.. image:: examples/tout_recenter_element100.png

An example of rendering with no annotations.
---------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_no_annotations100.yaml

Resulting image:

.. image:: examples/tout_render_3d_no_annotations100.png

An example of rendering custom background and foreground colors.
-----------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_bg_fg_colors100.yaml

Resulting image:

.. image:: examples/tout_render_3d_bg_fg_colors100.png

An example of using the isovolume filter.
------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_iso_volume100.yaml

Resulting image:

.. image:: examples/tout_iso_volume100.png

An example if using the re-center filter (to element).
-------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_recenter_vertex100.yaml

Resulting image:

.. image:: examples/tout_recenter_vertex100.png

An example of using the gradient filter and plotting the magnitude.
--------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_divergence_vel100.yaml

Resulting image:

.. image:: examples/tout_divergence_vel100.png

An example of the slice filter with a single plane.
----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_slice_offset_3d100.yaml

Resulting image:

.. image:: examples/tout_slice_offset_3d100.png

An example of using the gradient filter and plotting the magnitude.
--------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_qcriterion_vel100.yaml

Resulting image:

.. image:: examples/tout_qcriterion_vel100.png

Example of rendering multiple topologies
-----------------------------------------

YAML actions:

.. literalinclude:: examples/tout_multi_topo100.yaml

Resulting image:

.. image:: examples/tout_multi_topo100.png

An example of creating a render specifying the image size.
-----------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_domain_overload100.yaml

Resulting image:

.. image:: examples/tout_render_3d_domain_overload100.png

An example a blox clip
-----------------------

YAML actions:

.. literalinclude:: examples/tout_clip_box100.yaml

Resulting image:

.. image:: examples/tout_clip_box100.png

An example of creating a custom color map.
-------------------------------------------

YAML actions:

.. literalinclude:: examples/milk_chocolate100.yaml

Resulting image:

.. image:: examples/milk_chocolate100.png

An example if using the composite vector filter  to compose three scalar fields into a vector.
-----------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_composite_vector100.yaml

Resulting image:

.. image:: examples/tout_composite_vector100.png

An example a sphere clip using a center and radius
---------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_clip_sphere100.yaml

Resulting image:

.. image:: examples/tout_clip_sphere100.png

Example of adding 1 ghost field with 2 topologies
--------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_multi_topo_single_ghost100.yaml

Resulting image:

.. image:: examples/tout_multi_topo_single_ghost100.png

An example of using the gradient filter and plotting the magnitude.
--------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_cell_gradient_mag_braid100.yaml

Resulting image:

.. image:: examples/tout_cell_gradient_mag_braid100.png

An example of rendering a point field with variable radius.
------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_points_variable_radius100.yaml

Resulting image:

.. image:: examples/tout_render_3d_points_variable_radius100.png

An example of using the threshold filter.
------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_threshold_3d100.yaml

Resulting image:

.. image:: examples/tout_threshold_3d100.png

A more complex trigger example using several functions that evaluate positons on the mesh.
-------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_complex_trigger_actions100.yaml

Resulting image:

.. image:: examples/tout_complex_trigger_actions100.png

Example of adding multple ghosts with 2 topologies
---------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_multi_topo_ghosts100.yaml

Resulting image:

.. image:: examples/tout_multi_topo_ghosts100.png

An example of using the gradient filter and plotting the magnitude.
--------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_gradient_mag_braid100.yaml

Resulting image:

.. image:: examples/tout_gradient_mag_braid100.png

An example of using the gradient filter using cell gradients on a element centered field and plotting the magnitude.
---------------------------------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_cell_gradient_mag_radial100.yaml

Resulting image:

.. image:: examples/tout_cell_gradient_mag_radial100.png

An example of using the xray extract.
--------------------------------------

YAML actions:

.. literalinclude:: examples/tout_rover_xray_params100.yaml

An example of using a relay extract to save a subset of the data.
------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_relay_serial_extract_subset100.yaml

A more complex trigger example using several functions that evaluate positons on the mesh.
-------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_trigger_extract100.yaml

An example of using the xray extract.
--------------------------------------

YAML actions:

.. literalinclude:: examples/tout_rover_xray100.yaml

An example of rendering to a filename using format specifiers.
---------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_render_3d_name_format100.yaml

An example of quiering the current cycle.
------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_cycle_query100.yaml

An example of using devil ray for pseudocolor plot.
----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_dray_3slice100.yaml

An example of using devil ray for pseudocolor plot.
----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_dray_volume100.yaml

An example of using devil ray for pseudocolor plot.
----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_dray_surface100.yaml

An example of using an relay extract to save the results of  a pipeline to the file system.
--------------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_hd5f_iso100.yaml

An example of using an relay extract to save the published mesh to the file system.
------------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_hd5f_mesh100.yaml

An example of quiering the maximum value of a field from the result of a pipeline.
-----------------------------------------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_max_pipeline_query100.yaml


-

YAML actions:

.. literalinclude:: examples/tout_render_actions.yaml

An example of quiering the maximum value of a field.
-----------------------------------------------------

YAML actions:

.. literalinclude:: examples/tout_max_query100.yaml

