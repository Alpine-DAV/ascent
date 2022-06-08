// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/Element/bernstein_basis.hpp>
#include <dray/filters/isosurface.hpp>
#include <dray/newton_solver.hpp>

#include <dray/camera.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/math.hpp>
#include <dray/shaders.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

#include <stdlib.h>


TEST (dray_cool_shapes, dray_newton_solve)
{
  // Single tri-quadratic hex element with smooth edges.
  //
  // DOFs listed as corners, edges, faces, interior.
  float smooth_quad_loc[3 * 27] = {
    0,    0,    0, // 0    0
    0,    0,    1, // 2    1
    0,    1,    0, // 6    2
    0,    1,    1, // 8    3
    1,    0,    0, // 18   4
    1,    0,    1, // 20   5
    1,    1,    0, // 24   6
    1,    1,    1, // 26   7

    -.25, -.25, .5, // 1    8
    -.25, .5,   -.25, // 3    9
    -.25, .5,   1.25, // 5    10
    -.25, 1.25, .5, // 7    11
    .5,   -.25, -.25, // 9    12
    .5,   -.25, 1.25, // 11   13
    .5,   1.25, -.25, // 15   14
    .5,   1.25, 1.25, // 17   15
    1.25, -.25, .5, // 19   16
    1.25, .5,   -.25, // 21   17
    1.25, .5,   1.25, // 23   18
    1.25, 1.25, .5, // 25   19

    -1,   .5,   .5, // 4    20
    .5,   -1,   .5, // 10   21
    .5,   .5,   -1, // 12   22
    .5,   .5,   2, // 14   23
    .5,   2,    .5, // 16   24
    2,    .5,   .5, // 22   25

    .5,   .5,   .5 // 13   26
  };

  float smooth_quad_field[27] = { 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0 };

  int smooth_quad_ctrl_idx_inv[27] = { 0,  2,  6, 8,  18, 20, 24, 26, 1,
                                       3,  5,  7, 9,  11, 15, 17, 19, 21,
                                       23, 25, 4, 10, 12, 14, 16, 22, 13 };
  int smooth_quad_ctrl_idx[27];
  for (int ii = 0; ii < 27; ii++)
    smooth_quad_ctrl_idx[smooth_quad_ctrl_idx_inv[ii]] = ii;

  // Set up the mesh / field.
  dray::GridFunction<3> eltrans_space;
  dray::GridFunction<1> eltrans_field;
  eltrans_space.resize (1, 27, 27);
  eltrans_field.resize (1, 27, 27);

  // Initialize eltrans space and field with these values.
  memcpy (eltrans_field.m_ctrl_idx.get_host_ptr (), smooth_quad_ctrl_idx,
          27 * sizeof (int));
  memcpy (eltrans_space.m_ctrl_idx.get_host_ptr (), smooth_quad_ctrl_idx,
          27 * sizeof (int));
  memcpy (eltrans_field.m_values.get_host_ptr (), smooth_quad_field,
          27 * sizeof (float)); // scalar field values
  memcpy (eltrans_space.m_values.get_host_ptr (), smooth_quad_loc,
          3 * 27 * sizeof (float)); // space locations


  using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

  dray::Mesh<MeshElemT> mesh (eltrans_space, 2);
  dray::Field<FieldElemT> field (eltrans_field, 2);

  dray::DataSet<MeshElemT> dataset (mesh);
  dataset.add_field (field, "bananas");

  // -------------------


  // Camera.
  constexpr int c_width = 1024;
  constexpr int c_height = 1024;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.set_up (dray::make_vec3f (0, 0, 1));
  camera.set_pos (dray::make_vec3f (3.2, 4.3, 3));
  camera.set_look_at (dray::make_vec3f (0, 0, 0));
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  // Color tables.
  dray::ColorTable color_table1 ("ColdAndHot");
  const float alpha_hi = 0.10f;
  const float alpha_lo = 0.0f;
  color_table1.add_alpha (0.0000, alpha_hi);
  color_table1.add_alpha (0.0357, alpha_lo);
  color_table1.add_alpha (0.0714, alpha_hi);
  color_table1.add_alpha (0.1071, alpha_lo);
  color_table1.add_alpha (0.1429, alpha_hi);
  color_table1.add_alpha (0.1786, alpha_lo);
  color_table1.add_alpha (0.2143, alpha_hi);
  color_table1.add_alpha (0.2500, alpha_lo);
  color_table1.add_alpha (0.2857, alpha_hi);
  color_table1.add_alpha (0.3214, alpha_lo);
  color_table1.add_alpha (0.3571, alpha_hi);
  color_table1.add_alpha (0.3929, alpha_lo);
  color_table1.add_alpha (0.4286, alpha_hi);
  color_table1.add_alpha (0.4643, alpha_lo);
  color_table1.add_alpha (0.5000, alpha_hi);
  color_table1.add_alpha (0.5357, alpha_lo);
  color_table1.add_alpha (0.5714, alpha_hi);
  color_table1.add_alpha (0.6071, alpha_lo);
  color_table1.add_alpha (0.6429, alpha_hi);
  color_table1.add_alpha (0.6786, alpha_lo);
  color_table1.add_alpha (0.7143, alpha_hi);
  color_table1.add_alpha (0.7500, alpha_lo);
  color_table1.add_alpha (0.7857, alpha_hi);
  color_table1.add_alpha (0.8214, alpha_lo);
  color_table1.add_alpha (0.8571, alpha_hi);
  color_table1.add_alpha (0.8929, alpha_lo);
  color_table1.add_alpha (0.9286, alpha_hi);
  color_table1.add_alpha (0.9643, alpha_lo);
  color_table1.add_alpha (1.0000, alpha_hi);

  dray::ColorTable color_table2 ("ColdAndHot");
  color_table2.add_alpha (0.0000, 1.0f);
  color_table2.add_alpha (1.0000, 1.0f);

  // Lights.
  dray::PointLightSource light;
  light.m_pos = { 5.0f, 5.0f, 5.0f };
  light.m_amb = { 0.1f, 0.1f, 0.1f };
  light.m_diff = { 0.70f, 0.70f, 0.70f };
  light.m_spec = { 0.30f, 0.30f, 0.30f };
  light.m_spec_pow = 90.0;
  dray::Shader::set_light_properties (light);


  // Isosurface.
  {
    std::string output_path = prepare_output_dir ();
    std::string output_file =
    conduit::utils::join_file_path (output_path, "smooth_quad_iso");
    remove_test_image (output_file);
    remove_test_image (output_file + "_depth");

    dray::Shader::set_color_table (color_table1);
    dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

    const float isoval = 0.9;

    dray::Isosurface isosurface;
    isosurface.set_field ("bananas");
    isosurface.set_color_table (color_table1);
    isosurface.set_iso_value (isoval);
    isosurface.execute (dataset, rays, framebuffer);

    framebuffer.save (output_file);
    EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));

    framebuffer.save_depth (output_file + "_depth");
    EXPECT_TRUE (check_test_image (output_file + "_depth"));
  }
}
