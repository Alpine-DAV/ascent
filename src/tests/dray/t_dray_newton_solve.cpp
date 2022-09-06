// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/newton_solver.hpp>

#include <dray/camera.hpp>
#include <dray/filters/isosurface.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/math.hpp>


TEST (dray_test, dray_newton_solve)
{
  // Set up the mesh / field.

  // For this test we will use the R3->R3 transformation of {ref space} -> {phys space}.

  // There are two quadratic unit-cubes, adjacent along X, sharing a face in the YZ plane.
  // There are 45 total control points: 2 vol mids, 11 face mids, 20 edge mids, and 12 vertices.

  // 2 elts, 27 el_dofs, supply instance of ShType, 45 total control points.
  dray::GridFunction<3> eltrans_space;
  dray::GridFunction<1> eltrans_field;
  eltrans_space.resize (2, 27, 45);
  eltrans_field.resize (2, 27, 45);

  // Scalar field values of control points.
  float grid_vals[45] = {
    10, -10, // 0..1 vol mids A and B
    15, 7,   7,  7,  7,   0,   -15, -7,  -7,
    -7, -7, // 2..12 face mids A(+X,+Y,+Z,-Y,-Z) AB B(-X,+Y,+Z,-Y,-Z)
    12, 12,  12, 12, -12, -12, -12, -12, // 13..20 edge mids on ends +X/-X A(+Y,+Z,-Y,-Z) B(+Y,+Z,-Y,-Z)
    5,  5,   5,  5,  -5,  -5,  -5,  -5, // 21..28 edge mids YZ corners A(++,-+,--,+-) B(++,-+,--,+-)
    0,  0,   0,  0, // 29..32 edge mids on shared face AB(+Y,+Z,-Y,-Z)
    20, 20,  20, 20, -20, -20, -20, -20, // 33..40 vertices on ends +X/-X, YZ corners A(++,-+,--,+-) B(++,-+,--,+-)
    0,  0,   0,  0
  }; // 41..44 vertices on shared face, YZ corners AB(++,-+,--,+-)

  // Physical space locations of control points. (Non-deformed cubes).
  float grid_loc[3 * 45] = { .5,  .5, .5, // 0
                             -.5, .5, .5, // 1

                             1,   .5, .5, // 2
                             .5,  1,  .5, // 3
                             .5,  .5, 1, // 4
                             .5,  0,  .5, // 5
                             .5,  .5, 0, // 6

                             0,   .5, .5, // 7

                             -1,  .5, .5, // 8
                             -.5, 1,  .5, // 9
                             -.5, .5, 1, // 10
                             -.5, 0,  .5, // 11
                             -.5, .5, 0, // 12

                             1,   1,  .5, // 13
                             1,   .5, 1, // 14
                             1,   0,  .5, // 15
                             1,   .5, 0, // 16

                             -1,  1,  .5, // 17
                             -1,  .5, 1, // 18
                             -1,  0,  .5, // 19
                             -1,  .5, 0, // 20

                             .5,  1,  1, // 21
                             .5,  0,  1, // 22
                             .5,  0,  0, // 23
                             .5,  1,  0, // 24

                             -.5, 1,  1, // 25
                             -.5, 0,  1, // 26
                             -.5, 0,  0, // 27
                             -.5, 1,  0, // 28

                             0,   1,  .5, // 29
                             0,   .5, 1, // 30
                             0,   0,  .5, // 31
                             0,   .5, 0, // 32

                             1,   1,  1, // 33
                             1,   0,  1, // 34
                             1,   0,  0, // 35
                             1,   1,  0, // 36

                             -1,  1,  1, // 37
                             -1,  0,  1, // 38
                             -1,  0,  0, // 39
                             -1,  1,  0, // 40

                             0,   1,  1, // 41
                             0,   0,  1, // 42
                             0,   0,  0, // 43
                             0,   1,  0 }; // 44


  // Map the per-element degrees of freedom into the total set of control points.
  int ctrl_idx[54];
  int *const ax = ctrl_idx, *const bx = ctrl_idx + 27;

  // Nonshared nodes.
  ax[13] = 0;
  bx[13] = 1;

  ax[22] = 2;
  bx[4] = 8;
  ax[16] = 3;
  bx[16] = 9;
  ax[14] = 4;
  bx[14] = 10;
  ax[10] = 5;
  bx[10] = 11;
  ax[12] = 6;
  bx[12] = 12;

  ax[25] = 13;
  bx[7] = 17;
  ax[23] = 14;
  bx[5] = 18;
  ax[19] = 15;
  bx[1] = 19;
  ax[21] = 16;
  bx[3] = 20;

  ax[17] = 21;
  bx[17] = 25;
  ax[11] = 22;
  bx[11] = 26;
  ax[9] = 23;
  bx[9] = 27;
  ax[15] = 24;
  bx[15] = 28;

  ax[26] = 33;
  bx[8] = 37;
  ax[20] = 34;
  bx[2] = 38;
  ax[18] = 35;
  bx[0] = 39;
  ax[24] = 36;
  bx[6] = 40;

  // Shared nodes.
  ax[4] = bx[22] = 7;

  ax[7] = bx[25] = 29;
  ax[5] = bx[23] = 30;
  ax[1] = bx[19] = 31;
  ax[3] = bx[21] = 32;

  ax[8] = bx[26] = 41;
  ax[2] = bx[20] = 42;
  ax[0] = bx[18] = 43;
  ax[6] = bx[24] = 44;

  // Initialize eltrans space and field with these values.
  memcpy (eltrans_field.m_ctrl_idx.get_host_ptr (), ctrl_idx, 54 * sizeof (int));
  memcpy (eltrans_space.m_ctrl_idx.get_host_ptr (), ctrl_idx, 54 * sizeof (int));
  memcpy (eltrans_field.m_values.get_host_ptr (), grid_vals, 45 * sizeof (float)); // scalar field values
  memcpy (eltrans_space.m_values.get_host_ptr (), grid_loc, 3 * 45 * sizeof (float)); // space locations

  {

    using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
    using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

    dray::Mesh<MeshElemT> mesh (eltrans_space, 2);
    dray::Field<FieldElemT> field (eltrans_field, 2);

    dray::DataSet<MeshElemT> dataset (mesh);
    dataset.add_field (field, "bananas");

    constexpr int c_width = 1024;
    constexpr int c_height = 1024;

    //
    // Use camera to generate rays and points.
    //
    dray::Camera camera;
    camera.set_width (c_width);
    camera.set_height (c_height);
    camera.set_up (dray::make_vec3f (0, 0, 1));
    camera.set_pos (dray::make_vec3f (3.2, 4.3, 3));
    camera.set_look_at (dray::make_vec3f (0, 0, 0));
    // camera.reset_to_bounds(mesh_field.get_bounds());
    dray::Array<dray::Ray> rays;
    camera.create_rays (rays);

    dray::ColorTable color_table ("cool2warm");

    std::string output_path = prepare_output_dir ();
    // Output isosurface, colorized by field spatial gradient magnitude.
    {
      float isovalues[5] = { 15, 8, 0, -8, -15 };
      const char *filenames[5] = { "isosurface_+15", "isosurface_+08", "isosurface__00",
                                   "isosurface_-08", "isosurface_-15" };

      for (int iso_idx = 0; iso_idx < 5; iso_idx++)
      {
        dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());
        std::string output_file =
        conduit::utils::join_file_path (output_path, std::string (filenames[iso_idx]));
        remove_test_image (output_file);


        dray::Isosurface isosurface;
        isosurface.set_field ("bananas");
        isosurface.set_color_table (color_table);
        isosurface.set_iso_value (isovalues[iso_idx]);
        isosurface.execute (dataset, rays, framebuffer);

        framebuffer.save (output_file);

        EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
        printf ("Finished rendering isosurface idx %d\n", iso_idx);
      }
    }
  }
}
