// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/mfem_reader.hpp>

#include <dray/camera.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


// TEST()
//
TEST (dray_volume_render, dray_volume_render_simple)
{
  std::string file_name = std::string (ASCENT_T_DATA_DIR) + "taylor_green/Laghos";

  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;
  auto dataset = dray::MFEMReader::load32 (file_name);

  dray::Mesh<float32, MeshElemT> mesh = dataset.get_mesh ();
  dray::MeshAccess<float32, MeshElemT> host_mesh = mesh.access_host_mesh ();
  const dray::AABB<2> face_ref_box = dray::AABB<2>::ref_universe ();
  dray::AABB<3> bounds;
  const int num_elements = mesh.get_num_elem ();

  // Print out bounds and sub-bounds for each face of each element.
  /// for (int e = 0; e < num_elements; e++)
  for (int e = 0; e < 1; e++)
  {
    for (int f = 0; f < 6; f++)
    {
      fprintf (stdout, "Elem %2d face %2d \t", e, f);
      bounds.reset ();
      host_mesh.get_elem (e).get_face_element (f).get_bounds (bounds);
      std::cout << "bounds: " << bounds << " \t";

      bounds.reset ();
      host_mesh.get_elem (e).get_face_element (f).get_sub_bounds (face_ref_box.m_ranges, bounds);
      std::cout << "bounds: " << bounds << " \n";
    }
    std::cout << "\n";
  }
}
