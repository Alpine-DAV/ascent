// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>


#include <dray/filters/vector_component.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>
#include <dray/array_registry.hpp>

#include <fstream>
#include <stdlib.h>

//---------------------------------------------------------------------------//
bool
mfem_enabled()
{
#ifdef DRAY_MFEM_ENABLED
    return true;
#else
    return false;
#endif
}

//---------------------------------------------------------------------------//
TEST (dray_vector_ops, dray_vector_component)
{

  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }

  EXPECT_EQ(dray::ArrayRegistry::number_of_arrays(),0);
  EXPECT_EQ(dray::ArrayRegistry::host_usage(),0);
  EXPECT_EQ(dray::ArrayRegistry::device_usage(),0);
  dray::ArrayRegistry::summary();
  dray::stats::StatStore::clear();
  if(dray::stats::StatStore::stats_supported())
  {
    dray::stats::StatStore::enable_stats();
  }

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "vector_component");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(dataset);

  dray::VectorComponent vc;
  vc.field("velocity");
  vc.output_name("bananas_x");
  vc.component(0);
  faces = vc.execute(faces);

  dray::ColorTable color_table ("Spectral");

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("bananas_x");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  fb.save(output_file);
  // note: dray diff tolerance was 0.2f prior to import
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
  fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (output_file + "_stats",
                                          c_width, c_height);
}
