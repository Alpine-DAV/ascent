// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/subset.hpp>
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


TEST (dray_subset, dray_subset_basic)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  // between tests, we should always start with 0 arrays, 0 usage
  EXPECT_EQ(dray::ArrayRegistry::number_of_arrays(),0);
  EXPECT_EQ(dray::ArrayRegistry::host_usage(),0);
  EXPECT_EQ(dray::ArrayRegistry::device_usage(),0);
  dray::ArrayRegistry::summary();
  dray::stats::StatStore::clear();
  if(dray::stats::StatStore::stats_supported())
  {
    dray::stats::StatStore::enable_stats();
  }

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green_2d.cycle_000050.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file = conduit::utils::join_file_path (output_path, "subset");
  remove_test_image (output_file);

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::DataSet dataset = collection.domain(0);
  int32 elems = dataset.mesh()->cells();
  std::cout<<"elements "<<elems<<"\n";
  dray::Array<dray::int32> flags;
  flags.resize(elems);
  dray::int32 *flags_ptr = flags.get_host_ptr();
  for(int i = 0; i < elems; ++i)
  {
    if(i % 2 == 0)
    {
      flags_ptr[i] = 0;
    }
    else
    {
      flags_ptr[i] = 1;
    }
  }

  dray::Subset subset;
  dray::DataSet subsetted = subset.execute(dataset, flags);
  dray::Collection subset_col;
  subset_col.add_domain(subsetted);

  dray::AABB<3> bounds;
  bounds.include(subset_col.bounds());

  dray::ColorTable color_table ("Spectral");

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  std::string field_name = "density";

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (bounds);

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(subset_col);
  surface->field(field_name);
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
