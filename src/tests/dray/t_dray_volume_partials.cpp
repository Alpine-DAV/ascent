// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/math.hpp>
#include <dray/array_registry.hpp>

#include <dray/utils/appstats.hpp>

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
TEST (dray_volume_partials, dray_volume_partials)
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

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "impeller_vr_partial");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.3f, 0.19f);
  color_table.add_alpha (0.4f, 0.21f);
  color_table.add_alpha (1.0f, 0.9f);

  // Camera
  const int c_width  = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::PointLight light;
  light.m_pos= { 0.5f, 0.5f, 0.5f };
  light.m_amb = { 0.5f, 0.5f, 0.5f };
  light.m_diff = { 0.70f, 0.70f, 0.70f };
  light.m_spec = { 0.9f, 0.9f, 0.9f };
  light.m_spec_pow = 90.0;

  dray::Array<dray::PointLight> lights;
  lights.resize(1);
  dray::PointLight *l_ptr = lights.get_host_ptr();
  l_ptr[0] = light;

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(dataset);
  volume->field("diffusion");
  volume->samples(100);
  volume->color_map().color_table(color_table);
  dray::Array<dray::VolumePartial> partials = volume->integrate(rays, lights);

  dray::stats::StatStore::write_ray_stats (output_file + "_stats",
                                           camera.get_width (),
                                           camera.get_height ());

  volume->save(output_file, partials, c_width, c_height);
  // note: dray diff tolerance was 0.2f prior to import
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
}

//---------------------------------------------------------------------------//
TEST (dray_volume_partials, dray_empty_check)
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

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "partials_empty_check");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.3f, 0.19f);
  color_table.add_alpha (0.4f, 0.21f);
  color_table.add_alpha (1.0f, 0.9f);

  // Camera
  const int c_width  = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  // Look at nothing and make sure things don't break
  dray::Vec<float,3> la({-1000.f, 0.f, 0.f});
  camera.set_look_at(la);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::PointLight light;
  light.m_pos= { 0.5f, 0.5f, 0.5f };
  light.m_amb = { 0.5f, 0.5f, 0.5f };
  light.m_diff = { 0.70f, 0.70f, 0.70f };
  light.m_spec = { 0.9f, 0.9f, 0.9f };
  light.m_spec_pow = 90.0;

  dray::Array<dray::PointLight> lights;
  lights.resize(1);
  dray::PointLight *l_ptr = lights.get_host_ptr();
  l_ptr[0] = light;

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(dataset);
  volume->field("diffusion");
  volume->samples(100);
  volume->color_map().color_table(color_table);
  dray::Array<dray::VolumePartial> partials = volume->integrate(rays, lights);

  dray::stats::StatStore::write_ray_stats (output_file + "_stats",
                                           camera.get_width (),
                                           camera.get_height ());

  volume->save(output_file, partials, c_width, c_height);
  // note: dray diff tolerance was 0.2f prior to import
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
}
