// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"


#include <dray/filters/to_bernstein.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/data_model/unstructured_field.hpp>

#include <dray/rendering/camera.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/synthetic/spiral_sample.hpp>
#include <dray/synthetic/tet_sphere_sample.hpp>


/*
TEST (dray_to_bernstein_filter, dray_to_bernstein_filter_hex)
{
  dray::Collection collxn_raw = dray::SynthesizeSpiralSample(1, 0.9, 2, 10).synthesize();
  std::cout << "Synthesized.\n";

  dray::Collection collxn = dray::ToBernstein().execute(collxn_raw);
  std::cout << "Finished converting.\n";

  /// dray::Collection collxn = collxn_raw;
  /// std::cout << "Skipping conversion, using raw.\n";

  using DummyFieldHex = dray::Field<dray::Element<3, 1, dray::Tensor, -1>>;
  for (dray::DataSet &ds : collxn.domains())
    ds.add_field(std::make_shared<DummyFieldHex>( DummyFieldHex::uniform_field(
          ds.mesh()->cells(), dray::Vec<float,1>{{0}}, "uniform")));

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(collxn);

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "diy_spiral");
  remove_test_image (output_file);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.elevate(-30);
  camera.azimuth(0);

  /// camera.reset_to_bounds (collxn_raw.bounds());
  camera.reset_to_bounds (collxn.bounds());

  dray::ColorTable color_table ("ColdAndHot");

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("uniform");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.05);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
}
*/


TEST (dray_to_bernstein_filter, dray_to_bernstein_filter_tri)
{
  dray::Collection collxn_raw = dray::SynthesizeTetSphereSample(1, 10).synthesize();
  std::cout << "Synthesized.\n";

  dray::Collection collxn = dray::ToBernstein().execute(collxn_raw);
  std::cout << "Finished converting.\n";

  /// dray::Collection collxn = collxn_raw;
  /// std::cout << "Skipping conversion, using raw.\n";

  using DummyFieldTri = dray::UnstructuredField<dray::Element<2, 1, dray::Simplex, -1>>;
  for (dray::DataSet &ds : collxn.domains())
    ds.add_field(std::make_shared<DummyFieldTri>( DummyFieldTri::uniform_field(
          ds.mesh()->cells(), dray::Vec<dray::Float,1>{{0}}, "uniform")));

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "tet_sphere");
  remove_test_image (output_file);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.elevate(35);
  camera.azimuth(-35);

  /// camera.reset_to_bounds (collxn_raw.bounds());
  camera.reset_to_bounds (collxn.bounds());

  dray::ColorTable color_table ("ColdAndHot");

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(collxn);
  surface->field("uniform");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.05);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
}
