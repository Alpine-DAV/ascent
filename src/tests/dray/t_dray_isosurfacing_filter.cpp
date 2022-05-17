// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"


#include <dray/filters/isosurfacing.hpp>
#include <dray/filters/vector_component.hpp>
#include <dray/filters/to_bernstein.hpp>

#include <dray/data_model/unstructured_field.hpp>
#include <dray/rendering/camera.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/synthetic/affine_radial.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/error.hpp>


TEST (dray_isosurface_filter, dray_isosurface_filter_analytic)
{
  using dray::Float;

  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<Float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<Float, 3> radius = {{1.0f, 1.0f, 1.0f}};
  const dray::Vec<Float, 3> range_radius = {{1.0f, 1.0f, -1.0f}};
  const dray::Vec<Float, 3> range_radius_aux = {{1.0f, 1.0f, -1.0f}};

  dray::Collection collxn =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .equip("perfection", range_radius)
      .equip("aux", range_radius_aux)
      .synthesize();

  const std::string iso_field_name = "perfection";
  const Float isoval = 1.1;

  std::shared_ptr<dray::ExtractIsosurface> iso_extractor
    = std::make_shared<dray::ExtractIsosurface>();
  iso_extractor->iso_field(iso_field_name);
  iso_extractor->iso_value(isoval);

  // Extract isosurface. Partly made of tris, partly quads.
  auto isosurf_tri_quad = iso_extractor->execute(collxn);
  dray::Collection isosurf_tris = isosurf_tri_quad.first;
  dray::Collection isosurf_quads = isosurf_tri_quad.second;

  isosurf_quads = dray::ToBernstein().execute(isosurf_quads);
  isosurf_tris = dray::ToBernstein().execute(isosurf_tris);

  size_t count_cells = 0;
  for (dray::DataSet &ds : collxn.domains())
    count_cells += ds.mesh()->cells();
  std::cout << "input collxn contains " << count_cells << " cells.\n";

  count_cells = 0;
  for (dray::DataSet &ds : isosurf_tris.domains())
    count_cells += ds.mesh()->cells();
  std::cout << "isosurf_tris collxn contains " << count_cells << " cells.\n";

  count_cells = 0;
  for (dray::DataSet &ds : isosurf_quads.domains())
    count_cells += ds.mesh()->cells();
  std::cout << "isosurf_quads collxn contains " << count_cells << " cells.\n";

  // Add a field so that it can be rendered.
  using DummyFieldTri = dray::UnstructuredField<dray::Element<2, 1, dray::Simplex, -1>>;
  using DummyFieldQuad = dray::UnstructuredField<dray::Element<2, 1, dray::Tensor, -1>>;
  for (dray::DataSet &ds : isosurf_tris.domains())
    ds.add_field(std::make_shared<DummyFieldTri>( DummyFieldTri::uniform_field(
            ds.mesh()->cells(), dray::Vec<Float,1>{{0}}, "uniform")));
  for (dray::DataSet &ds : isosurf_quads.domains())
    ds.add_field(std::make_shared<DummyFieldQuad>( DummyFieldQuad::uniform_field(
            ds.mesh()->cells(), dray::Vec<Float,1>{{0}}, "uniform")));

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "isosurface_meshed");
  remove_test_image (output_file);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(-40);

  camera.reset_to_bounds (collxn.bounds());

  /// dray::Range aux_range;
  /// aux_range.include(isosurf_tris.range("aux"));
  /// aux_range.include(isosurf_quads.range("aux"));
  dray::Range aux_range = collxn.range("aux");

  dray::ColorTable color_table ("ColdAndHot");

  std::shared_ptr<dray::Surface> surface_tris
    = std::make_shared<dray::Surface>(isosurf_tris);
  std::shared_ptr<dray::Surface> surface_quads
    = std::make_shared<dray::Surface>(isosurf_quads);

  surface_tris->field("aux");
  surface_tris->color_map().color_table(color_table);
  surface_tris->color_map().scalar_range(aux_range);
  surface_tris->draw_mesh (false);
  surface_tris->line_thickness(.1);
  surface_quads->field("aux");
  surface_quads->color_map().color_table(color_table);
  surface_quads->color_map().scalar_range(aux_range);
  surface_quads->draw_mesh (false);
  surface_quads->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface_tris);
  renderer.add(surface_quads);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
}



TEST (dray_isosurface_filter, dray_isosurface_filter_tg_velx_density)
{
  using dray::Float;

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "isosurface_meshed_simple");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_000190.root";

  dray::Collection collxn = dray::BlueprintReader::load (root_file);

  dray::VectorComponent vc;
  vc.field("velocity");
  vc.output_name("velocity_x");
  vc.component(0);
  collxn = vc.execute(collxn);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(-40);

  camera.reset_to_bounds (collxn.bounds());

  dray::ColorTable color_table ("ColdAndHot");

  const Float isoval = 0.09;
  const std::string iso_field_name = "velocity_x";
  const std::string color_field_name = "velocity_x";


  std::shared_ptr<dray::ExtractIsosurface> iso_extractor
    = std::make_shared<dray::ExtractIsosurface>();
  iso_extractor->iso_field(iso_field_name);
  iso_extractor->iso_value(isoval);

  // Extract isosurface. Partly made of tris, partly quads.
  auto isosurf_tri_quad = iso_extractor->execute(collxn);
  dray::Collection isosurf_tris = isosurf_tri_quad.first;
  dray::Collection isosurf_quads = isosurf_tri_quad.second;

  isosurf_quads = dray::ToBernstein().execute(isosurf_quads);
  isosurf_tris = dray::ToBernstein().execute(isosurf_tris);

  size_t count_cells = 0;
  for (dray::DataSet &ds : collxn.domains())
    count_cells += ds.mesh()->cells();
  std::cout << "input collxn contains " << count_cells << " cells.\n";

  count_cells = 0;
  for (dray::DataSet &ds : isosurf_tris.domains())
    count_cells += ds.mesh()->cells();
  std::cout << "isosurf_tris collxn contains " << count_cells << " cells.\n";

  count_cells = 0;
  for (dray::DataSet &ds : isosurf_quads.domains())
    count_cells += ds.mesh()->cells();
  std::cout << "isosurf_quads collxn contains " << count_cells << " cells.\n";


  std::shared_ptr<dray::Surface> surface_tris
    = std::make_shared<dray::Surface>(isosurf_tris);
  std::shared_ptr<dray::Surface> surface_quads
    = std::make_shared<dray::Surface>(isosurf_quads);

  /// dray::Range color_field_range;
  /// color_field_range.include(isosurf_tris.range(color_field_name));
  /// color_field_range.include(isosurf_quads.range(color_field_name));
  dray::Range color_field_range = collxn.range(color_field_name);

  surface_tris->field(color_field_name);
  surface_tris->color_map().color_table(color_table);
  surface_tris->color_map().scalar_range(color_field_range);
  surface_tris->draw_mesh (false);
  surface_tris->line_thickness(.1);
  surface_quads->field(color_field_name);
  surface_quads->color_map().color_table(color_table);
  surface_quads->color_map().scalar_range(color_field_range);
  surface_quads->draw_mesh (false);
  surface_quads->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface_tris);
  renderer.add(surface_quads);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  ///EXPECT_TRUE (check_test_image (output_file));
}
