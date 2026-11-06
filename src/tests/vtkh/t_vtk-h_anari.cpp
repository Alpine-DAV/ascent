//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_anari.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/ANARIRenderer.hpp>
#include "t_viskores_test_utils.hpp"

namespace
{

bool HasNonBackgroundPixels(vtkh::Render& render)
{
  auto colors = render.GetCanvas().GetColorBuffer().ReadPortal();
  for (viskores::Id pixel = 0; pixel < colors.GetNumberOfValues(); ++pixel)
  {
    const auto color = colors.Get(pixel);
    if (color[0] > 0.f || color[1] > 0.f || color[2] > 0.f)
    {
      return true;
    }
  }
  return false;
}

} // anonymous namespace

//----------------------------------------------------------------------------
TEST(vtkh_anari, unsupported_unstructured_volume)
{
#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif

  vtkh::DataSet data_set;

  data_set.AddDomain(CreateTestDataPoints(8), 0);

  viskores::Bounds bounds = data_set.GetGlobalBounds();

  viskores::rendering::Camera camera;
  viskores::Vec<viskores::Float32,3> pos = camera.GetPosition();
  pos[0]+=.1;
  pos[1]+=.1;
  camera.SetPosition(pos);
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(256,
                                         256,
                                         camera,
                                         data_set,
                                         "volume_unstructured");
  render.DoRenderAnnotations(false);

  viskores::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, 0.01);
  color_map.AddPointAlpha(1.0, 0.6);

  vtkh::ANARIVolumeRenderer tracer;
  tracer.SetColorTable(color_map);
  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  EXPECT_THROW(scene.Render(), vtkh::Error);
}

TEST(vtkh_anari, structured_volume)
{
#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif

  vtkh::DataSet data_set;

  const int base_size = 16;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  viskores::Bounds bounds = data_set.GetGlobalBounds();

  viskores::rendering::Camera camera;
  viskores::Vec<viskores::Float32,3> pos = camera.GetPosition();
  pos[0]+=.1;
  pos[1]+=.1;
  camera.SetPosition(pos);
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(256,
                                         256,
                                         camera,
                                         data_set,
                                         "volume");
  render.DoRenderAnnotations(false);

  viskores::cont::ColorTable color_map("Cool to Warm");
  color_map.AddPointAlpha(0.0, 0.01);
  color_map.AddPointAlpha(1.0, 0.6);
  vtkh::ANARIVolumeRenderer tracer;
  tracer.SetColorTable(color_map);
  tracer.SetInput(&data_set);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();
  EXPECT_TRUE(HasNonBackgroundPixels(render));
}

TEST(vtkh_anari, triangle_geometry)
{
  vtkh::DataSet data_set;

  const int base_size = 16;
  data_set.AddDomain(CreateTestData(0, 1, base_size), 0);

  vtkh::MarchingCubes marcher;
  marcher.SetInput(&data_set);
  marcher.SetField("point_data_Float64");
  marcher.SetIsoValue(base_size * 0.5);
  marcher.AddMapField("point_data_Float64");
  marcher.Update();

  vtkh::DataSet* iso_output = marcher.GetOutput();
  viskores::Bounds bounds = iso_output->GetGlobalBounds();

  viskores::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(256,
                                         256,
                                         camera,
                                         *iso_output,
                                         "anari_triangle");
  render.DoRenderAnnotations(false);

  vtkh::ANARITriangleRenderer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();

  EXPECT_TRUE(HasNonBackgroundPixels(render));
  delete iso_output;
}
