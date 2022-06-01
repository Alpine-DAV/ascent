//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>

#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_emtpy_data, vtkh_empty_vtkm)
{
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 1;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::IsoVolume iso;

  // create a range that does not exist
  vtkm::Range iso_range;
  iso_range.Min = -100.;
  iso_range.Max = -10.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);
  iso.Update();

  vtkh::DataSet *iso_output = iso.GetOutput();


  const int num_vals = 2;
  double iso_vals [num_vals];
  iso_vals[0] = -1; // ask for something that does not exist
  iso_vals[1] = (float)base_size * (float)num_blocks * 0.5f;

  vtkh::MarchingCubes marcher;
  marcher.SetField("point_data_Float64");
  marcher.SetInput(iso_output);
  marcher.SetIsoValues(iso_vals, num_vals);
  marcher.AddMapField("point_data_Float64");
  marcher.AddMapField("cell_data_Float64");
  marcher.Update();

  vtkh::DataSet *contour_output = marcher.GetOutput();

  vtkh::Clip clipper;

  vtkm::Bounds clip_bounds = contour_output->GetGlobalBounds();
  vtkm::Vec<vtkm::Float64, 3> center = clip_bounds.Center();
  clip_bounds.X.Max = center[0] + .5;
  clip_bounds.Y.Max = center[1] + .5;
  clip_bounds.Z.Max = center[2] + .5;

  clipper.SetBoxClip(clip_bounds);
  clipper.SetInput(contour_output);
  clipper.Update();

  vtkh::DataSet *clip_output = clipper.GetOutput();

  vtkh::Threshold thresher;
  thresher.SetInput(clip_output);
  thresher.SetField("point_data_Float64");

  double upper_bound = 1.;
  double lower_bound = 0.;

  thresher.SetUpperThreshold(upper_bound);
  thresher.SetLowerThreshold(lower_bound);
  thresher.Update();
  vtkh::DataSet *thresh_output = thresher.GetOutput();

  vtkm::Bounds bounds = thresh_output->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(16,-32,-32));
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *thresh_output,
                                         "empty_vtkm",
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::RayTracer tracer;
  tracer.SetInput(iso_output);
  tracer.SetField("point_data_Float64");

  scene.AddRenderer(&tracer);
  scene.Render();

  delete iso_output;
  delete contour_output;
  delete clip_output;
  delete thresh_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_emtpy_data, vtkh_empty_vtkh)
{
  vtkh::DataSet data_set;

  vtkh::IsoVolume iso;

  // create a range that does not exist
  vtkm::Range iso_range;
  iso_range.Min = -100.;
  iso_range.Max = -10.;
  iso.SetRange(iso_range);
  iso.SetField("point_data_Float64");
  iso.SetInput(&data_set);

  bool got_error = false;
  try
  {
    iso.Update();
  }
  catch(vtkh::Error &e)
  {
    got_error = true;
  }
  catch(...)
  {
    got_error = true;
  }

  ASSERT_TRUE(got_error);
}
