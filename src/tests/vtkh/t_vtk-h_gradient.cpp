//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_gradient.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_vtkm_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_gradient, vtkh_gradient)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::Gradient grad;
  grad.SetInput(&data_set);
  //grad.SetField("vector_data");
  grad.SetField("point_data_Float64");

  vtkh::GradientParameters params;
  params.output_name = "grad";
  grad.SetParameters(params);

  grad.Update();

  vtkh::DataSet *grad_output = grad.GetOutput();
  grad_output->PrintSummary(std::cout);

  vtkh::VectorMagnitude mag;
  mag.SetInput(grad_output);
  mag.SetField("grad");

  mag.SetResultName("mag");
  mag.Update();

  vtkh::DataSet *mag_output = mag.GetOutput();

  vtkm::Bounds bounds = mag_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.Azimuth(70.f);
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *mag_output,
                                         "gradient_magnitude",
                                          bg_color);

  vtkh::RayTracer tracer;
  tracer.SetInput(mag_output);
  tracer.SetField("mag");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete grad_output;
  delete mag_output;
}

//----------------------------------------------------------------------------
TEST(vtkh_gradient, vtkh_qcriterion)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::SelectKokkosDevice(1);
#endif
  vtkh::DataSet data_set;

  const int base_size = 32;
  const int num_blocks = 2;

  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::Gradient grad;
  grad.SetInput(&data_set);
  grad.SetField("vector_data_Float64");

  vtkh::GradientParameters params;
  params.compute_qcriterion = true;
  params.qcriterion_name = "qcriterion";
  grad.SetParameters(params);

  grad.Update();

  vtkh::DataSet *grad_output = grad.GetOutput();
  grad_output->PrintSummary(std::cout);

  vtkm::Bounds bounds = grad_output->GetGlobalBounds();
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkm::rendering::Camera camera;
  camera.Azimuth(70.f);
  camera.ResetToBounds(bounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *grad_output,
                                         "qcriterion",
                                          bg_color);

  vtkh::RayTracer tracer;
  tracer.SetInput(grad_output);
  tracer.SetField("qcriterion");

  vtkh::Scene scene;
  scene.AddRenderer(&tracer);
  scene.AddRender(render);
  scene.Render();

  delete grad_output;
}
