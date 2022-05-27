//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/rendering/LineRenderer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include "t_test_utils.hpp"
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <iostream>

vtkm::cont::DataSet MakeTestUniformDataSet(vtkm::Id time)
{
  vtkm::Float64 xmin, xmax, ymin, ymax, zmin, zmax;
  xmin = 0.0;
  ymin = 0.0;
  zmin = 0.0;

  xmax = 10.0;
  ymax = 10.0;
  zmax = 10.0;

  const vtkm::Id3 DIMS(16, 16, 16);

  vtkm::cont::DataSetBuilderUniform dsb;

  vtkm::Float64 xdiff = (xmax - xmin) / (static_cast<vtkm::Float64>(DIMS[0] - 1));
  vtkm::Float64 ydiff = (ymax - ymin) / (static_cast<vtkm::Float64>(DIMS[1] - 1));
  vtkm::Float64 zdiff = (zmax - zmin) / (static_cast<vtkm::Float64>(DIMS[2] - 1));

  vtkm::Vec<vtkm::Float64, 3> ORIGIN(0, 0, 0);
  vtkm::Vec<vtkm::Float64, 3> SPACING(xdiff, ydiff, zdiff);

  vtkm::cont::DataSet dataset = dsb.Create(DIMS, ORIGIN, SPACING);

  vtkm::Id numPoints = DIMS[0] * DIMS[1] * DIMS[2];

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>> velocityField;
  velocityField.Allocate(numPoints);

  vtkm::Id count = 0;
  for (vtkm::Id i = 0; i < DIMS[0]; i++)
  {
    for (vtkm::Id j = 0; j < DIMS[1]; j++)
    {
      for (vtkm::Id k = 0; k < DIMS[2]; k++)
      {
        velocityField.WritePortal().Set(count, vtkm::Vec<vtkm::Float64, 3>(0.01*time, 0.01*time, 0.01*time));
        count++;
      }
    }
  }
  dataset.AddPointField("velocity", velocityField);
  return dataset;
}

void render_output(vtkh::DataSet *data, std::string file_name)
{
  data->AddConstantPointField(1.f,"lines");

  vtkm::Bounds bounds = data->GetGlobalBounds();

  vtkm::rendering::Camera camera;
  camera.ResetToBounds(bounds);
  float bg_color[4] = { 0.f, 0.f, 0.f, 1.f};
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *data,
                                         file_name,
                                         bg_color);

  vtkh::Scene scene;
  scene.AddRender(render);

  vtkh::LineRenderer tracer;
  tracer.SetRadius(.1f);
  tracer.SetInput(data);
  tracer.SetField("lines");

  scene.AddRenderer(&tracer);
  scene.Render();
}

//----------------------------------------------------------------------------
TEST(vtkh_lagrangian, vtkh_serial_lagrangian)
{
  vtkh::Lagrangian lagrangian;
  lagrangian.SetField("velocity");
  lagrangian.SetStepSize(0.1);
  lagrangian.SetWriteFrequency(5);
  lagrangian.SetCustomSeedResolution(1);
  lagrangian.SetSeedResolutionInX(1);
  lagrangian.SetSeedResolutionInY(1);
  lagrangian.SetSeedResolutionInZ(1);

   std::cout << "Running Lagrangian filter test - vtkh" << std::endl;

  for(vtkm::Id time = 1; time <= 10; ++time)
  {
    vtkh::DataSet data_set;
    data_set.AddDomain(MakeTestUniformDataSet(time),0);
    lagrangian.SetInput(&data_set);
    lagrangian.Update();
    vtkh::DataSet *extracted_basis = lagrangian.GetOutput();
    extracted_basis->PrintSummary(std::cout);
    if(time == 10) render_output(extracted_basis, "basis");
  }


}
