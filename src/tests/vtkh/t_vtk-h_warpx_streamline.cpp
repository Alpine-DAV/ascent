//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_warpx_steamlines_par.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Streamline.hpp>
#include <vtkh/filters/WarpXStreamline.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <viskores/io/VTKDataSetWriter.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/CellSetSingleType.h>
#include "t_viskores_test_utils.hpp"
#include <iostream>


void checkValidity(vtkh::DataSet *data, const int maxSteps, bool isSL)
{
  int numDomains = data->GetNumberOfDomains();

  //Check all domains
  for(int i = 0; i < numDomains; i++)
  {
    auto currentDomain = data->GetDomain(i);
    auto cs = currentDomain.GetCellSet();
    if (isSL)
    {
      //Ensure that streamlines took <= to the max number of steps
      for(int j = 0; j < cs.GetNumberOfCells(); j++)
      {
        EXPECT_LE(cs.GetNumberOfPointsInCell(j), maxSteps);
      }
    }
    else
    {
      if (!cs.IsType<viskores::cont::CellSetSingleType<>>())
        EXPECT_TRUE(false);
    }
  }
}

void writeDataSet(vtkh::DataSet *data, std::string fName)
{
  int numDomains = data->GetNumberOfDomains();
  std::cerr << "num domains " << numDomains << std::endl;
  for(int i = 0; i < numDomains; i++)
  {
    char fileNm[1024];
    snprintf(fileNm, 1024,"%s.domain%d.vtk", fName.c_str(), i);
    viskores::io::VTKDataSetWriter write(fileNm);
    write.WriteDataSet(data->GetDomain(i));
  }
}

static inline viskores::FloatDefault
rand01()
{
  return (viskores::FloatDefault)rand() / (RAND_MAX+1.0f);
}

static inline viskores::FloatDefault
randRange(const viskores::FloatDefault &a, const viskores::FloatDefault &b)
{
    return a + (b-a)*rand01();
}


template <typename FilterType>
vtkh::DataSet *
RunFilter(vtkh::DataSet& input,
          const std::string& fieldName,
          const std::vector<viskores::Particle>& seeds,
          int maxAdvSteps,
          double stepSize)
{
  FilterType filter;

  filter.SetInput(&input);
  filter.SetField(fieldName);
  filter.SetNumberOfSteps(maxAdvSteps);
  filter.SetSeeds(seeds);
  filter.SetStepSize(stepSize);
  filter.Update();

  return filter.GetOutput();
}

template <typename FilterType>
vtkh::DataSet *
RunWFilter(vtkh::DataSet& input,
          int maxAdvSteps,
	  std::string output_field,
          double stepSize)
{
  FilterType filter;

  filter.SetInput(&input);
  filter.SetNumberOfSteps(maxAdvSteps);
  filter.SetStepSize(stepSize);
  //warpxstreamline will make its own seeds
  filter.SetTubeSize(0.00000007);
  filter.SetTubeCapping(true);
  filter.SetTubeValue(1.0);
  filter.SetTubeSides(3);
  filter.SetOutputField(output_field);
  filter.Update();

  return filter.GetOutput();
}
//----------------------------------------------------------------------------
TEST(vtkh_serial_warpx_streamlines, vtkh_serial_warpx_streamlines)
{
#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  const int maxAdvSteps = 1000;

  std::cout << "Running serial WarpX Charged Particle Advection" << std::endl;

  vtkh::DataSet warpx_data_set;
  
  std::string warpxParticlesFile = test_data_file("warpXparticles.vtk");
  std::string warpxFieldsFile = test_data_file("warpXfields.vtk");

  viskores::io::VTKDataSetReader seedsReader(warpxParticlesFile);
  viskores::cont::DataSet seedsData = seedsReader.ReadDataSet();
  viskores::io::VTKDataSetReader fieldsReader(warpxFieldsFile);
  viskores::cont::DataSet fieldsData = fieldsReader.ReadDataSet();
  warpx_data_set.AddDomain(seedsData,0);
  warpx_data_set.AddDomain(fieldsData,1);
  viskores::cont::UnknownCellSet cells = fieldsData.GetCellSet();
  viskores::cont::CoordinateSystem coords = fieldsData.GetCoordinateSystem();

  auto w_bounds = coords.GetBounds();
  using Structured3DType = viskores::cont::CellSetStructured<3>;
  Structured3DType castedCells;
  cells.AsCellSet(castedCells);
  auto dims = castedCells.GetSchedulingRange(viskores::TopologyElementTagPoint());
  viskores::Vec3f spacing = { static_cast<viskores::FloatDefault>(w_bounds.X.Length()) / (dims[0] - 1),
                          static_cast<viskores::FloatDefault>(w_bounds.Y.Length()) / (dims[1] - 1),
                          static_cast<viskores::FloatDefault>(w_bounds.Z.Length()) / (dims[2] - 1) };
  constexpr static viskores::FloatDefault SPEED_OF_LIGHT =
    static_cast<viskores::FloatDefault>(2.99792458e8);
  spacing = spacing * spacing;

  viskores::FloatDefault length = static_cast<viskores::FloatDefault>(
    1.0 / (SPEED_OF_LIGHT * viskores::Sqrt(1. / spacing[0] + 1. / spacing[1] + 1. / spacing[2])));
  std::cout << "CFL length : " << length << std::endl;

  vtkh::DataSet *outWSL=NULL;
  
  warpx_data_set.PrintSummary(std::cerr);
  vtkh::WarpXStreamline  warpx;
  warpx.SetNumberOfSteps(maxAdvSteps);
  warpx.SetStepSize(length);
  warpx.SetTubes(true);
  warpx.SetInput(&warpx_data_set);
  warpx.SetOutputField("streamlines");
  warpx.Update();
  outWSL = warpx.GetOutput();

  outWSL->PrintSummary(std::cerr);

  checkValidity(outWSL, maxAdvSteps+1, true);
  writeDataSet(outWSL, "warpx_streamline");
  viskores::Bounds tBounds = outWSL->GetGlobalBounds();

  viskores::rendering::Camera camera;
  camera.SetPosition(viskores::Vec<viskores::Float64,3>(-16, -16, -16));
  camera.ResetToBounds(tBounds);
  vtkh::Render render = vtkh::MakeRender(512,
                                         512,
                                         camera,
                                         *outWSL,
                                         "tout_warpx_render");

  vtkh::RayTracer tracer;
  tracer.SetInput(outWSL);
  tracer.SetField("streamlines");
  std::string fieldName = "streamlines";

  vtkh::Scene scene;
  scene.AddRender(render);
  scene.AddRenderer(&tracer);
  scene.Render();

}
