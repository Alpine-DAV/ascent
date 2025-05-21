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
#include <vtkm/io/VTKDataSetWriter.h>
#include <vtkm/io/VTKDataSetReader.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/CellSetSingleType.h>
#include "t_vtkm_test_utils.hpp"
#include <iostream>
#include <mpi.h>

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
      if (!cs.IsType<vtkm::cont::CellSetSingleType<>>())
        EXPECT_TRUE(false);
    }
  }
}

void writeDataSet(vtkh::DataSet *data, std::string fName, int rank)
{
  int numDomains = data->GetNumberOfDomains();
  std::cerr << "num domains " << numDomains << std::endl;
  for(int i = 0; i < numDomains; i++)
  {
    char fileNm[1024];
    snprintf(fileNm, 1024, "%s.rank%d.domain%d.vtk", fName.c_str(), rank, i);
    vtkm::io::VTKDataSetWriter write(fileNm);
    write.WriteDataSet(data->GetDomain(i));
  }
}

static inline vtkm::FloatDefault
rand01()
{
  return (vtkm::FloatDefault)rand() / (RAND_MAX+1.0f);
}

static inline vtkm::FloatDefault
randRange(const vtkm::FloatDefault &a, const vtkm::FloatDefault &b)
{
    return a + (b-a)*rand01();
}


template <typename FilterType>
vtkh::DataSet *
RunFilter(vtkh::DataSet& input,
          const std::string& fieldName,
          const std::vector<vtkm::Particle>& seeds,
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
  filter.SetTubeSize(0.1);
  filter.SetTubeCapping(true);
  filter.SetTubeValue(1.0);
  filter.SetTubeSides(2);
  filter.SetOutputField(output_field);
  filter.Update();

  return filter.GetOutput();
}
//----------------------------------------------------------------------------
TEST(vtkh_warpx_streamlines_par, vtkh_warpx_streamlines_par)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  const int maxAdvSteps = 1000;

  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  std::cout << "Running parallel WarpX Charged Particle Advection, vtkh - with " << comm_size << " ranks" << std::endl;

  vtkh::DataSet warpx_data_set;
  
  std::string warpxParticlesFile = test_data_file("warpXparticles.vtk");
  std::string warpxFieldsFile = test_data_file("warpXfields.vtk");

  vtkm::io::VTKDataSetReader seedsReader(warpxParticlesFile);
  vtkm::cont::DataSet seedsData = seedsReader.ReadDataSet();
  vtkm::io::VTKDataSetReader fieldsReader(warpxFieldsFile);
  vtkm::cont::DataSet fieldsData = fieldsReader.ReadDataSet();
  warpx_data_set.AddDomain(seedsData,0);
  warpx_data_set.AddDomain(fieldsData,1);
  vtkm::cont::UnknownCellSet cells = fieldsData.GetCellSet();
  vtkm::cont::CoordinateSystem coords = fieldsData.GetCoordinateSystem();

  auto w_bounds = coords.GetBounds();
  using Structured3DType = vtkm::cont::CellSetStructured<3>;
  Structured3DType castedCells;
  cells.AsCellSet(castedCells);
  auto dims = castedCells.GetSchedulingRange(vtkm::TopologyElementTagPoint());
  vtkm::Vec3f spacing = { static_cast<vtkm::FloatDefault>(w_bounds.X.Length()) / (dims[0] - 1),
                          static_cast<vtkm::FloatDefault>(w_bounds.Y.Length()) / (dims[1] - 1),
                          static_cast<vtkm::FloatDefault>(w_bounds.Z.Length()) / (dims[2] - 1) };
  constexpr static vtkm::FloatDefault SPEED_OF_LIGHT =
    static_cast<vtkm::FloatDefault>(2.99792458e8);
  spacing = spacing * spacing;

  vtkm::FloatDefault length = static_cast<vtkm::FloatDefault>(
    1.0 / (SPEED_OF_LIGHT * vtkm::Sqrt(1. / spacing[0] + 1. / spacing[1] + 1. / spacing[2])));
  std::cout << "CFL length : " << length << std::endl;

  vtkh::DataSet *outWSL=NULL;
  
  //warpx_data_set.PrintSummary(std::cerr);

  outWSL = RunWFilter<vtkh::WarpXStreamline>(warpx_data_set, maxAdvSteps, "streamlines", length);
  outWSL->PrintSummary(std::cerr);

  checkValidity(outWSL, maxAdvSteps+1, true);
  writeDataSet(outWSL, "warpx_streamline", rank);
//  vtkm::Bounds tBounds = outWSL->GetGlobalBounds();
//
//  vtkm::rendering::Camera camera;
//  camera.SetPosition(vtkm::Vec<vtkm::Float64,3>(-16, -16, -16));
//  camera.ResetToBounds(tBounds);
//  vtkh::Render render = vtkh::MakeRender(512,
//                                         512,
//                                         camera,
//                                         *outWSL,
//                                         "tout_warpx_streamline_render");
//
//  vtkh::RayTracer tracer;
//  tracer.SetInput(outWSL);
//  tracer.SetField("streamlines");
//
//  vtkh::Scene scene;
//  scene.AddRender(render);
//  scene.AddRenderer(&tracer);
//  scene.Render();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
