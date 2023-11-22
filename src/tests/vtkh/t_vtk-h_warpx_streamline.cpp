//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_particle_advection_par.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Streamline.hpp>
#include <vtkh/filters/WarpXStreamline.hpp>
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
      auto cellSet = cs.AsCellSet<vtkm::cont::CellSetExplicit<>>();
      //Ensure that streamlines took <= to the max number of steps
      for(int j = 0; j < cellSet.GetNumberOfCells(); j++)
      {
        EXPECT_LE(cellSet.GetNumberOfPointsInCell(j), maxSteps);
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
    char fileNm[128];
    sprintf(fileNm, "%s.rank%d.domain%d.vtk", fName.c_str(), rank, i);
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
          double stepSize)
{
  FilterType filter;

  filter.SetInput(&input);
  filter.SetNumberOfSteps(maxAdvSteps);
  //warpxstreamline will make its own seeds
  //if(!std::is_same<FilterType,vtkh::WarpXStreamline>::value)
  //  filter.SetSeeds(seeds);
  filter.SetStepSize(stepSize);
  filter.Update();

  return filter.GetOutput();
}
//----------------------------------------------------------------------------
TEST(vtkh_particle_advection, vtkh_serial_particle_advection)
{
#ifdef VTKM_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  std::cout << "Running parallel Particle Advection, vtkh - with " << comm_size << " ranks" << std::endl;

  vtkh::DataSet data_set, warpx_data_set;
  const int base_size = 32;
  const int blocks_per_rank = 1;
  const int maxAdvSteps = 1000;
  const int num_blocks = comm_size * blocks_per_rank;

  for(int i = 0; i < blocks_per_rank; ++i)
  {
    int domain_id = rank * blocks_per_rank + i;
    data_set.AddDomain(CreateTestDataRectilinear(domain_id, num_blocks, base_size), domain_id);
  }

  std::vector<vtkm::Particle> seeds;

  vtkm::Bounds bounds = data_set.GetGlobalBounds();
  std::cout<<"Bounds= "<<bounds<<std::endl;

  for (int i = 0; i < 100; i++)
  {
    vtkm::Particle p;
    p.SetPosition(vtkm::Vec3f(randRange(bounds.X.Min, bounds.X.Max),
                              randRange(bounds.Y.Min, bounds.Y.Max),
      	                      randRange(bounds.Z.Min, bounds.Z.Max)));
    p.SetID(static_cast<vtkm::Id>(i));

    seeds.push_back(p);
  }
  
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
  std::cout << "Bounds : " << w_bounds << std::endl;
  using Structured3DType = vtkm::cont::CellSetStructured<3>;
  std::cerr << "HERE 1" << std::endl;
  Structured3DType castedCells;
  std::cerr << "HERE 2" << std::endl;
  cells.AsCellSet(castedCells);
  std::cerr << "HERE 3" << std::endl;
  auto dims = castedCells.GetSchedulingRange(vtkm::TopologyElementTagPoint());
  std::cerr << "HERE 4" << std::endl;
  vtkm::Vec3f spacing = { static_cast<vtkm::FloatDefault>(w_bounds.X.Length()) / (dims[0] - 1),
                          static_cast<vtkm::FloatDefault>(w_bounds.Y.Length()) / (dims[1] - 1),
                          static_cast<vtkm::FloatDefault>(w_bounds.Z.Length()) / (dims[2] - 1) };
  std::cerr << "HERE 5" << std::endl;
  constexpr static vtkm::FloatDefault SPEED_OF_LIGHT =
    static_cast<vtkm::FloatDefault>(2.99792458e8);
  spacing = spacing * spacing;
  std::cerr << "HERE 6" << std::endl;

  vtkm::FloatDefault length = static_cast<vtkm::FloatDefault>(
    1.0 / (SPEED_OF_LIGHT * vtkm::Sqrt(1. / spacing[0] + 1. / spacing[1] + 1. / spacing[2])));
  std::cout << "CFL length : " << length << std::endl;


  vtkh::DataSet *outWSL=NULL;
  
  warpx_data_set.PrintSummary(std::cerr);
  outWSL = RunWFilter<vtkh::WarpXStreamline>(warpx_data_set, maxAdvSteps, length);
  outWSL->PrintSummary(std::cout);
  checkValidity(outWSL, maxAdvSteps+1, true);
  writeDataSet(outWSL, "warpx_streamline", rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
