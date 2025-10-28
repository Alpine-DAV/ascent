//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include "t_viskores_test_utils.hpp"

#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/PartitionedDataSet.h>
#include <viskores/cont/EnvironmentTracker.h>

#include <viskores/filter/scalar_topology/worklet/contourtree_augmented/Types.h>
#include <viskores/filter/MapFieldPermutation.h>

#include <iostream>

#ifdef VISKORES_ENABLE_MPI
#include <mpi.h>

// This is from Viskores diy mpi_cast.hpp. Need the make_DIY_MPI_Comm
namespace viskoresdiy
{
namespace mpi
{

#define DEFINE_MPI_CAST(mpitype)                                                                              \
inline mpitype& mpi_cast(DIY_##mpitype& obj) { return *reinterpret_cast<mpitype*>(&obj); }                    \
inline const mpitype& mpi_cast(const DIY_##mpitype& obj) { return *reinterpret_cast<const mpitype*>(&obj); }  \
inline DIY_##mpitype make_DIY_##mpitype(const mpitype& obj) { DIY_##mpitype ret; mpi_cast(ret) = obj; return ret; }

DEFINE_MPI_CAST(MPI_Comm)
#undef DEFINE_MPI_CAST

}
} // diy::mpi

#endif

using ValueType = viskores::Float64;

inline viskores::IdComponent FindSplitAxis(viskores::Id3 globalSize)
{
  viskores::IdComponent splitAxis = 0;
  for (viskores::IdComponent d = 1; d < 3; ++d)
  {
    if (globalSize[d] > globalSize[splitAxis])
    {
      splitAxis = d;
    }
  }
  return splitAxis;
}

inline viskores::Id3 ComputeNumberOfBlocksPerAxis(viskores::Id3 globalSize, viskores::Id numberOfBlocks)
{
  // Split numberOfBlocks into a power of two and a remainder
  viskores::Id powerOfTwoPortion = 1;
  while (numberOfBlocks % 2 == 0)
  {
    powerOfTwoPortion *= 2;
    numberOfBlocks /= 2;
  }

  viskores::Id3 blocksPerAxis{ 1, 1, 1 };
  if (numberOfBlocks > 1)
  {
    // Split the longest axis according to remainder
    viskores::IdComponent splitAxis = FindSplitAxis(globalSize);
    blocksPerAxis[splitAxis] = numberOfBlocks;
    globalSize[splitAxis] /= numberOfBlocks;
  }

  // Now perform splits for the power of two remainder of numberOfBlocks
  while (powerOfTwoPortion > 1)
  {
    viskores::IdComponent splitAxis = FindSplitAxis(globalSize);
    VISKORES_ASSERT(globalSize[splitAxis] > 1);
    blocksPerAxis[splitAxis] *= 2;
    globalSize[splitAxis] /= 2;
    powerOfTwoPortion /= 2;
  }

  return blocksPerAxis;
}

inline std::tuple<viskores::Id3, viskores::Id3, viskores::Id3> ComputeBlockExtents(viskores::Id3 globalSize,
                                                                       viskores::Id3 blocksPerAxis,
                                                                       viskores::Id blockNo)
{
  // DEBUG: std::cout << "ComputeBlockExtents("<<globalSize <<", " << blocksPerAxis << ", " << blockNo << ")" << std::endl;
  // DEBUG: std::cout << "Block " << blockNo;

  viskores::Id3 blockIndex, blockOrigin, blockSize;
  for (viskores::IdComponent d = 0; d < 3; ++d)
  {
    blockIndex[d] = blockNo % blocksPerAxis[d];
    blockNo /= blocksPerAxis[d];

    float dx = float(globalSize[d] - 1) / float(blocksPerAxis[d]);
    blockOrigin[d] = viskores::Id(blockIndex[d] * dx);
    viskores::Id maxIdx =
      blockIndex[d] < blocksPerAxis[d] - 1 ? viskores::Id((blockIndex[d] + 1) * dx) : globalSize[d] - 1;
    blockSize[d] = maxIdx - blockOrigin[d] + 1;
    // DEBUG: std::cout << " " << blockIndex[d] <<  dx << " " << blockOrigin[d] << " " << maxIdx << " " << blockSize[d] << "; ";
  }
  // DEBUG: std::cout << " -> " << blockIndex << " "  << blockOrigin << " " << blockSize << std::endl;
  return std::make_tuple(blockIndex, blockOrigin, blockSize);
}

inline viskores::cont::DataSet CreateSubDataSet(const viskores::cont::DataSet& ds,
                                            viskores::Id3 blockOrigin,
                                            viskores::Id3 blockSize,
                                            const std::string& fieldName)
{
  viskores::Id3 globalSize;
  viskores::cont::CastAndCall(
    ds.GetCellSet(), viskores::worklet::contourtree_augmented::GetPointDimensions(), globalSize);
  const viskores::Id nOutValues = blockSize[0] * blockSize[1] * blockSize[2];

  const auto inDataArrayHandle = ds.GetPointField(fieldName).GetData();

  viskores::cont::ArrayHandle<viskores::Id> copyIdsArray;
  copyIdsArray.Allocate(nOutValues);
  auto copyIdsPortal = copyIdsArray.WritePortal();

  viskores::Id3 outArrIdx;
  for (outArrIdx[2] = 0; outArrIdx[2] < blockSize[2]; ++outArrIdx[2])
    for (outArrIdx[1] = 0; outArrIdx[1] < blockSize[1]; ++outArrIdx[1])
      for (outArrIdx[0] = 0; outArrIdx[0] < blockSize[0]; ++outArrIdx[0])
      {
        viskores::Id3 inArrIdx = outArrIdx + blockOrigin;
        viskores::Id inIdx = (inArrIdx[2] * globalSize[1] + inArrIdx[1]) * globalSize[0] + inArrIdx[0];
        viskores::Id outIdx =
          (outArrIdx[2] * blockSize[1] + outArrIdx[1]) * blockSize[0] + outArrIdx[0];
        VISKORES_ASSERT(inIdx >= 0 && inIdx < inDataArrayHandle.GetNumberOfValues());
        VISKORES_ASSERT(outIdx >= 0 && outIdx < nOutValues);
        copyIdsPortal.Set(outIdx, inIdx);
      }
  // DEBUG: std::cout << copyIdsPortal.GetNumberOfValues() << std::endl;

  viskores::cont::Field permutedField;
  bool success =
    viskores::filter::MapFieldPermutation(ds.GetPointField(fieldName), copyIdsArray, permutedField);
  if (!success)
    throw viskores::cont::ErrorBadType("Field copy failed (probably due to invalid type)");

  viskores::cont::DataSetBuilderUniform dsb;
  if (globalSize[2] <= 1) // 2D Data Set
  {
    viskores::Id2 dimensions{ blockSize[0], blockSize[1] };
    viskores::cont::DataSet dataSet = dsb.Create(dimensions);
    viskores::cont::CellSetStructured<2> cellSet;
    cellSet.SetPointDimensions(dimensions);
    cellSet.SetGlobalPointDimensions(viskores::Id2{ globalSize[0], globalSize[1] });
    cellSet.SetGlobalPointIndexStart(viskores::Id2{ blockOrigin[0], blockOrigin[1] });
    dataSet.SetCellSet(cellSet);
    dataSet.AddField(permutedField);
    return dataSet;
  }
  else
  {
    viskores::cont::DataSet dataSet = dsb.Create(blockSize);
    viskores::cont::CellSetStructured<3> cellSet;
    cellSet.SetPointDimensions(blockSize);
    cellSet.SetGlobalPointDimensions(globalSize);
    cellSet.SetGlobalPointIndexStart(blockOrigin);
    dataSet.SetCellSet(cellSet);
    dataSet.AddField(permutedField);
    return dataSet;
  }
}

//
// Viskores data read code from "TestingContourTreeUniformDistributedFilter.h"
// function: RunContourTreeDUniformDistributed
//
void GetPartitionedDataSet( const viskores::cont::DataSet& ds, const std::string &fieldName, const int numberOfBlocks,
                            const int rank, const int numberOfRanks, viskores::cont::PartitionedDataSet &pds )
{
  // Get dimensions of data set
  viskores::Id3 globalSize;
  viskores::cont::CastAndCall(ds.GetCellSet(), viskores::worklet::contourtree_augmented::GetPointDimensions(), globalSize);

  // Determine split
  viskores::Id3 blocksPerAxis = ComputeNumberOfBlocksPerAxis(globalSize, numberOfBlocks);
  viskores::Id blocksPerRank = numberOfBlocks / numberOfRanks;
  viskores::Id numRanksWithExtraBlock = numberOfBlocks % numberOfRanks;
  viskores::Id blocksOnThisRank, startBlockNo;

  if (rank < numRanksWithExtraBlock)
  {
    blocksOnThisRank = blocksPerRank + 1;
    startBlockNo = (blocksPerRank + 1) * rank;
  }
  else
  {
    blocksOnThisRank = blocksPerRank;
    startBlockNo = numRanksWithExtraBlock * (blocksPerRank + 1) + (rank - numRanksWithExtraBlock) * blocksPerRank;
  }

  // Created partitioned (split) data set
  //viskores::cont::PartitionedDataSet pds;
  viskores::cont::ArrayHandle<viskores::Id3> localBlockIndices;

  localBlockIndices.Allocate(blocksOnThisRank);

  auto localBlockIndicesPortal = localBlockIndices.WritePortal();

    for (viskores::Id blockNo = 0; blockNo < blocksOnThisRank; ++blockNo)
  {
    viskores::Id3 blockOrigin, blockSize, blockIndex;
    std::tie(blockIndex, blockOrigin, blockSize) =
      ComputeBlockExtents(globalSize, blocksPerAxis, startBlockNo + blockNo);
    pds.AppendPartition(CreateSubDataSet(ds, blockOrigin, blockSize, fieldName));
    localBlockIndicesPortal.Set(blockNo, blockIndex);
  }
}


#ifdef VISKORES_ENABLE_MPI
  #define VDATASET viskores::cont::PartitionedDataSet
#else
  #define VDATASET viskores::cont::DataSet
#endif

//----------------------------------------------------------------------------
bool ReadTestData(const std::string& filename, VDATASET &inDataSet,
                  const int mpiRank, const int mpiSize)
{
  std::ifstream inFile(filename);
  if(inFile.bad())
  {
    std::cout << "Error reading data file: " << filename << std::endl;
    return( false );
  }

  // Read the dimensions of the mesh, i.e,. number of elementes in x, y, and z
  // y, x, z
  std::vector<std::size_t> dims;
  std::string line;
  getline(inFile, line);
  std::istringstream linestream(line);
  std::size_t dimVertices;
  while (linestream >> dimVertices)
  {
    dims.push_back(dimVertices);
  }

  // swap y to x and x to y.
  std::swap( dims[0], dims[1] );

  // Compute the number of vertices, i.e., xdim * ydim * zdim
  unsigned short nDims = static_cast<unsigned short>(dims.size());
  std::size_t nVertices = static_cast<std::size_t>(
    std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>()));

  // Print the mesh metadata
  if(mpiRank == 0 && 0)
  {
    std::cout << "Number of dimensions: " << nDims << std::endl;
    std::cout << "Number of mesh vertices: " << nVertices << std::endl;
  }

  // Check the the number of dimensiosn is either 2D or 3D
  bool invalidNumDimensions = (nDims < 2 || nDims > 3);
  if(invalidNumDimensions)
  {
    if(mpiRank == 0)
      std::cout << "The input mesh is " << nDims << "D. Input data must be either 2D or 3D."
                << std::endl;
    return( false );
  }

  // Read data
  std::vector<ValueType> values(nVertices);
  for(std::size_t vertex = 0; vertex < nVertices; ++vertex)
  {
    inFile >> values[vertex];
  }

  // Finish reading the data from file.
  inFile.close();

  viskores::cont::DataSetBuilderUniform dsb;

#ifdef VISKORES_ENABLE_MPI
  viskores::cont::DataSet ds;
  viskores::cont::DataSet *pds = &ds;
#else // VISKORES_ENABLE_MPI
  viskores::cont::DataSet *pds = &inDataSet;
#endif // VISKORES_ENABLE_MPI

  {
    // build the input dataset
    // 2D data
    if(nDims == 2)
    {
      viskores::Id2 vdims;
      vdims[0] = static_cast<viskores::Id>(dims[0]);
      vdims[1] = static_cast<viskores::Id>(dims[1]);
      *pds = dsb.Create(vdims);
    }
    // 3D data
    else
    {
      viskores::Id3 vdims;
      vdims[0] = static_cast<viskores::Id>(dims[0]);
      vdims[1] = static_cast<viskores::Id>(dims[1]);
      vdims[2] = static_cast<viskores::Id>(dims[2]);
      *pds = dsb.Create(vdims);
    }
    pds->AddPointField("values", values);
  }

#ifdef VISKORES_ENABLE_MPI
  GetPartitionedDataSet( ds, "values", mpiSize, mpiRank, mpiSize, inDataSet );
#endif // VISKORES_ENABLE_MPI

  return( true );
}

//----------------------------------------------------------------------------
bool GetDataSet( vtkh::DataSet &data_set, const int mpiRank, const int mpiSize )
{
  const std::string filename = test_data_file("fuel.txt");
  VDATASET ds;

  if( ReadTestData(filename, ds, mpiRank, mpiSize) == false )
    return( false );

#ifdef VISKORES_ENABLE_MPI
  for(viskores::Id id = 0; id < ds.GetNumberOfPartitions(); ++id)
  {
    viskores::cont::DataSet dom = ds.GetPartition(id);

    data_set.AddDomain(dom, id);
  }
#else
  data_set.AddDomain(ds, 0);
#endif

  return( true );
}

int StdoutToFile( int rank )
{
  // Redirect stdout to file if we are using MPI with Debugging
  // From https://www.unix.com/302983597-post2.html
  char cstr_filename[32];

  snprintf(cstr_filename, sizeof(cstr_filename), "cout_%d.log", rank);
  int out = open(cstr_filename, O_RDWR | O_CREAT | O_APPEND, 0600);
  if (-1 == out)
  {
    perror("opening cout.log");
    return 255;
  }

  snprintf(cstr_filename, sizeof(cstr_filename), "cerr_%d.log", rank);
  int err = open(cstr_filename, O_RDWR | O_CREAT | O_APPEND, 0600);
  if (-1 == err)
  {
    perror("opening cerr.log");
    return 255;
  }

  int save_out = dup(fileno(stdout));
  int save_err = dup(fileno(stderr));

  if (-1 == dup2(out, fileno(stdout)))
  {
    perror("cannot redirect stdout");
    return 255;
  }
  if (-1 == dup2(err, fileno(stderr)))
  {
    perror("cannot redirect stderr");
    return 255;
  }
  return 0;
}


//----------------------------------------------------------------------------
TEST(vtkh_contour_tree, vtkh_contour_tree)
{
  // Default values if we are serial.
  int mpiSize = 1, mpiRank = 0;

#ifdef VISKORES_ENABLE_KOKKOS
  vtkh::InitializeKokkos();
#endif

#ifdef VISKORES_ENABLE_MPI
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  // Simple way to dump cout and cerr to files for MPI applications.
  //StdoutToFile( mpiRank );

  // Setup MPI comm for VTK-h.
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  // Setup Viskores GlobalCommuncator. 
  // This is need because the GlobalCommuncator does not setup it self up right if you call MPI_Init.
  auto comm = MPI_COMM_WORLD;
  viskores::cont::EnvironmentTracker::SetCommunicator(viskoresdiy::mpi::communicator(viskoresdiy::mpi::make_DIY_MPI_Comm(comm)));

  auto envComm = viskores::cont::EnvironmentTracker::GetCommunicator();
  if( mpiRank != envComm.rank() || mpiSize != envComm.size() )
  {
    // Print message to check for how this was built.
    std::cout << "mpiRank:  " << mpiRank        << " mpiSize:  " << mpiSize        << std::endl;
    std::cout << "Env Rank: " << envComm.rank() << " Env Size: " << envComm.size() << std::endl;
    std::cout << "If the Rank and Size do not match, Viskores needs to be built with Viskores_ENABLE_MPI." << std::endl;
  }
#endif

  vtkh::DataSet data_set;

  if( GetDataSet(data_set, mpiRank, mpiSize) == false )
  {
    std::cout << "Error getting data." << std::endl;
    return;
  }

  vtkh::MarchingCubes marcher;
  const int num_levels = 5;

  marcher.SetInput(&data_set);
  marcher.SetField("values");
  marcher.SetLevels(num_levels);
  marcher.SetUseContourTree(true);
  marcher.AddMapField("values");
  marcher.Update();

  std::vector<double> isoValues = marcher.GetIsoValues();
  std::sort(isoValues.begin(), isoValues.end());

  EXPECT_FLOAT_EQ(isoValues[0], 1e-05);
  EXPECT_FLOAT_EQ(isoValues[1], 82);
  EXPECT_FLOAT_EQ(isoValues[2], 133);
  EXPECT_FLOAT_EQ(isoValues[3], 168);
  EXPECT_FLOAT_EQ(isoValues[4], 177);

  vtkh::DataSet *output = marcher.GetOutput();
  if( output )
    delete output;

#ifdef VISKORES_ENABLE_MPI
  MPI_Finalize();
#endif
}
