#include <vtkh/filters/ContourTree.hpp>

#include <vtkh/filters/Recenter.hpp>

// vtkm includes
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/Storage.h>
#include <vtkm/internal/Configure.h>
#include <vtkm/filter/ContourTreeUniformAugmented.h>
#include <vtkm/worklet/contourtree_augmented/PrintVectors.h>
#include <vtkm/worklet/contourtree_augmented/ProcessContourTree.h>
#include <vtkm/worklet/contourtree_augmented/processcontourtree/Branch.h>
#include <vtkm/worklet/contourtree_augmented/processcontourtree/PiecewiseLinearFunction.h>

namespace caugmented_ns = vtkm::worklet::contourtree_augmented;
using DataValueType = vtkm::Float64;
using ValueArray = vtkm::cont::ArrayHandle<DataValueType>;
using BranchType = vtkm::worklet::contourtree_augmented::process_contourtree_inc::Branch<DataValueType>;
using PLFType = vtkm::worklet::contourtree_augmented::process_contourtree_inc::PiecewiseLinearFunction<DataValueType>;

#ifdef VTKH_PARALLEL
#include <mpi.h>

// This is from VTK-m diy mpi_cast.hpp. Need the make_DIY_MPI_Comm
namespace vtkmdiy
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


namespace vtkh
{

  template<typename T, typename S>
  void PrintArrayHandle( const vtkm::cont::ArrayHandle<T, S> &a, const char *name )
  {
    vtkm::Id s = a.GetNumberOfValues();
    auto p = a.ReadPortal();

    std::cout << "--- " << name << " - size: " << s << " ---\n";
    for(vtkm::Id i = 0; i < s ;++i)
    {
      if( p.Get(i) != (T)0 )
        std::cout << p.Get(i) << " ";
    }
    std::cout << "\n---\n";
  };


class SplitColumnComparer
{
public:
  SplitColumnComparer(double* range, int splitDim):
    m_range(range), m_splitDim(splitDim)
  {
  }
  bool operator() (int i,int j)
  {
    return m_range[i*6 + 2*m_splitDim] < m_range[j*6 + 2*m_splitDim];
  }
private:
  double* m_range;
  int m_splitDim;
};

ContourTree::ContourTree()
  : m_levels(5)
{

}

ContourTree::~ContourTree()
{

}

void
ContourTree::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
ContourTree::SetNumLevels(int levels)
{
  m_levels = levels;
}

std::vector<double>
ContourTree::GetIsoValues()
{
  return m_iso_values;
}

void ContourTree::PreExecute()
{
  Filter::PreExecute();
}

void ContourTree::PostExecute()
{
  Filter::PostExecute();
}

void ContourTree::DoExecute()
{
  vtkh::DataSet *old_input = this->m_input;

  // make sure we have a node-centered field
  bool valid_field = false;
  bool is_cell_assoc = m_input->GetFieldAssociation(m_field_name, valid_field) ==
                       vtkm::cont::Field::Association::CELL_SET;
  bool delete_input = false;
  if(valid_field && is_cell_assoc)
  {
    Recenter recenter;
    recenter.SetInput(m_input);
    recenter.SetField(m_field_name);
    recenter.SetResultAssoc(vtkm::cont::Field::Association::POINTS);
    recenter.Update();
    m_input = recenter.GetOutput();
    delete_input = true;
  }

  int mpi_rank = 0;

  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();
  assert(num_domains == 1);

#ifndef VTKH_PARALLEL
  vtkm::cont::DataSet inDataSet;
  vtkm::Id domain_id;

  this->m_input->GetDomain(0, inDataSet, domain_id);
  this->m_output->AddDomain(inDataSet, domain_id);

#else // VTKH_PARALLEL
  int mpi_size;

  // Setup VTK-h and VTK-m comm.
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));

  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  vtkm::cont::PartitionedDataSet inDataSet;

  vtkm::cont::ArrayHandle<vtkm::Id3> localBlockIndices;
  vtkm::cont::ArrayHandle<vtkm::Id3> localBlockOrigins;
  vtkm::cont::ArrayHandle<vtkm::Id3> localBlockSizes;
  vtkm::Id3 globalSize(0, 0, 0);
  vtkm::Id3 blocksPerDim(1, 1, 1);

  // Each rank has one or more blocks.
  // All ranks have the same number of blocks.
  double range[mpi_size][6];
  double localRange[6];

  vtkm::Id domain_id;
  vtkm::cont::DataSet dom;

  this->m_input->GetDomain(0, dom, domain_id);
  auto domRange = dom.GetCoordinateSystem(0).GetRange();
  inDataSet.AppendPartition(dom);

  if( mpi_size != 1 )
  {
    for(int j = 0; j < 3; ++j)
    {
      localRange[2*j] = domRange[j].Min;
      localRange[2*j + 1] = domRange[j].Max;
    }

#ifdef DEBUG
    std::ostringstream ostr;
    ostr << "rank: " << mpi_rank
       << " coord system range: " << dom.GetCoordinateSystem(0).GetRange() << std::endl;
    std::cout << ostr.str();
#endif

    int blocksPerRank = num_domains;
    MPI_Allgather(localRange, 2*3*blocksPerRank, MPI_DOUBLE,
                  range, 2*3*blocksPerRank, MPI_DOUBLE, mpi_comm);

#ifdef DEBUG
    {
      std::ostringstream ostr;
      ostr << "rank: " << mpi_rank << " values: " << std::endl;
      for(int i = 0; i < mpi_size; ++i)
      {
        for(int j = 0; j < 6; ++j)
          ostr << range[i][j] << " ";
        ostr << std::endl;
      }

      ostr << std::endl;
      std::cout << ostr.str();
    }
#endif

    // Compute globalSize
    for(int i = 0; i < mpi_size; ++i)
    {
      for(int j = 0; j < 3; ++j)
      {
        if(range[i][2*j+1] > globalSize[j])
          globalSize[j] = range[i][2*j+1];
      }
    }
    for(int j = 0; j < 3; ++j)
    {
      ++globalSize[j];
    }

    // Compute blocksPerDim
    double minRange[3] = {range[0][0], range[0][2], range[0][4]};
    int splitDim = -1;

    for(int i = 1; i < mpi_size; ++i)
    {
      for(int j = 0; j < 3; ++j)
      {
        if(range[i][2*j] != minRange[j])
        {
          blocksPerDim[j] = mpi_size;
          splitDim = j;

          i = mpi_size;  // Break outside loop too.
          break;
        }
      }
    }

    // Initalize localBlock variables
    localBlockIndices.Allocate(blocksPerRank);
    localBlockOrigins.Allocate(blocksPerRank);
    localBlockSizes.Allocate(blocksPerRank);

    auto localBlockIndicesPortal = localBlockIndices.WritePortal();
    auto localBlockOriginsPortal = localBlockOrigins.WritePortal();
    auto localBlockSizesPortal = localBlockSizes.WritePortal();

    // Compute localBlockIndices
    int start[mpi_size]; // start of the splitDim column
    for(int i = 0; i < mpi_size; ++i)
    {
      start[i] = i;
    }

    SplitColumnComparer comp(&range[0][0], splitDim);
    std::sort(start, start+mpi_size, comp);

    vtkm::Id3 id(0, 0, 0);

    id[splitDim] = start[mpi_rank];
    localBlockIndicesPortal.Set(0, id);

    // Compute localBlockOrigins
    id = vtkm::Id3(0, 0, 0);
    id[splitDim] = range[start[mpi_rank]][2*splitDim];
    localBlockOriginsPortal.Set(0, id);

    // Compute localBlockSizes
    id = globalSize;
    id[splitDim] = range[start[mpi_rank]][2*splitDim + 1] -
      range[start[mpi_rank]][2*splitDim] + 1;
    localBlockSizesPortal.Set(0, id);
  }
#endif // VTKH_PARALLEL

  bool useMarchingCubes = false;
  // Compute the fully augmented contour tree.
  // This should always be true for now in order for the isovalue selection to work.
  bool computeRegularStructure = true;

  //Convert the mesh of values into contour tree, pairs of vertex ids
  vtkm::filter::ContourTreeAugmented filter(useMarchingCubes,
                                            computeRegularStructure);

  std::vector<DataValueType> iso_values;

#ifdef VTKH_PARALLEL
  if(mpi_size != 1)
  {
    // blocksPerDim - number of blocks in each dim.
    // globalSize - extents - full size of the data.
    // localBlockIndices - the order of the data blocks index.
    // localBlockOrigins - block origins in extent.
    // localBlockSize - extents of the data.
    filter.SetSpatialDecomposition(
      blocksPerDim, globalSize, localBlockIndices, localBlockOrigins, localBlockSizes);
  }
#endif // VTKH_PARALLEL

  filter.SetActiveField(m_field_name);

#ifdef DEBUG
    std::cout << "--- inDataSet \n";
    inDataSet.PrintSummary( std::cout );
    std::cout << "--- end inDataSet \n";

#ifdef VTKH_PARALLEL
    ValueArray dataField;
    inDataSet.GetPartitions()[0].GetField(0).GetData().CopyTo(dataField);
    PrintArrayHandle( dataField, "dataField" );
#endif // VTKH_PARALLEL
#endif // DEBUG

    auto result = filter.Execute(inDataSet);

#ifdef DEBUG
    std::cout << "--- result \n";
    result.PrintSummary( std::cout );
    std::cout << "--- end result \n";

    {
    ValueArray dataField;
#ifdef VTKH_PARALLEL
    std::cout << "--- field name: " << m_field_name << std::endl;
    if( result.GetPartitions()[0].GetNumberOfFields() > 1 )
      result.GetPartitions()[0].GetField("values").GetData().CopyTo(dataField);
    else
      result.GetPartitions()[0].GetField(0).GetData().CopyTo(dataField);
#endif // VTKH_PARALLEL
    PrintArrayHandle( dataField, "result dataField" );
    }
#endif // DEBUG

  m_iso_values.resize(m_levels);
  if(mpi_rank == 0)
  {
    DataValueType eps = 0.00001;        // Error away from critical point
    vtkm::Id numComp = m_levels + 1;    // Number of components the tree should be simplified to
    vtkm::Id contourType = 0;           // Approach to be used to select contours based on the tree
    vtkm::Id contourSelectMethod = 0;   // Method to be used to compute the relevant iso values
    bool usePersistenceSorter = true;

    // Compute the branch decomposition
    // Compute the volume for each hyperarc and superarc
    caugmented_ns::IdArrayType superarcIntrinsicWeight;
    caugmented_ns::IdArrayType superarcDependentWeight;
    caugmented_ns::IdArrayType supernodeTransferWeight;
    caugmented_ns::IdArrayType hyperarcDependentWeight;

    caugmented_ns::ProcessContourTree::ComputeVolumeWeightsSerial(
      filter.GetContourTree(),
      filter.GetNumIterations(),
      superarcIntrinsicWeight,  // (output)
      superarcDependentWeight,  // (output)
      supernodeTransferWeight,  // (output)
      hyperarcDependentWeight); // (output)

#ifdef DEBUG
    PrintArrayHandle( superarcIntrinsicWeight, "superarcIntrinsicWeight" );
    PrintArrayHandle( superarcDependentWeight, "superarcDependentWeight" );
    PrintArrayHandle( supernodeTransferWeight, "superarcDependentWeight" );
    PrintArrayHandle( hyperarcDependentWeight, "hyperarcDependentWeight" );
#endif // DEBUG

    // Compute the branch decomposition by volume
    caugmented_ns::IdArrayType whichBranch;
    caugmented_ns::IdArrayType branchMinimum;
    caugmented_ns::IdArrayType branchMaximum;
    caugmented_ns::IdArrayType branchSaddle;
    caugmented_ns::IdArrayType branchParent;

    caugmented_ns::ProcessContourTree::ComputeVolumeBranchDecompositionSerial(
      filter.GetContourTree(),
      superarcDependentWeight,
      superarcIntrinsicWeight,
      whichBranch,               // (output)
      branchMinimum,             // (output)
      branchMaximum,             // (output)
      branchSaddle,              // (output)
      branchParent);             // (output)

#ifdef DEBUG
    PrintArrayHandle( whichBranch,   "whichBranch" );
    PrintArrayHandle( branchMinimum, "branchMinimum" );
    PrintArrayHandle( branchMaximum, "branchMaximum" );
    PrintArrayHandle( branchSaddle,  "branchSaddle" );
    PrintArrayHandle( branchParent,  "branchParent" );
#endif // DEBUG

    // This is from ContourTree.h
    using IdArrayType = vtkm::cont::ArrayHandle<vtkm::Id>;

    // Create explicit representation of the branch decompostion from the array representation
#ifndef VTKH_PARALLEL
    ValueArray dataField;
    inDataSet.GetField(0).GetData().CopyTo(dataField);

    bool dataFieldIsSorted = false;
#else // VTKH_PARALLEL
    ValueArray dataField;
    bool dataFieldIsSorted;

    // Why do this? Because right now aa is an int array and we want a double array.
    if(mpi_size == 1)
    {
      inDataSet.GetPartition(0).GetField(0).GetData().CopyTo(dataField);
      dataFieldIsSorted = false;
    }else{
      if( result.GetPartitions()[0].GetNumberOfFields() > 1 )
        result.GetPartitions()[0].GetField("values").GetData().CopyTo(dataField);
      else
        result.GetPartitions()[0].GetField(0).GetData().CopyTo(dataField);
      dataFieldIsSorted = true;
    }
#endif // VTKH_PARALLEL
#ifdef DEBUG
    //resultDataset.PrintSummary( std::cerr );
    //PrintArrayHandle( (IdArrayType)aa );
    PrintArrayHandle( dataField, "dataField" );
#endif // DEBUG

    BranchType* branchDecompostionRoot = caugmented_ns::ProcessContourTree::ComputeBranchDecomposition<DataValueType>(
      filter.GetContourTree().Superparents,
      filter.GetContourTree().Supernodes,
      whichBranch,
      branchMinimum,
      branchMaximum,
      branchSaddle,
      branchParent,
      filter.GetSortOrder(),
      dataField,
      dataFieldIsSorted
    );

    // Simplify the contour tree of the branch decompostion
    branchDecompostionRoot->SimplifyToSize(numComp, usePersistenceSorter);

    // Compute the relevant iso-values
    switch(contourSelectMethod)
    {
    default:
    case 0:
      {
        branchDecompostionRoot->GetRelevantValues(contourType, eps, iso_values);
      }
      break;
    case 1:
      {
        PLFType plf;
        branchDecompostionRoot->AccumulateIntervals(contourType, eps, plf);
        iso_values = plf.nLargest(m_levels);
      }
      break;
    }

    // Print the compute iso values
    std::sort(iso_values.begin(), iso_values.end());

#ifdef DEBUG
    std::cout << "Isovalues: ";
    for(DataValueType val : iso_values) std::cout << val << " ";
    std::cout << std::endl;
#endif

    // Unique isovalues
    std::vector<DataValueType>::iterator it = std::unique (iso_values.begin(), iso_values.end());
    iso_values.resize( std::distance(iso_values.begin(), it) );

#ifdef DEBUG
    std::cout << iso_values.size() << "  Unique Isovalues: ";
    for(DataValueType val : iso_values) std::cout << val << " ";
    std::cout << std::endl;

    std::cout<<"Acrs : " << filter.GetContourTree().Arcs.GetNumberOfValues() <<std::endl;
#endif

    for(size_t x = 0; x < iso_values.size(); ++x)
    {
      m_iso_values[x] = iso_values[x];
    }

    if(branchDecompostionRoot)
    {
      delete branchDecompostionRoot;
    }
  }

#ifdef VTKH_PARALLEL
  MPI_Bcast(&m_iso_values[0], m_levels, MPI_DOUBLE, 0, mpi_comm);
#endif // VTKH_PARALLEL

  if(delete_input)
  {
    delete m_input;
    this->m_input = old_input;
  }
}

std::string
ContourTree::GetName() const
{
  return "vtkh::ContourTree";
}

} //  namespace vtkh
