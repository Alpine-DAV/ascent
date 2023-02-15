#include <vtkh/filters/ContourTree.hpp>

#include <vtkh/filters/Recenter.hpp>

// vtkm includes
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/Storage.h>
#include <vtkm/internal/Configure.h>
#include <vtkm/filter/ContourTreeUniformAugmented.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/PrintVectors.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ProcessContourTree.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/Branch.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/processcontourtree/PiecewiseLinearFunction.h>

#include <vtkh/filters/GhostStripper.hpp> // AES

namespace caugmented_ns = vtkm::worklet::contourtree_augmented;
using DataValueType = vtkm::Float64;
using ValueArray = vtkm::cont::ArrayHandle<DataValueType>;
using BranchType = vtkm::worklet::contourtree_augmented::process_contourtree_inc::Branch<DataValueType>;
using PLFType = vtkm::worklet::contourtree_augmented::process_contourtree_inc::PiecewiseLinearFunction<DataValueType>;

#ifdef VTKH_PARALLEL
#include <mpi.h>

static void ShiftLogicalOriginToZero(vtkm::cont::PartitionedDataSet& pds)
{
  // Shift the logical origin (minimum of LocalPointIndexStart) to zero
  // along each dimension

  // Compute minimum global point index start for all data sets on this MPI rank
  std::vector<vtkm::Id> minimumGlobalPointIndexStartThisRank;
  using ds_const_iterator = vtkm::cont::PartitionedDataSet::const_iterator;
  for (ds_const_iterator ds_it = pds.cbegin(); ds_it != pds.cend(); ++ds_it)
  {
    ds_it->GetCellSet().CastAndCallForTypes<vtkm::cont::CellSetListStructured>(
      [&minimumGlobalPointIndexStartThisRank](const auto& css)
      {
        minimumGlobalPointIndexStartThisRank.resize(css.Dimension,
                                                    std::numeric_limits<vtkm::Id>::max());
        for (vtkm::IdComponent d = 0; d < css.Dimension; ++d)
        {
          minimumGlobalPointIndexStartThisRank[d] =
            std::min(minimumGlobalPointIndexStartThisRank[d], css.GetGlobalPointIndexStart()[d]);
        }
      });
  }

  // Perform global reduction to find GlobalPointDimensions across all ranks
  std::vector<vtkm::Id> minimumGlobalPointIndexStart;
  auto comm = vtkm::cont::EnvironmentTracker::GetCommunicator();
  vtkmdiy::mpi::all_reduce(comm,
                           minimumGlobalPointIndexStartThisRank,
                           minimumGlobalPointIndexStart,
                           vtkmdiy::mpi::minimum<vtkm::Id>{});

  // Shift all cell sets so that minimum global point index start
  // along each dimension is zero
  using ds_iterator = vtkm::cont::PartitionedDataSet::iterator;
  for (ds_iterator ds_it = pds.begin(); ds_it != pds.end(); ++ds_it)
  {
    // This does not work, i.e., it does not really change the cell set for the DataSet
    ds_it->GetCellSet().CastAndCallForTypes<vtkm::cont::CellSetListStructured>(
      [&minimumGlobalPointIndexStart, &ds_it](auto& css) {
        auto pointIndexStart = css.GetGlobalPointIndexStart();
        typename std::remove_reference_t<decltype(css)>::SchedulingRangeType shiftedPointIndexStart;
        for (vtkm::IdComponent d = 0; d < css.Dimension; ++d)
        {
          shiftedPointIndexStart[d] = pointIndexStart[d] - minimumGlobalPointIndexStart[d];
        }
        css.SetGlobalPointIndexStart(shiftedPointIndexStart);
        // Why is the following necessary? Shouldn't it be sufficient to update the
        // CellSet through the reference?
        ds_it->SetCellSet(css);
      });
  }
}

static void ComputeGlobalPointSize(vtkm::cont::PartitionedDataSet& pds)
{
  // Compute GlobalPointDimensions as maximum of GlobalPointIndexStart + PointDimensions
  // for each dimension across all blocks

  // Compute GlobalPointDimensions for all data sets on this MPI rank
  std::vector<vtkm::Id> globalPointDimensionsThisRank;
  using ds_const_iterator = vtkm::cont::PartitionedDataSet::const_iterator;
  for (ds_const_iterator ds_it = pds.cbegin(); ds_it != pds.cend(); ++ds_it)
  {
    ds_it->GetCellSet().CastAndCallForTypes<vtkm::cont::CellSetListStructured>(
      [&globalPointDimensionsThisRank](const auto& css) {
        globalPointDimensionsThisRank.resize(css.Dimension, -1);
        for (vtkm::IdComponent d = 0; d < css.Dimension; ++d)
        {
          globalPointDimensionsThisRank[d] =
            std::max(globalPointDimensionsThisRank[d],
                     css.GetGlobalPointIndexStart()[d] + css.GetPointDimensions()[d]);
        }
      });
  }

  // Perform global reduction to find GlobalPointDimensions across all ranks
  std::vector<vtkm::Id> globalPointDimensions;
  auto comm = vtkm::cont::EnvironmentTracker::GetCommunicator();
  vtkmdiy::mpi::all_reduce(
    comm, globalPointDimensionsThisRank, globalPointDimensions, vtkmdiy::mpi::maximum<vtkm::Id>{});

  // Set this information in all cell sets
  using ds_iterator = vtkm::cont::PartitionedDataSet::iterator;
  for (ds_iterator ds_it = pds.begin(); ds_it != pds.end(); ++ds_it)
  {
    // This does not work, i.e., it does not really change the cell set for the DataSet
    ds_it->GetCellSet().CastAndCallForTypes<vtkm::cont::CellSetListStructured>(
      [&globalPointDimensions, &ds_it](auto& css) {
        typename std::remove_reference_t<decltype(css)>::SchedulingRangeType gpd;
        for (vtkm::IdComponent d = 0; d < css.Dimension; ++d)
        {
          gpd[d] = globalPointDimensions[d];
        }
        css.SetGlobalPointDimensions(gpd);
        // Why is the following necessary? Shouldn't it be sufficient to update the
        // CellSet through the reference?
        ds_it->SetCellSet(css);
      });
  }
}

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

  bool has_ghosts = false; // Experimental when enabled .... 

  if(has_ghosts)
  {
    vtkh::GhostStripper stripper;

    stripper.SetInput(m_input);
    stripper.SetField("ascent_ghosts");
    stripper.SetMinValue(0);
    stripper.SetMaxValue(0);
    stripper.Update();
    vtkh::DataSet* stripped_input = stripper.GetOutput(); // AES FIX THIS ...

    if (delete_input) {
       delete(m_input);
    }

    m_input = stripped_input;
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

  vtkm::Id domain_id;
  vtkm::cont::DataSet dom;

  this->m_input->GetDomain(0, dom, domain_id);
  inDataSet.AppendPartition(dom);

#ifdef DEBUG
  if( mpi_size != 1 )
  {
    std::ostringstream ostr;
    ostr << "rank: " << mpi_rank
       << " coord system range: " << dom.GetCoordinateSystem(0).GetRange() << std::endl;
    std::cout << ostr.str();
  }
#endif
#endif // VTKH_PARALLEL

  bool useMarchingCubes = false;
  // Compute the fully augmented contour tree.
  // This should always be true for now in order for the isovalue selection to work.
  bool computeRegularStructure = true;

  //Convert the mesh of values into contour tree, pairs of vertex ids
  vtkm::filter::ContourTreeAugmented filter(useMarchingCubes,
                                            computeRegularStructure);

  std::vector<DataValueType> iso_values;

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

#ifdef VTKH_PARALLEL
    ShiftLogicalOriginToZero(inDataSet);
    ComputeGlobalPointSize(inDataSet);
#endif // VTKH_PARALLEL

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
