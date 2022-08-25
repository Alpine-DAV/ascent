#include "DataSet.hpp"

#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>

// FIXME:UDA: vtkm_dataset_info depends on vtkm::rendering
#include <vtkh/utils/vtkm_dataset_info.hpp>
// std includes
#include <limits>
#include <sstream>
//vtkm includes
#include <vtkm/cont/Error.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/TryExecute.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/DispatcherMapField.h>
#ifdef VTKH_PARALLEL
  #include <mpi.h>
#endif
namespace vtkh {
namespace detail
{
//
// returns true if all ranks say true
//
bool GlobalAgreement(bool local)
{
  bool agreement = local;
#ifdef VTKH_PARALLEL
  int local_boolean = local ? 1 : 0;
  int global_boolean;
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean != vtkh::GetMPISize())
  {
    agreement = false;
  }
#endif
  return agreement;
}

bool GlobalSomeoneAgrees(bool local)
{
  bool agreement = local;
#ifdef VTKH_PARALLEL
  int local_boolean = local ? 1 : 0;
  int global_boolean;
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean == 0)
  {
    agreement = false;
  }
#endif
  return agreement;
}

template<typename T>
class MemSetWorklet : public vtkm::worklet::WorkletMapField
{
protected:
  T Value;
public:
  VTKM_CONT
  MemSetWorklet(const T value)
    : Value(value)
  {
  }

  typedef void ControlSignature(FieldOut);
  typedef void ExecutionSignature(_1);

  VTKM_EXEC
  void operator()(T &value) const
  {
    value = Value;
  }
}; //class MemSetWorklet

template<typename T>
void MemSet(vtkm::cont::ArrayHandle<T> &array, const T value, const vtkm::Id num_values)
{
  array.Allocate(num_values);
  vtkm::worklet::DispatcherMapField<MemSetWorklet<T>>(MemSetWorklet<T>(value))
    .Invoke(array);
}

} // namespace detail

bool
DataSet::OneDomainPerRank() const
{
  bool one = GetNumberOfDomains() == 1;
  return detail::GlobalAgreement(one);
}

void
DataSet::AddDomain(vtkm::cont::DataSet data_set, vtkm::Id domain_id)
{
  if(m_domains.size() != 0)
  {
    // TODO: verify same number / name of:
    // cellsets coords and fields
  }

  assert(m_domains.size() == m_domain_ids.size());
  m_domains.push_back(data_set);
  m_domain_ids.push_back(domain_id);
}

vtkm::cont::Field
DataSet::GetField(const std::string &field_name, const vtkm::Id domain_index)
{
  assert(domain_index >= 0);
  assert(domain_index < m_domains.size());

  return m_domains[domain_index].GetField(field_name);
}

vtkm::cont::DataSet&
DataSet::GetDomain(const vtkm::Id index)
{
  const size_t num_domains = m_domains.size();

  if(index >= num_domains || index < 0)
  {
    std::stringstream msg;
    msg<<"Get domain call failed. Invalid index "<<index
       <<" in "<<num_domains<<" domains.";
    throw Error(msg.str());
  }

  return  m_domains[index];

}

std::vector<vtkm::Id>
DataSet::GetDomainIds() const
{
  return m_domain_ids;
}

void
DataSet::GetDomain(const vtkm::Id index,
                   vtkm::cont::DataSet &data_set,
                   vtkm::Id &domain_id)
{
  const size_t num_domains = m_domains.size();

  if(index >= num_domains || index < 0)
  {
    std::stringstream msg;
    msg<<"Get domain call failed. Invalid index "<<index
       <<" in "<<num_domains<<" domains.";
    throw Error(msg.str());
  }

  data_set = m_domains[index];
  domain_id = m_domain_ids[index];

}

vtkm::Id
DataSet::GetNumberOfDomains() const
{
  return static_cast<vtkm::Id>(m_domains.size());
}

vtkm::Id
DataSet::GetNumberOfCells() const
{
  vtkm::Id num_cells = 0;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    num_cells += m_domains[i].GetCellSet().GetNumberOfCells();
  }
  return num_cells;
}

vtkm::Id
DataSet::GetGlobalNumberOfCells() const
{
  vtkm::Id num_cells = GetNumberOfCells();;
#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  long long int local_cells = static_cast<long long int>(num_cells);
  long long int global_cells = 0;
  MPI_Allreduce(&local_cells,
                &global_cells,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                mpi_comm);
  num_cells = global_cells;
#endif
  return num_cells;
}



vtkm::Id
DataSet::GetGlobalNumberOfDomains() const
{
  vtkm::Id domains = this->GetNumberOfDomains();
#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  int local_doms = static_cast<int>(domains);
  int global_doms = 0;
  MPI_Allreduce(&local_doms,
                &global_doms,
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);
  domains = global_doms;
#endif
  return domains;
}

vtkm::Bounds
DataSet::GetDomainBounds(const int &domain_index,
                         vtkm::Id coordinate_system_index) const
{
  const vtkm::Id index = coordinate_system_index;
  vtkm::cont::CoordinateSystem coords;
  try
  {
    coords = m_domains[domain_index].GetCoordinateSystem(index);
  }
  catch (const vtkm::cont::Error &error)
  {
    std::stringstream msg;
    msg<<"GetBounds call failed. vtk-m error was encountered while "
       <<"attempting to get coordinate system "<<index<<" from "
       <<"domaim "<<domain_index<<". vtkm error message: "<<error.GetMessage();
    throw Error(msg.str());
  }

  return coords.GetBounds();
}


vtkm::Bounds
DataSet::GetBounds(vtkm::Id coordinate_system_index) const
{
  const vtkm::Id index = coordinate_system_index;
  const size_t num_domains = m_domains.size();

  vtkm::Bounds bounds;

  for(size_t i = 0; i < num_domains; ++i)
  {
    vtkm::Bounds dom_bounds = GetDomainBounds(i, index);
    bounds.Include(dom_bounds);
  }

  return bounds;
}

vtkm::Bounds
DataSet::GetGlobalBounds(vtkm::Id coordinate_system_index) const
{
  VTKH_DATA_OPEN("GetGlobalBounds");
  vtkm::Bounds bounds;
  bounds = GetBounds(coordinate_system_index);

#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

  vtkm::Float64 x_min = bounds.X.Min;
  vtkm::Float64 x_max = bounds.X.Max;
  vtkm::Float64 y_min = bounds.Y.Min;
  vtkm::Float64 y_max = bounds.Y.Max;
  vtkm::Float64 z_min = bounds.Z.Min;
  vtkm::Float64 z_max = bounds.Z.Max;
  vtkm::Float64 global_x_min = 0;
  vtkm::Float64 global_x_max = 0;
  vtkm::Float64 global_y_min = 0;
  vtkm::Float64 global_y_max = 0;
  vtkm::Float64 global_z_min = 0;
  vtkm::Float64 global_z_max = 0;

  MPI_Allreduce((void *)(&x_min),
                (void *)(&global_x_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                mpi_comm);

  MPI_Allreduce((void *)(&x_max),
                (void *)(&global_x_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                mpi_comm);

  MPI_Allreduce((void *)(&y_min),
                (void *)(&global_y_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                mpi_comm);

  MPI_Allreduce((void *)(&y_max),
                (void *)(&global_y_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                mpi_comm);

  MPI_Allreduce((void *)(&z_min),
                (void *)(&global_z_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                mpi_comm);

  MPI_Allreduce((void *)(&z_max),
                (void *)(&global_z_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                mpi_comm);

  bounds.X.Min = global_x_min;
  bounds.X.Max = global_x_max;
  bounds.Y.Min = global_y_min;
  bounds.Y.Max = global_y_max;
  bounds.Z.Min = global_z_min;
  bounds.Z.Max = global_z_max;
#endif
  VTKH_DATA_CLOSE();
  return bounds;
}

vtkm::cont::ArrayHandle<vtkm::Range>
DataSet::GetRange(const std::string &field_name) const
{
  const size_t num_domains = m_domains.size();

  vtkm::cont::ArrayHandle<vtkm::Range> range;
  vtkm::Id num_components = 0;

  for(size_t i = 0; i < num_domains; ++i)
  {
    if(!m_domains[i].HasField(field_name))
    {
      continue;
    }

    const vtkm::cont::Field &field = m_domains[i].GetField(field_name);
    vtkm::cont::ArrayHandle<vtkm::Range> sub_range;
    sub_range = field.GetRange();

    vtkm::Id components = sub_range.ReadPortal().GetNumberOfValues();

    // first range with data. Set range and keep looking
    if(num_components == 0)
    {
      num_components = components;
      range = sub_range;
      continue;
    }

    // This is not the first valid range encountered.
    // Validate and expand the current range
    if(components != num_components)
    {
      std::stringstream msg;
      msg<<"GetRange call failed. The number of components ("<<components<<") in field "
         <<field_name<<" from domain "<<i<<" does not match the number of components "
         <<"("<<num_components<<") in another domain";
      throw Error(msg.str());
    }

    for(vtkm::Id c = 0; c < components; ++c)
    {
      vtkm::Range s_range = sub_range.ReadPortal().Get(c);
      vtkm::Range c_range = range.ReadPortal().Get(c);
      c_range.Include(s_range);
      range.WritePortal().Set(c, c_range);
    }
  }
  return range;
}

vtkm::cont::ArrayHandle<vtkm::Range>
DataSet::GetGlobalRange(const std::string &field_name) const
{
  VTKH_DATA_OPEN("GetGlobalRange");
  vtkm::cont::ArrayHandle<vtkm::Range> range;
  range = GetRange(field_name);

#ifdef VTKH_PARALLEL
  vtkm::Id num_components = range.GetNumberOfValues();
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  //
  // it is possible to have an empty dataset at one of the ranks
  // so we must check for this so MPI comm does not hang.
  // We also want to check for num components mis-match
  //
  int *global_components = new int[vtkh::GetMPISize()];
  int comps = static_cast<int>(num_components);

  MPI_Allgather(&comps,
                1,
                MPI_INT,
                global_components,
                1,
                MPI_INT,
                mpi_comm);

  int components = 0;
  //
  // find the largest component
  //
  for(int i = 0; i < vtkh::GetMPISize(); ++i)
  {
    if(components == 0 && global_components[i] != 0)
    {
      components = global_components[i];
      continue;
    }

    // verify that this matches are current components
    if(global_components[i] != 0 && components != global_components[i])
    {
      std::stringstream msg;
      msg<<"GetRange call failed. The number of components ("
         <<global_components[i]<<") in field "
         <<field_name<<" from rank"<<i<<" does not match the number of components in"
         <<" the other ranks "<<components;
      throw Error(msg.str());
    }
  }

  // at least one rank has data. Find the global range
  if(components != 0)
  {
    range.Allocate(components);
    for(int i = 0; i < components; ++i)
    {

      vtkm::Range c_range = range.ReadPortal().Get(i);

      vtkm::Float64 local_min;
      vtkm::Float64 local_max;

      if(num_components != 0)
      {
        local_min = c_range.Min;
        local_max = c_range.Max;
      }
      else
      {
        local_min = std::numeric_limits<vtkm::Float64>::max();
        local_max = std::numeric_limits<vtkm::Float64>::lowest();
      }

      vtkm::Float64 global_min = 0;
      vtkm::Float64 global_max = 0;

      MPI_Allreduce((void *)(&local_min),
                    (void *)(&global_min),
                    1,
                    MPI_DOUBLE,
                    MPI_MIN,
                    mpi_comm);

      MPI_Allreduce((void *)(&local_max),
                    (void *)(&global_max),
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    mpi_comm);
      c_range.Min = global_min;
      c_range.Max = global_max;
      range.WritePortal().Set(i, c_range);
    }
  }

  delete[] global_components;
#endif
  VTKH_DATA_CLOSE();
  return range;
}

void
DataSet::PrintSummary(std::ostream &stream) const
{
  for(size_t dom = 0; dom < m_domains.size(); ++dom)
  {
    stream<<"Domain "<<m_domain_ids[dom]<<"\n";
    m_domains[dom].PrintSummary(stream);
  }
}

bool
DataSet::IsEmpty() const
{
  bool is_empty = true;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    auto cellset = m_domains[i].GetCellSet();
    if(cellset.GetNumberOfCells() > 0)
    {
      is_empty = false;
      break;
    }
  }

  return is_empty;
}

bool
DataSet::GlobalIsEmpty() const
{
  bool is_empty = IsEmpty();
  is_empty = detail::GlobalAgreement(is_empty);
  return is_empty;
}

bool
DataSet::IsPointMesh() const
{
  const bool is_empty = GlobalIsEmpty();
  if(is_empty) return false;

  // since we are not empty, start with the affirmative is_points.
  // if someone is not points, the we will figure it out here
  bool is_points = true;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    const vtkm::cont::DataSet &dom = m_domains[i];
    vtkm::UInt8 shape_type;
    bool single_type = VTKMDataSetInfo::IsSingleCellShape(dom.GetCellSet(), shape_type);

    if(dom.GetCellSet().GetNumberOfCells() > 0)
    {
      is_points = (single_type && (shape_type == 1)) && is_points;
    }
  }

  is_points = detail::GlobalAgreement(is_points);
  return is_points;
}

bool
DataSet::IsUnstructured() const
{
  bool is_unstructured = true;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    const vtkm::cont::DataSet &dom = m_domains[i];
    int dims;
    is_unstructured = !VTKMDataSetInfo::IsStructured(dom, dims) && is_unstructured;

    (void) dims;

    if(!is_unstructured)
    {
      break;
    }
  }

  is_unstructured = detail::GlobalAgreement(is_unstructured);

  return is_unstructured;
}

bool
DataSet::IsStructured(int &topological_dims) const
{
  topological_dims = -1;
  bool is_structured = true;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    const vtkm::cont::DataSet &dom = m_domains[i];
    int dims;
    is_structured = VTKMDataSetInfo::IsStructured(dom, dims) && is_structured;

    if(i == 0)
    {
      topological_dims = dims;
    }

    if(!is_structured || dims != topological_dims)
    {
      topological_dims = -1;
      break;
    }
  }

  is_structured = detail::GlobalAgreement(is_structured);

  if(!is_structured)
  {
    topological_dims = -1;
  }
  return is_structured;
}

void
DataSet::SetCycle(const vtkm::UInt64 cycle)
{
  m_cycle = cycle;
}

vtkm::UInt64
DataSet::GetCycle() const
{
  return m_cycle;
}

void
DataSet::SetTime(const vtkm::UInt64 time)
{
  m_time = time;
}

vtkm::UInt64
DataSet::GetTime() const
{
  return m_time;
}

DataSet::DataSet()
  : m_cycle(0), m_time(0)
{
}

DataSet::~DataSet()
{
}

vtkm::cont::DataSet&
DataSet::GetDomainById(const vtkm::Id domain_id)
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    if(m_domain_ids[i] == domain_id) return m_domains[i];
  }

  std::stringstream msg;
  msg<<"GetDomainById call failed. Invalid domain_id "<<domain_id;
  throw Error(msg.str());
}

bool DataSet::HasDomainId(const vtkm::Id &domain_id) const
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    if(m_domain_ids[i] == domain_id) return true;
  }

  return false;
}

void
DataSet::AddConstantPointField(const vtkm::Float32 value, const std::string fieldname)
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    vtkm::Id num_points = m_domains[i].GetCoordinateSystem().GetData().GetNumberOfValues();
    vtkm::cont::ArrayHandle<vtkm::Float32> array;
    detail::MemSet(array, value, num_points);
    vtkm::cont::Field field(fieldname, vtkm::cont::Field::Association::Points, array);
    m_domains[i].AddField(field);
  }
}

bool
DataSet::FieldExists(const std::string &field_name) const
{
  bool exists = false;

  const size_t size = m_domains.size();
  for(size_t i = 0; i < size; ++i)
  {
    if(m_domains[i].HasField(field_name))
    {
      exists = true;
      break;
    }
  }
  return exists;
}

bool
DataSet::GlobalFieldExists(const std::string &field_name) const
{
  bool exists = FieldExists(field_name);
#ifdef VTKH_PARALLEL
  int local_boolean = exists ? 1 : 0;
  int global_boolean;

  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);


  if(global_boolean > 0)
  {
    exists = true;
  }
  else
  {
    // this is technically not needed but added for clarity
    exists = false;
  }
#endif
  return exists;
}

vtkm::cont::Field::Association
DataSet::GetFieldAssociation(const std::string field_name, bool &valid_field) const
{
  valid_field = true;
  if(!this->GlobalFieldExists(field_name))
  {
    valid_field = false;
    return vtkm::cont::Field::Association::Any;
  }

  int assoc_id = -1;
  if(this->FieldExists(field_name))
  {
    const size_t num_domains = m_domains.size();
    vtkm::Bounds bounds;

    vtkm::cont::Field::Association local_assoc;
    for(size_t i = 0; i < num_domains; ++i)
    {
      vtkm::cont::DataSet dom = m_domains[i];
      if(dom.HasField(field_name))
      {
        local_assoc = dom.GetField(field_name).GetAssociation();
        if(local_assoc == vtkm::cont::Field::Association::Any)
        {
          assoc_id = 0;
        }
        else if ( local_assoc == vtkm::cont::Field::Association::WholeMesh)
        {
          assoc_id = 1;
        }
        else if ( local_assoc == vtkm::cont::Field::Association::Points)
        {
          assoc_id = 2;
        }
        else if ( local_assoc == vtkm::cont::Field::Association::Cells)
        {
          assoc_id = 3;
        }
        break;
      }
    }
  }

#ifdef VTKH_PARALLEL

  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());


  int *global_assocs = new int[vtkh::GetMPISize()];

  MPI_Allgather(&assoc_id,
                1,
                MPI_INT,
                global_assocs,
                1,
                MPI_INT,
                mpi_comm);

  int id = -1;

  for(int i = 0; i < vtkh::GetMPISize(); ++i)
  {
    if(global_assocs[i] != -1)
    {
      if(id != -1 && global_assocs[i] != id)
      {
        std::stringstream msg;
        msg<<"field "<< field_name
           <<" has inconsistent associations";;
        throw Error(msg.str());
      }
      else
      {
        id = std::max(id, global_assocs[i]);
      }
    }
  }
  assoc_id = id;
  delete[] global_assocs;
#endif

  vtkm::cont::Field::Association assoc;

  if(assoc_id == 0)
  {
    assoc = vtkm::cont::Field::Association::Any;
  }
  else if ( assoc_id == 1)
  {
    assoc = vtkm::cont::Field::Association::WholeMesh;
  }
  else if ( assoc_id == 2)
  {
    assoc = vtkm::cont::Field::Association::Points;
  }
  else if ( assoc_id == 3)
  {
    assoc = vtkm::cont::Field::Association::Cells;
  }
  else
  {
    throw Error("Get association: unknown association");
  }
  return assoc;
}

vtkm::Id DataSet::NumberOfComponents(const std::string &field_name) const
{
  int num_components = 0;

  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    if(m_domains[i].HasField(field_name))
    {
      num_components = m_domains[i].GetField(field_name).GetData().GetNumberOfComponentsFlat();
      break;
    }
  }

#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

  int global_comps;
  MPI_Allreduce((void *)(&num_components),
                (void *)(&global_comps),
                1,
                MPI_INT,
                MPI_MAX,
                mpi_comm);

  num_components = global_comps;
#endif
  return num_components;
}

} // namspace vtkh
