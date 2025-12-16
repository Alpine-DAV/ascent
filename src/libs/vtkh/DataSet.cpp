#include "DataSet.hpp"

#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>

// FIXME:UDA: viskores_dataset_info depends on viskores::rendering
#include <vtkh/utils/viskores_dataset_info.hpp>
// std includes
#include <limits>
#include <sstream>
//viskores includes
#include <viskores/cont/Error.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/TryExecute.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/worklet/DispatcherMapField.h>
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
class MemSetWorklet : public viskores::worklet::WorkletMapField
{
protected:
  T Value;
public:
  VISKORES_CONT
  MemSetWorklet(const T value)
    : Value(value)
  {
  }

  typedef void ControlSignature(FieldOut);
  typedef void ExecutionSignature(_1);

  VISKORES_EXEC
  void operator()(T &value) const
  {
    value = Value;
  }
}; //class MemSetWorklet

template<typename T>
void MemSet(viskores::cont::ArrayHandle<T> &array, const T value, const viskores::Id num_values)
{
  array.Allocate(num_values);
  viskores::worklet::DispatcherMapField<MemSetWorklet<T>>(MemSetWorklet<T>(value))
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
DataSet::AddDomain(viskores::cont::DataSet data_set, viskores::Id domain_id)
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

viskores::cont::Field
DataSet::GetField(const std::string &field_name, const viskores::Id domain_index)
{
  assert(domain_index >= 0);
  assert(domain_index < m_domains.size());

  return m_domains[domain_index].GetField(field_name);
}

viskores::cont::DataSet&
DataSet::GetDomain(const viskores::Id index)
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

std::vector<viskores::Id>
DataSet::GetDomainIds() const
{
  return m_domain_ids;
}

void
DataSet::GetDomain(const viskores::Id index,
                   viskores::cont::DataSet &data_set,
                   viskores::Id &domain_id)
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

viskores::Id
DataSet::GetNumberOfDomains() const
{
  return static_cast<viskores::Id>(m_domains.size());
}

viskores::Id
DataSet::GetNumberOfCells() const
{
  viskores::Id num_cells = 0;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    num_cells += m_domains[i].GetCellSet().GetNumberOfCells();
  }
  return num_cells;
}

viskores::Id
DataSet::GetGlobalNumberOfCells() const
{
  viskores::Id num_cells = GetNumberOfCells();;
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



viskores::Id
DataSet::GetGlobalNumberOfDomains() const
{
  viskores::Id domains = this->GetNumberOfDomains();
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

viskores::Bounds
DataSet::GetDomainBounds(const int &domain_index,
                         viskores::Id coordinate_system_index) const
{
  const viskores::Id index = coordinate_system_index;
  viskores::cont::CoordinateSystem coords;
  try
  {
    coords = m_domains[domain_index].GetCoordinateSystem(index);
  }
  catch (const viskores::cont::Error &error)
  {
    std::stringstream msg;
    msg<<"GetBounds call failed. viskores error was encountered while "
       <<"attempting to get coordinate system "<<index<<" from "
       <<"domaim "<<domain_index<<". viskores error message: "<<error.GetMessage();
    throw Error(msg.str());
  }

  return coords.GetBounds();
}


viskores::Bounds
DataSet::GetBounds(viskores::Id coordinate_system_index) const
{
  const viskores::Id index = coordinate_system_index;
  const size_t num_domains = m_domains.size();

  viskores::Bounds bounds;

  for(size_t i = 0; i < num_domains; ++i)
  {
    viskores::Bounds dom_bounds = GetDomainBounds(i, index);
    bounds.Include(dom_bounds);
  }

  return bounds;
}

viskores::Bounds
DataSet::GetGlobalBounds(viskores::Id coordinate_system_index) const
{
  VTKH_DATA_OPEN("GetGlobalBounds");
  viskores::Bounds bounds;
  bounds = GetBounds(coordinate_system_index);

#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

  viskores::Float64 x_min = bounds.X.Min;
  viskores::Float64 x_max = bounds.X.Max;
  viskores::Float64 y_min = bounds.Y.Min;
  viskores::Float64 y_max = bounds.Y.Max;
  viskores::Float64 z_min = bounds.Z.Min;
  viskores::Float64 z_max = bounds.Z.Max;
  viskores::Float64 global_x_min = 0;
  viskores::Float64 global_x_max = 0;
  viskores::Float64 global_y_min = 0;
  viskores::Float64 global_y_max = 0;
  viskores::Float64 global_z_min = 0;
  viskores::Float64 global_z_max = 0;

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

viskores::cont::ArrayHandle<viskores::Range>
DataSet::GetRange(const std::string &field_name) const
{
  const size_t num_domains = m_domains.size();

  viskores::cont::ArrayHandle<viskores::Range> range;
  viskores::Id num_components = 0;

  for(size_t i = 0; i < num_domains; ++i)
  {
    if(!m_domains[i].HasField(field_name))
    {
      continue;
    }

    const viskores::cont::Field &field = m_domains[i].GetField(field_name);
    viskores::cont::ArrayHandle<viskores::Range> sub_range;
    sub_range = field.GetRange();

    viskores::Id components = sub_range.ReadPortal().GetNumberOfValues();

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

    for(viskores::Id c = 0; c < components; ++c)
    {
      viskores::Range s_range = sub_range.ReadPortal().Get(c);
      viskores::Range c_range = range.ReadPortal().Get(c);
      c_range.Include(s_range);
      range.WritePortal().Set(c, c_range);
    }
  }
  return range;
}

viskores::cont::ArrayHandle<viskores::Range>
DataSet::GetGlobalRange(const std::string &field_name) const
{
  VTKH_DATA_OPEN("GetGlobalRange");
  viskores::cont::ArrayHandle<viskores::Range> range;
  range = GetRange(field_name);

#ifdef VTKH_PARALLEL
  viskores::Id num_components = range.GetNumberOfValues();
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

      viskores::Range c_range = range.ReadPortal().Get(i);

      viskores::Float64 local_min;
      viskores::Float64 local_max;

      if(num_components != 0)
      {
        local_min = c_range.Min;
        local_max = c_range.Max;
      }
      else
      {
        local_min = std::numeric_limits<viskores::Float64>::max();
        local_max = std::numeric_limits<viskores::Float64>::lowest();
      }

      viskores::Float64 global_min = 0;
      viskores::Float64 global_max = 0;

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
    const viskores::cont::DataSet &dom = m_domains[i];
    viskores::UInt8 shape_type;
    bool single_type = VISKORESDataSetInfo::IsSingleCellShape(dom.GetCellSet(), shape_type);

    if(dom.GetCellSet().GetNumberOfCells() > 0)
    {
      is_points = (single_type && (shape_type == 1)) && is_points;
    }
  }

  is_points = detail::GlobalAgreement(is_points);
  return is_points;
}

bool
DataSet::IsLineMesh() const
{
  const bool is_empty = GlobalIsEmpty();
  if(is_empty) return false;

  // since we are not empty, start with the affirmative is_lines.
  // if someone is not lines, the we will figure it out here
  bool is_lines = true;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    const viskores::cont::DataSet &dom = m_domains[i];
    viskores::UInt8 shape_type;
    bool single_type = VISKORESDataSetInfo::IsSingleCellShape(dom.GetCellSet(), shape_type);

    if(dom.GetCellSet().GetNumberOfCells() > 0)
    {
      is_lines = (single_type && (shape_type == 3)) && is_lines;
    }
  }

  is_lines = detail::GlobalAgreement(is_lines);
  return is_lines;
}

bool
DataSet::IsUnstructured() const
{
  bool is_unstructured = true;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    const viskores::cont::DataSet &dom = m_domains[i];
    int dims;
    is_unstructured = !VISKORESDataSetInfo::IsStructured(dom, dims) && is_unstructured;

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
    const viskores::cont::DataSet &dom = m_domains[i];
    int dims;
    is_structured = VISKORESDataSetInfo::IsStructured(dom, dims) && is_structured;

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
DataSet::SetCycle(const viskores::UInt64 cycle)
{
  m_cycle = cycle;
}

viskores::UInt64
DataSet::GetCycle() const
{
  return m_cycle;
}

void
DataSet::SetTime(const double time)
{
  m_time = time;
}

double
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

viskores::cont::DataSet&
DataSet::GetDomainById(const viskores::Id domain_id)
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

bool DataSet::HasDomainId(const viskores::Id &domain_id) const
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    if(m_domain_ids[i] == domain_id) return true;
  }

  return false;
}

void
DataSet::AddConstantCellField(const viskores::Float32 value, const std::string &fieldname)
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    viskores::Id num_cells = m_domains[i].GetNumberOfCells();
    viskores::cont::ArrayHandle<viskores::Float32> array;
    detail::MemSet(array, value, num_cells);
    viskores::cont::Field field(fieldname, viskores::cont::Field::Association::Cells, array);
    m_domains[i].AddField(field);
  }
}

void
DataSet::AddConstantPointField(const viskores::Float32 value, const std::string &fieldname)
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    viskores::Id num_points = m_domains[i].GetCoordinateSystem().GetData().GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Float32> array;
    detail::MemSet(array, value, num_points);
    viskores::cont::Field field(fieldname, viskores::cont::Field::Association::Points, array);
    m_domains[i].AddField(field);
  }
}

void
DataSet::AddLinearPointField(const viskores::Float32 value, const std::string &fieldname)
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    viskores::Id num_points = m_domains[i].GetCoordinateSystem().GetData().GetNumberOfValues();
    viskores::cont::ArrayHandle<viskores::Float32> array;
    detail::MemSet(array, value, num_points);
    for(int j = 0; j < num_points; ++j)
      array.WritePortal().Set(j,j);
    viskores::cont::Field field(fieldname, viskores::cont::Field::Association::Points, array);
    m_domains[i].AddField(field);
  }
}

void
DataSet::AddDomainIdField(const std::string &fieldname)
{
  const size_t size = m_domain_ids.size();

  for(size_t i = 0; i < size; ++i)
  {
    viskores::Id domain_id = m_domain_ids[i];
    viskores::Id num_cells = m_domains[i].GetNumberOfCells();
    viskores::cont::ArrayHandle<viskores::Float32> array;
    detail::MemSet(array, (viskores::Float32)domain_id, num_cells);
    viskores::cont::Field field(fieldname, viskores::cont::Field::Association::Cells, array);
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

void
DataSet::RemoveField(const std::string &field_name)
{

  const size_t ndomains = m_domains.size();
  for(size_t i = 0; i < ndomains; ++i)
  {
    if(m_domains[i].HasField(field_name))
    {
        // to remove, one must first clone
        viskores::cont::DataSet domain_new;
        domain_new.CopyStructure(m_domains[i]);

        // loop over fields and all add except for the
        // one we want to remove
        viskores::IdComponent nfields = m_domains[i].GetNumberOfFields();
                for(viskores::IdComponent f_idx = 0; f_idx < nfields; f_idx++)
        {
            viskores::cont::Field &field = m_domains[i].GetField(f_idx);
            if(field.GetName() != field_name)
            {
                domain_new.AddField(field);
            }
        }

        m_domains[i] = domain_new;
    }
  }
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

viskores::cont::Field::Association
DataSet::GetFieldAssociation(const std::string &field_name, bool &valid_field) const
{
  valid_field = true;
  if(!this->GlobalFieldExists(field_name))
  {
    valid_field = false;
    return viskores::cont::Field::Association::Any;
  }

  int assoc_id = -1;
  if(this->FieldExists(field_name))
  {
    const size_t num_domains = m_domains.size();
    viskores::Bounds bounds;

    viskores::cont::Field::Association local_assoc;
    for(size_t i = 0; i < num_domains; ++i)
    {
      viskores::cont::DataSet dom = m_domains[i];
      if(dom.HasField(field_name))
      {
        local_assoc = dom.GetField(field_name).GetAssociation();
        if(local_assoc == viskores::cont::Field::Association::Any)
        {
          assoc_id = 0;
        }
        else if ( local_assoc == viskores::cont::Field::Association::WholeDataSet)
        {
          assoc_id = 1;
        }
        else if ( local_assoc == viskores::cont::Field::Association::Points)
        {
          assoc_id = 2;
        }
        else if ( local_assoc == viskores::cont::Field::Association::Cells)
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

  viskores::cont::Field::Association assoc;

  if(assoc_id == 0)
  {
    assoc = viskores::cont::Field::Association::Any;
  }
  else if ( assoc_id == 1)
  {
    assoc = viskores::cont::Field::Association::WholeDataSet;
  }
  else if ( assoc_id == 2)
  {
    assoc = viskores::cont::Field::Association::Points;
  }
  else if ( assoc_id == 3)
  {
    assoc = viskores::cont::Field::Association::Cells;
  }
  else
  {
    throw Error("Get association: unknown association");
  }
  return assoc;
}

viskores::Id
DataSet::GetFieldType(const std::string &field_name, bool &valid_field) const
{
  valid_field = true;
  if(!this->GlobalFieldExists(field_name))
  {
    valid_field = false;
    return -1;
  }

  using scalarI = viskores::cont::ArrayHandle<viskores::Int32>;
  using scalarF = viskores::cont::ArrayHandle<viskores::Float32>;
  using scalarD = viskores::cont::ArrayHandle<viskores::Float64>;
  using vec2F   = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32,2>>; 
  using vec2D   = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,2>>; 
  using vec3F   = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32,3>>; 
  using vec3D   = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,3>>; 

  int field_id = -1;
  if(this->FieldExists(field_name))
  {
    const size_t num_domains = m_domains.size();
    viskores::Bounds bounds;

    viskores::cont::Field::Association local_assoc;
    for(size_t i = 0; i < num_domains; ++i)
    {
      viskores::cont::DataSet dom = m_domains[i];
      if(dom.HasField(field_name))
      {
	viskores::cont::Field local_field = dom.GetField(field_name);
        if(local_field.GetData().IsType<scalarI>())
        {
          field_id = 0;
        }
	else if(local_field.GetData().IsType<scalarF>())
        {
          field_id = 1;
        }
	else if(local_field.GetData().IsType<scalarD>())
        {
          field_id = 2;
        }
	else if(local_field.GetData().IsType<vec2F>())
        {
          field_id = 3;
        }
	else if(local_field.GetData().IsType<vec2D>())
        {
          field_id = 4;
        }
	else if(local_field.GetData().IsType<vec3F>())
        {
          field_id = 5;
        }
	else if(local_field.GetData().IsType<vec3D>())
        {
          field_id = 6;
        }
        break;
      }
    }
  }

#ifdef VTKH_PARALLEL

  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());


  int *global_field_ids = new int[vtkh::GetMPISize()];

  MPI_Allgather(&field_id,
                1,
                MPI_INT,
                global_field_ids,
                1,
                MPI_INT,
                mpi_comm);

  int id = -1;

  for(int i = 0; i < vtkh::GetMPISize(); ++i)
  {
    if(global_field_ids[i] != -1)
    {
      if(id != -1 && global_field_ids[i] != id)
      {
        std::stringstream msg;
        msg<<"field "<< field_name
           <<" has inconsistent types";;
        throw Error(msg.str());
      }
      else
      {
        id = global_field_ids[i];
      }
    }
  }
  field_id = id;
  delete[] global_field_ids;
#endif

  return field_id;
}

viskores::Id DataSet::NumberOfComponents(const std::string &field_name) const
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
