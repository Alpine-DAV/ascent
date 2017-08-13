#include <vtkh_error.hpp>
#include <vtkh_data_set.hpp>
#include <utils/vtkm_dataset_info.hpp>
// std includes
#include <sstream>
//vtkm includes
#include <vtkm/cont/Error.h>
#ifdef PARALLEL
  #include <mpi.h>
#endif
namespace vtkh {

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

vtkm::cont::DataSet
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
DataSet::GetGlobalNumberOfDomains() const
{
  vtkm::Id domains = this->GetNumberOfDomains(); 
#ifdef PARALLEL 
  MPI_Comm mpi_comm = vtkh::GetMPIComm();
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
  vtkm::Bounds bounds;
  bounds = GetBounds(coordinate_system_index);

#ifdef PARALLEL
  MPI_Comm mpi_comm = vtkh::GetMPIComm();

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
  return bounds;
}

vtkm::cont::ArrayHandle<vtkm::Range> 
DataSet::GetGlobalRange(const vtkm::Id index) const
{
  assert(m_domains.size() > 0); 
  vtkm::cont::Field field = m_domains.at(0).GetField(index);
  std::string field_name = field.GetName();
  return this->GetGlobalRange(field_name);
}

vtkm::cont::ArrayHandle<vtkm::Range> 
DataSet::GetGlobalRange(const std::string &field_name) const
{
  bool valid_field = true;
  const size_t num_domains = m_domains.size();

  vtkm::cont::ArrayHandle<vtkm::Range> range;
  vtkm::Id num_components = 0;

  for(size_t i = 0; i < num_domains; ++i)
  {
    if(!m_domains[0].HasField(field_name))
    {
      valid_field = false;
      break;
    }

    const vtkm::cont::Field &field = m_domains[i].GetField(field_name);
    vtkm::cont::ArrayHandle<vtkm::Range> sub_range;
    sub_range = field.GetRange();
     
    if(i == 0)
    {
      num_components = sub_range.GetPortalConstControl().GetNumberOfValues();    
      range = sub_range;
      continue;
    }

    vtkm::Id components = sub_range.GetPortalConstControl().GetNumberOfValues();    

    if(components != num_components)
    {
      std::stringstream msg;
      msg<<"GetRange call failed. The number of components ("<<components<<") in field "
         <<field_name<<" from domain "<<i<<" does not match the number of components "
         <<"("<<num_components<<") in domain 0";
      throw Error(msg.str());
    }

    for(vtkm::Id c = 0; c < components; ++c)
    {
      vtkm::Range s_range = sub_range.GetPortalControl().Get(c);
      vtkm::Range c_range = range.GetPortalControl().Get(c);
      c_range.Include(s_range);
      range.GetPortalControl().Set(c, c_range);
    }
   

  }

  if(!valid_field)
  {
    std::string msg = "GetRange call failed. ";
    msg += " Field " +  field_name + " did not exist in at least one domain."; 
    throw Error(msg);
  }

#ifdef PARALLEL
  MPI_Comm mpi_comm = vtkh::GetMPIComm();
  //
  // it is possible to have an empty dataset at on of the ranks
  // so we must check for this and so MPI comm does not hang.
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
  int max_non_zero = 0;;
  //
  // find the largest component
  //
  for(int i = 0; i < vtkh::GetMPISize(); ++i)
  {
    if(global_components[i] != 0)
    {
      max_non_zero = std::max(global_components[i], max_non_zero);
    }
  }
  //
  // verify uniform component length
  //
  for(int i = 0; i < vtkh::GetMPISize(); ++i)
  {
    if(global_components[i] != 0)
    {
      if(max_non_zero != global_components[i])
      {
        std::stringstream msg;
        msg<<"GetRange call failed. The number of components ("
           <<global_components[i]<<") in field "
           <<field_name<<" from rank"<<i<<" does not match the number of components in"
           <<" the other ranks "<<max_non_zero;
        throw Error(msg.str());
      }
    }
  }
  //
  // if we do not have any components, then we need to init some
  // empty ranges to participate in comm
  //
  if(num_components == 0)
  {
    range.Allocate(max_non_zero);
    num_components = max_non_zero;
  }

  delete[] global_components;
  for(int i = 0; i < num_components; ++i)
  {
    vtkm::Range c_range = range.GetPortalControl().Get(i);
    vtkm::Float64 local_min = c_range.Min;
    vtkm::Float64 local_max = c_range.Max;
    
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
    range.GetPortalControl().Set(i, c_range);
  }
#endif

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
DataSet::IsStructured(int &topological_dims, const vtkm::Id cell_set_index) const
{
  topological_dims = -1;
  bool is_structured = false;
  const size_t num_domains = m_domains.size();
  for(size_t i = 0; i < num_domains; ++i)
  {
    const vtkm::cont::DataSet &dom = m_domains[i];
    int dims; 
    is_structured = VTKMDataSetInfo::IsStructured(dom, dims, cell_set_index);

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

#ifdef PARALLEL
  int local_boolean = is_structured ? 1 : 0; 
  int global_boolean;
  MPI_Comm mpi_comm = vtkh::GetMPIComm();
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean != vtkh::GetMPISize())
  {
    is_structured = false;
  }
  if(!is_structured)
  {
    topological_dims = -1;
  }
#endif
  return is_structured;
}
} // namspace vtkh
