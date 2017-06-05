#include <vtkh_error.hpp>
#include <vtkh_data_set.hpp>
// std includes
#include <sstream>
//vtkm includes
#include <vtkm/cont/Error.h>
#ifdef PARALLEL
  #include <mpi.h>
#endif
namespace vtkh {

void 
vtkhDataSet::AddDomain(vtkm::cont::DataSet data_set, int domain_id) 
{

  assert(m_domains.size() == m_domain_ids.size());
  m_domains.push_back(data_set);
  m_domain_ids.push_back(domain_id);
}

void 
vtkhDataSet::GetDomain(const int index, vtkm::cont::DataSet &data_set, int &domain_id) 
{
  const size_t num_domains = m_domains.size();

  if(index >= num_domains && index < 0)
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
vtkhDataSet::GetNumDomains() const
{
  return m_domains.size();
}

vtkm::Bounds 
vtkhDataSet::GetBounds(vtkm::Id index) const
{
  const size_t num_domains = m_domains.size();
  vtkm::Bounds bounds;
  for(size_t i = 0; i < num_domains; ++i)
  {
    vtkm::cont::CoordinateSystem coords;
    try
    {
      coords = m_domains[i].GetCoordinateSystem(index); 
    } 
    catch (const vtkm::cont::Error &error)
    {
      std::stringstream msg;
      msg<<"GetBounds call failed. vtk-m error was encountered while "
         <<"attempting to get coordinate system "<<index<<" from "
         <<"domaim "<<i<<". vtkm error message: "<<error.GetMessage();
      throw Error(msg.str());
    }

    vtkm::Bounds dom_bounds = coords.GetBounds();
  
    bounds.Include(dom_bounds);
  }

#ifdef PARALLEL
  MPI_Comm mpi_comm = VTKH::GetMPIComm();

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
vtkhDataSet::GetRange(const std::string &field_name) const
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
      range.Allocate(num_components);
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
  MPI_Comm mpi_comm = VTKH::GetMPIComm();
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

} // namspace vtkh
