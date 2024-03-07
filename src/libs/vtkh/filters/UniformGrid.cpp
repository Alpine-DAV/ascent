
#include <vtkh/filters/UniformGrid.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/vtkm_filters/vtkmProbe.hpp>

#include <limits>

#ifdef VTKH_PARALLEL
#include <vtkm/thirdparty/diy/diy.h>
#include <vtkm/thirdparty/diy/mpi-cast.h>
#include <vtkm/cont/EnvironmentTracker.h>
#include <mpi.h>
#endif

namespace vtkh
{

UniformGrid::UniformGrid()
	: m_invalid_value(std::numeric_limits<double>::min())
{

}

UniformGrid::~UniformGrid()
{

}

void
UniformGrid::PreExecute()
{
  Filter::PreExecute();
}

void
UniformGrid::DoExecute()
{
#ifdef VTKH_PARALLEL
  // Setup VTK-h and VTK-m comm.
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
  int par_rank;
  int par_size;
  MPI_Comm_rank(mpi_comm, &par_rank);
  MPI_Comm_size(mpi_comm, &par_size);  
#endif
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();
  this->m_input->AddConstantPointField(0.0, "mask");

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkh::vtkmProbe probe;
    probe.dims(m_dims);
    probe.origin(m_origin);
    probe.spacing(m_spacing);
    probe.invalidValue(m_invalid_value);

    auto dataset = probe.Run(dom);

    this->m_output->AddDomain(dataset, domain_id);
  }

//if parallel collect valid results on root rank
#ifdef VTKH_PARALLEL

  int num_points = m_dims[0]*m_dims[1]*m_dims[2];
  std::cerr << "dims size: " << num_points << std::endl;
  std::cerr << "m_dims: "<< m_dims[0] << " " << m_dims[1] << " " << m_dims[2] << std::endl;
  //create invalid mask where true == invalid
  std::vector<float> l_mask(num_points,1);
  std::vector<float> g_mask(num_points,1);
  //loop over local domains
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_output->GetDomain(i, dom, domain_id);
    std::cerr << "domain: " << i << " START" << std::endl;
    dom.PrintSummary(std::cerr);
    std::cerr << "domain: " << i << " END" << std::endl;
    vtkm::cont::ArrayHandle<vtkm::Float32> ah_mask;
    dom.GetField("mask").GetData().AsArrayHandle(ah_mask);
    auto mask_portal = ah_mask.ReadPortal();
    for(int j = 0; j < num_points; ++j)
    {
      l_mask[j] = l_mask[j] && mask_portal.Get(j);
    }
  std::cerr << "mask: ";
  for(int m : l_mask)
    std::cerr << m << " ";
  std::cerr << std::endl;
  }
  std::cerr << "local mask: ";
  for(int m : l_mask)
    std::cerr << m << " ";
  std::cerr << std::endl;

   MPI_Reduce(l_mask.data(), g_mask.data(), num_points, MPI_FLOAT, MPI_LAND, 0, MPI_COMM_WORLD);

  if(par_rank == 0)
  {
    std::cerr << "global mask: ";
    for(int m : g_mask)
      std::cerr << m << " ";
    std::cerr << std::endl;
  }
#endif

}

void
UniformGrid::PostExecute()
{
  Filter::PostExecute();
}

std::string
UniformGrid::GetName() const
{
  return "vtkh::UniformGrid";
}

void
UniformGrid::Dims(const Vec3f dims)
{
  m_dims = dims;
}

void
UniformGrid::Origin(const Vec3f origin)
{
  m_origin = origin;
}

void
UniformGrid::Spacing(const Vec3f spacing)
{
  m_spacing = spacing;
}

void
UniformGrid::InvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

} // namespace vtkh
