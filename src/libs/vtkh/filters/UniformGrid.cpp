
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

namespace detail
{

class GlobalReduceFields
{
  vtkm::cont::DataSet m_dataset;

public:
  GlobalReduceFields(vtkm::cont::DataSet dataset)
    : m_dataset(dataset)
  {}
  ~GlobalReduceFields()
  {}

  vtkm::cont::DataSet Reduce()
  {
    vtkm::cont::DataSet res;
    int num_fields = m_dataset.GetNumberOfFields();
    for(int i = 0; i < num_fields; ++i)
    { 
      vtkm::cont::Field field = m_dataset.GetField(i);
      ReduceField r_field(res);
      auto reduce_field = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
      reduce_field.CastAndCall(r_field);
    } 

    return res;
  }

  struct ReduceField
  {
    vtkm::cont::DataSet &m_data_set;
  
    ReduceField(vtkm::cont::DataSet &data_set)
      : m_data_set(data_set)
    {}
  
    template<typename T, typename S>
    void operator()(const vtkm::cont::ArrayHandle<T,S> &vtkmNotUsed(field)) const
    {
      //check to see if this is a supported field ;
      //const vtkm::cont::Field &scalar_field = m_in_data_sets[0].GetField(m_field_index);
      //bool is_supported = (scalar_field.GetAssociation() == vtkm::cont::Field::Association::Points ||
      //                     scalar_field.GetAssociation() == vtkm::cont::Field::Association::Cells);
  
      //if(!is_supported) return;
  
      //bool assoc_points = scalar_field.GetAssociation() == vtkm::cont::Field::Association::Points;
      //vtkm::cont::ArrayHandle<T> out;
      //if(assoc_points)
      //{
      //  out.Allocate(m_num_points);
      //}
      //else
      //{
      //  out.Allocate(m_num_cells);
      //}
  
      //for(size_t i = 0; i < m_in_data_sets.size(); ++i)
      //{
      //  const vtkm::cont::Field &f = m_in_data_sets[i].GetField(m_field_index);
      //  vtkm::cont::ArrayHandle<T,S> in = f.GetData().AsArrayHandle<vtkm::cont::ArrayHandle<T,S>>();
      //  vtkm::Id start = 0;
      //  vtkm::Id copy_size = in.GetNumberOfValues();
      //  vtkm::Id offset = assoc_points ? m_point_offsets[i] : m_cell_offsets[i];
  
      //  vtkm::cont::Algorithm::CopySubRange(in, start, copy_size, out, offset);
      //}
  
      //vtkm::cont::Field out_field(scalar_field.GetName(),
      //                            scalar_field.GetAssociation(),
      //                            out);
      //m_data_set.AddField(out_field);
  
    }
  }; //struct reduceFields

};//class globalReduceFields

} //namespace detail

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
    //take uniform sampled grid and reduce to root process
    vtkh::detail::GlobalReduceFields g_reducefields(dataset);
    auto output = g_reducefields.Reduce();
    //auto full = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
    //full.CastAndCall(g_reducefields);

    this->m_output->AddDomain(output, domain_id);
  }

//if parallel collect valid results on root rank
#ifdef VTKH_PARALLEL

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

    int num_fields = dom.GetNumberOfFields();
    //loop through fields, zero out invalid value
    for(int j = 0; j < num_fields; ++j)
    {
      vtkm::cont::ArrayHandle<vtkm::Float64> ah_field;
      dom.GetField(j).GetData().AsArrayHandle(ah_field);
    }
    //send to root process
    
  }
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
