
#include <vtkh/filters/UniformGrid.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/vtkm_filters/vtkmProbe.hpp>

#include <limits>

#ifdef VTKH_PARALLEL
#include <vtkh/utils/vtkm_array_utils.hpp>

#include <vtkm/thirdparty/diy/diy.h>
#include <vtkm/thirdparty/diy/mpi-cast.h>
#include <vtkm/cont/EnvironmentTracker.h>

#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

#ifdef VTKH_PARALLEL
class GlobalReduceFields
{
  vtkm::cont::DataSet &m_dataset;
  vtkm::Float64       m_invalid_value;

public:
  GlobalReduceFields(vtkm::cont::DataSet &dataset, vtkm::Float64 invalid_value)
    : m_dataset(dataset),
      m_invalid_value(invalid_value)
  {}
  ~GlobalReduceFields()
  {}

  vtkm::cont::DataSet Reduce()
  {
    vtkm::cont::DataSet res;
    int num_fields = m_dataset.GetNumberOfFields();
    std::cerr << "NUM fields: " << num_fields << std::endl;
    m_dataset.PrintSummary(std::cerr);
    for(int i = 0; i < num_fields; ++i)
    { 
      vtkm::cont::Field field = m_dataset.GetField(i);
      field.PrintSummary(std::cerr);
      ReduceField r_field(field, m_dataset, m_invalid_value);
      vtkm::cont::Field res_field = r_field.reduce();
      //auto reduce_field = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
      //reduce_field.CastAndCall(r_field);
      res.AddField(res_field);
    } 

    return res;
  }

  struct ReduceField
  {
    vtkm::cont::Field &m_input_field;
    vtkm::cont::DataSet &m_data_set;
    vtkm::Float64 &m_invalid_value;
  
    ReduceField(vtkm::cont::Field &input_field, vtkm::cont::DataSet &data_set, vtkm::Float64 &invalid_value)
      : m_input_field(input_field),
        m_data_set(data_set),
	m_invalid_value(invalid_value)
    {}
  
    vtkm::cont::Field 
    reduce()
    {
      vtkm::cont::Field res;
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
      int par_rank;
      int par_size;
      MPI_Comm_rank(mpi_comm, &par_rank);
      MPI_Comm_size(mpi_comm, &par_size);  
      //if parallel collect valid results on root rank
      vtkm::cont::ArrayHandle<vtkm::Float32> ah_mask;
      m_data_set.GetField("mask").GetData().AsArrayHandle(ah_mask);
      auto mask_portal = ah_mask.ReadPortal();
      int num_points = mask_portal.GetNumberOfValues();
      vtkm::cont::UnknownArrayHandle ah_field = m_input_field.GetData();
      using ah_d = vtkm::cont::ArrayHandle<vtkm::Float64>;
      using ah_f = vtkm::cont::ArrayHandle<vtkm::Float32>;
      std::cerr << "HEREERERER" << std::endl;
      std::cerr << ah_field.GetValueTypeName() << std::endl;
      m_input_field.PrintSummary(std::cerr);
      if(ah_field.CanConvert<vtkm::cont::ArrayHandle<vtkm::Float64>>())
	      std::cerr << "FLOAT FLOAT FLOAT " << std::endl;
      else if(ah_field.CanConvert<ah_d>())
	      std::cerr << "DOUBLE DOUBLE DOUBLE " << std::endl;
      else
        return m_input_field;
      std::cerr << "got to this" << std::endl;
      return m_input_field;

      //loop through field, zero out invalid value
      //for(int i = 0; i < num_points; ++i)
      //{
      //  if(mask_portal.Get(i) == 1)
      //    ah_field.WritePortal().Set(i,0);
      //}
      ////send to root process
      //vtkm::Float64 * local_field = GetVTKMPointer(ah_field);
      //std::vector<vtkm::Float64> global_field(num_points,0);
      ////MPI_Reduce(local_field, global_field.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      ////create invalid mask where true == invalid
      //std::vector<float> l_mask(num_points,1);
      //std::vector<float> g_mask(num_points,1);
      //for(int j = 0; j < num_points; ++j)
      //{
      //  l_mask[j] = l_mask[j] && mask_portal.Get(j);
      //}

      //std::cerr << "local mask: ";
      //for(int m : l_mask)
      //  std::cerr << m << " ";
      //std::cerr << std::endl;

      //MPI_Reduce(l_mask.data(), g_mask.data(), num_points, MPI_DOUBLE, MPI_LAND, 0, MPI_COMM_WORLD);

      //if(par_rank == 0)
      //{
      //  std::cerr << "global mask: ";
      //  for(int m : g_mask)
      //    std::cerr << m << " ";
      //  std::cerr << std::endl;
      //}
      return res;
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
#endif

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
#ifdef VTKH_PARALLEL
    vtkh::detail::GlobalReduceFields g_reducefields(dataset,m_invalid_value);
    auto output = g_reducefields.Reduce();
    if(par_rank == 0)
    {
      std::cerr << "output after reduce: " << std::endl;
      output.PrintSummary(std::cerr);
    }
    //auto full = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
    //full.CastAndCall(g_reducefields);
    this->m_output->AddDomain(output, domain_id);
#else
    this->m_output->AddDomain(dataset, domain_id);
#endif
  }

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
