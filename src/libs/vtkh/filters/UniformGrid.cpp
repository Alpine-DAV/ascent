
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
      std::cerr << "inputing field: " << std::endl;
      field.PrintSummary(std::cerr);
      std::cerr << "inputing field END: " << std::endl;
      ReduceField r_field(field, m_dataset, m_invalid_value);
      vtkm::cont::Field res_field = r_field.reduce();
      //auto reduce_field = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
      //reduce_field.CastAndCall(r_field);
      std::cerr << "resulting field: " << std::endl;
      res_field.PrintSummary(std::cerr);
      std::cerr << "resulting field END: " << std::endl;
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

      vtkm::cont::UnknownArrayHandle uah_field = m_input_field.GetData();

      using scalar32 = vtkm::cont::ArrayHandle<vtkm::Float32>;
      using scalar64 = vtkm::cont::ArrayHandle<vtkm::Float64>;
      using vec32    = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>; 
      using vec64    = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>; 
      using vecSOA32 = vtkm::cont::ArrayHandleSOA<vtkm::Vec<vtkm::Float32,3>>; 
      using vecSOA64 = vtkm::cont::ArrayHandleSOA<vtkm::Vec<vtkm::Float64,3>>; 

      vtkm::cont::ArrayHandle<vtkm::Float32> ah_mask;
      m_data_set.GetField("mask").GetData().AsArrayHandle(ah_mask);
      auto mask_portal = ah_mask.ReadPortal();
      int num_points = mask_portal.GetNumberOfValues();

      std::cerr << "HEREERERER" << std::endl;
      std::cerr << uah_field.GetValueTypeName() << std::endl;
      m_input_field.PrintSummary(std::cerr);
      //create invalid mask where true == invalid
      std::vector<float> l_mask(num_points,1);
      std::vector<float> g_mask(num_points,1);
      for(int j = 0; j < num_points; ++j)
      {
        l_mask[j] = l_mask[j] && mask_portal.Get(j);
      }
      
      MPI_Reduce(l_mask.data(), g_mask.data(), num_points, MPI_FLOAT, MPI_LAND, 0, MPI_COMM_WORLD);
      std::cerr << "got to this" << std::endl;

      ////send to root process
      if(uah_field.CanConvert<scalar32>())
      {
        //loop through field, zero out invalid value
	scalar32 ah_field = m_input_field.GetData().AsArrayHandle<scalar32>();
        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
            ah_field.WritePortal().Set(i,0);
        }
	std::cerr << "FLOAT FLOAT FLOAT " << std::endl;
        float * local_field = GetVTKMPointer(ah_field);
        std::vector<float> global_field(num_points,0);
        MPI_Reduce(local_field, global_field.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	std::cerr << "global data: " << std::endl;
	for(int i = 0; i < num_points; ++i)
	{
		std::cerr << global_field[i] << " ";
	}
	std::cerr << std::endl;
	std::cerr << "global data END" << std::endl;
	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	  }
	  scalar32 ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end scalar32
      else if(uah_field.CanConvert<scalar64>())
      {
	scalar64 ah_field = uah_field.AsArrayHandle<scalar64>();
        //loop through field, zero out invalid value
        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1.0)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
        }
        double * local_field = GetVTKMPointer(ah_field);
        std::vector<double> global_field(num_points,0);
        MPI_Reduce(local_field, global_field.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	  }
	  scalar64 ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      } //end scalar64
      if(uah_field.CanConvert<vec32>())
      {
        //loop through field, zero out invalid value
	vec32 ah_field = m_input_field.GetData().AsArrayHandle<vec32>();
        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i*3,0);
            ah_field.WritePortal().Set(i*3+1,0);
            ah_field.WritePortal().Set(i*3+2,0);
	  }
        }
	std::cerr << "FLOAT FLOAT FLOAT " << std::endl;
        float * local_field = GetVTKMPointer(ah_field);
	int vec_points = num_points*3;
        std::vector<float> global_field(vec_points,0);
        MPI_Reduce(local_field, global_field.data(), vec_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	std::cerr << "global data: " << std::endl;
	for(int i = 0; i < vec_points; ++i)
	{
		std::cerr << global_field[i][0] << " ";
	}
	std::cerr << std::endl;
	std::cerr << "global data END" << std::endl;
	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i*3] = m_invalid_value;
              global_field[i*3+1] = m_invalid_value;
              global_field[i*3+2] = m_invalid_value;
	    }
	  }
	  vec32 ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),vec_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec32
      else
        return m_input_field;

      return res;
  
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
    std::cerr << "AFTER Reduce();" << std::endl;
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
