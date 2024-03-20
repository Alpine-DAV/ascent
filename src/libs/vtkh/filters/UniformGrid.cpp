
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
    res.CopyStructure(m_dataset);
    int num_fields = m_dataset.GetNumberOfFields();
    for(int i = 0; i < num_fields; ++i)
    { 
      vtkm::cont::Field field = m_dataset.GetField(i);
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
      if(m_input_field.GetName() == "mask")
        return m_input_field;
      vtkm::cont::Field res;
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
      int par_rank;
      int par_size;
      MPI_Comm_rank(mpi_comm, &par_rank);
      MPI_Comm_size(mpi_comm, &par_size);  

      vtkm::cont::UnknownArrayHandle uah_field = m_input_field.GetData();

      using scalarI = vtkm::cont::ArrayHandle<vtkm::Int32>;
      using scalarF = vtkm::cont::ArrayHandle<vtkm::Float32>;
      using scalarD = vtkm::cont::ArrayHandle<vtkm::Float64>;
      using vec3_32  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>; 
      using vec3_64  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>; 
      using vec2_32  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>; 
      using vec2_64  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>; 

      vtkm::cont::ArrayHandle<vtkm::Float32> ah_mask;
      m_data_set.GetField("mask").GetData().AsArrayHandle(ah_mask);
      auto mask_portal = ah_mask.ReadPortal();
      int num_points = mask_portal.GetNumberOfValues();

      //create invalid mask where true == invalid
      std::vector<int> l_mask(num_points,1);
      std::vector<int> g_mask(num_points,1);
      std::vector<int> g_valid(num_points,0);
      std::vector<int> l_valid(num_points,0);
      for(int j = 0; j < num_points; ++j)
      {
        l_mask[j] = l_mask[j] && mask_portal.Get(j);
	if(l_mask[j] == 0)
	  l_valid[j] = 1;
      }
      
      MPI_Reduce(l_mask.data(), g_mask.data(), num_points, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
      MPI_Reduce(l_valid.data(), g_valid.data(), num_points, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

      ////send to root process
      if(uah_field.CanConvert<scalarI>())
      {
        //loop through field, zero out invalid value
	scalarI ah_field = m_input_field.GetData().AsArrayHandle<scalarI>();
        int * local_field = GetVTKMPointer(ah_field);
        std::vector<int> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
            ah_field.WritePortal().Set(i,0);
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_field[i] = global_field[i]/g_valid[i];
	    }
	  }
	  scalarI ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end scalarI
      else if(uah_field.CanConvert<scalarF>())
      {
        //loop through field, zero out invalid value
	scalarF ah_field = m_input_field.GetData().AsArrayHandle<scalarF>();
        float * local_field = GetVTKMPointer(ah_field);
        std::vector<float> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
            ah_field.WritePortal().Set(i,0);
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_field[i] = global_field[i]/g_valid[i];
	    }
	  }
	  scalarF ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end scalarF
      else if(uah_field.CanConvert<scalarD>())
      {
	scalarD ah_field = uah_field.AsArrayHandle<scalarD>();
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
	    if(g_valid[i] > 1)
	    {
              global_field[i] = global_field[i]/g_valid[i];
	    }
	  }
	  scalarD ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      } //end scalarD
      else if(uah_field.CanConvert<vec2_32>())
      {
        //loop through field, zero out invalid value
	vec2_32 ah_field = m_input_field.GetData().AsArrayHandle<vec2_32>();
        std::vector<float> local_x_points(num_points,0);
        std::vector<float> local_y_points(num_points,0);
        std::vector<float> global_x_points(num_points,0);
        std::vector<float> global_y_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
	  vec2_32 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
	    }
            vtkm::Vec<vtkm::Float32,2> points_vec = vtkm::make_Vec(global_x_points[i],global_y_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec2_32
      else if(uah_field.CanConvert<vec2_64>())
      {
        //loop through field, zero out invalid value
	vec2_64 ah_field = m_input_field.GetData().AsArrayHandle<vec2_64>();
        std::vector<double> local_x_points(num_points,0);
        std::vector<double> local_y_points(num_points,0);
        std::vector<double> global_x_points(num_points,0);
        std::vector<double> global_y_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
	  vec2_64 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
	    }
            vtkm::Vec<vtkm::Float64,2> points_vec = vtkm::make_Vec(global_x_points[i],global_y_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec2_64
      else if(uah_field.CanConvert<vec3_32>())
      {
        //loop through field, zero out invalid value
	vec3_32 ah_field = m_input_field.GetData().AsArrayHandle<vec3_32>();
        std::vector<float> local_x_points(num_points,0);
        std::vector<float> local_y_points(num_points,0);
        std::vector<float> local_z_points(num_points,0);
        std::vector<float> global_x_points(num_points,0);
        std::vector<float> global_y_points(num_points,0);
        std::vector<float> global_z_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
          local_z_points[i] = ah_field.ReadPortal().Get(i)[2];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
	  vec3_32 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
              global_z_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
              global_z_points[i] = global_z_points[i]/g_valid[i];
	    }

            vtkm::Vec<vtkm::Float32,3> points_vec = vtkm::make_Vec(global_x_points[i],
                                                                   global_y_points[i],
                                                                   global_z_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec3_32
      else if(uah_field.CanConvert<vec3_64>())
      {
        //loop through field, zero out invalid value
	vec3_64 ah_field = m_input_field.GetData().AsArrayHandle<vec3_64>();
        std::vector<double> local_x_points(num_points,0);
        std::vector<double> local_y_points(num_points,0);
        std::vector<double> local_z_points(num_points,0);
        std::vector<double> global_x_points(num_points,0);
        std::vector<double> global_y_points(num_points,0);
        std::vector<double> global_z_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
          local_z_points[i] = ah_field.ReadPortal().Get(i)[2];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
          vec3_64 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
              global_z_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
              global_z_points[i] = global_z_points[i]/g_valid[i];
	    }

            vtkm::Vec<vtkm::Float64,3> points_vec = vtkm::make_Vec(global_x_points[i],
                                                                   global_y_points[i],
                                                                   global_z_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec3_64
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
