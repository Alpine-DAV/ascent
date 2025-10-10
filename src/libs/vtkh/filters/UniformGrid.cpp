
#include <vtkh/filters/UniformGrid.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/vtkm_filters/vtkmProbe.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>

#include <limits>

#ifdef VTKH_PARALLEL
#include <vtkm/thirdparty/diy/diy.h>
#include <vtkm/thirdparty/diy/mpi-cast.h>
#include <vtkm/cont/EnvironmentTracker.h>
#include <vtkm/cont/DataSetBuilderUniform.h>

#include <mpi.h>
#endif

#include <vtkm/cont/Algorithm.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/cont/DataSetBuilderUniform.h>

using scalarI = vtkm::cont::ArrayHandle<vtkm::Int32>;
using scalarF = vtkm::cont::ArrayHandle<vtkm::Float32>;
using scalarD = vtkm::cont::ArrayHandle<vtkm::Float64>;
using vec3_32  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>;
using vec3_64  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>;
using vec2_32  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>;
using vec2_64  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>;
using Vec2d    = vtkm::Vec<double, 2>;
using Vec3d    = vtkm::Vec<double, 3>;

#define _DEBUG 0

namespace vtkh
{

namespace detail
{

vtkm::cont::Field 
MakeEmptyField(std::string field_name , vtkm::Id field_id, Vec3f dims, vtkm::cont::Field::Association assoc, vtkm::Float64 val)
{
  int num_values = 0;
  if(assoc == vtkm::cont::Field::Association::Cells) //cell centered field
  {
    int nx = (dims[0] > 1) ? (dims[0] - 1) : 1;
    int ny = (dims[1] > 1) ? (dims[1] - 1) : 1;
    int nz = (dims[2] > 1) ? (dims[2] - 1) : 1;

    num_values = nx * ny * nz;
  }
  else
  {
    int nx = (dims[0] > 0) ? (dims[0]) : 1;
    int ny = (dims[1] > 0) ? (dims[1]) : 1;
    int nz = (dims[2] > 0) ? (dims[2]) : 1;

    num_values = nx * ny * nz;
  }

  if(field_id == 0)
  {
    std::vector<int> v_empty(num_values, (int) val);
    scalarI ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 1)
  {
    std::vector<float> v_empty(num_values, (float) val);
    scalarF ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 2)
  {
    std::vector<double> v_empty(num_values, (double) val);
    scalarD ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 3)
  {
    vec2_32 ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
    // note Vec2f was declared as float64 in the vtkmProbe filter ...
    vtkm::Vec<vtkm::Float32,2> empty_vec = vtkm::make_Vec((float) val, (float) val);
    for(int i = 0; i < num_values; ++i)
    {

      ah_empty.WritePortal().Set(i,empty_vec);
    }
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 4)
  {
    vec2_64 ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>();
    vtkm::Vec<vtkm::Float64,2> empty_vec = vtkm::make_Vec(val, val);
    for(int i = 0; i < num_values; ++i)
    {
      ah_empty.WritePortal().Set(i,empty_vec);
    }
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 5)
  {
    vec3_32 ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>();
    vtkm::Vec<vtkm::Float32,3> empty_vec = vtkm::make_Vec((float) val, (float) val, (float) val);
    for(int i = 0; i < num_values; ++i)
    {
      ah_empty.WritePortal().Set(i,empty_vec);
    }
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 6)
  {
    vec3_64 ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>();
    for(int i = 0; i < num_values; ++i)
    {
      Vec3d empty_vec = vtkm::make_Vec(val, val, val);
      ah_empty.WritePortal().Set(i,empty_vec);
    }
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  vtkm::cont::Field field;
  return field;
}

#ifdef VTKH_PARALLEL
class GlobalReduceField
{
  const vtkm::cont::DataSet &m_dataset;
  const std::string         m_field;
  const vtkm::Float64       m_invalid_value;
  const Vec3f               &m_dims;
  const vtkm::Id            m_field_id;
  vtkm::cont::DataSet       &m_result;

public:
  GlobalReduceField(const vtkm::cont::DataSet &dataset, 
                    const std::string &field, 
                    const vtkm::Float64 &invalid_value, 
                    const Vec3f &dims,
                    const vtkm::Id &field_id,
                    vtkm::cont::DataSet &result)
    : m_dataset(dataset),
      m_result(result),
      m_field(field),
      m_dims(dims),
      m_field_id(field_id),
      m_invalid_value(invalid_value)
  {}
  ~GlobalReduceField()
  {}

  void Reduce()
  {
    m_result.CopyStructure(m_dataset);
    ReduceField r_field(m_field, m_dataset, m_invalid_value, m_dims, m_field_id, m_result);
    r_field.reduce();

    return;
  }

  struct ReduceField
  {
    const std::string &m_field_name;
    const vtkm::cont::DataSet &m_data_set;
    const vtkm::Float64 &m_invalid_value;
    const Vec3f &m_dims;
    const vtkm::Id &m_field_id;
    vtkm::cont::DataSet &m_result;
  
    ReduceField(const std::string &field_name,
                const vtkm::cont::DataSet &data_set, 
                const vtkm::Float64 &invalid_value,
                const Vec3f &dims, 
                const vtkm::Id &field_id, 
                vtkm::cont::DataSet &result)
      : m_field_name(field_name),
        m_data_set(data_set),
        m_dims(dims),
        m_field_id(field_id),
        m_result(result),
        m_invalid_value(invalid_value)
    {}

    void reduce()
    {

      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
      int par_rank;
      int par_size;
      MPI_Comm_rank(mpi_comm, &par_rank);
      MPI_Comm_size(mpi_comm, &par_size);
      std::cerr << "par_rank is here: " << par_rank << std::endl;

      vtkm::cont::Field res;
      vtkm::cont::Field field;

      bool is_empty = false;
      int num_points = m_dims[0]*m_dims[1];
      if(m_dims[2] != 0)
        num_points *= m_dims[2];

      if(m_data_set.HasField(m_field_name))
      {
        is_empty = false;
        field = m_data_set.GetField(m_field_name);
      }
      else
      {
        is_empty = true;
        field = detail::MakeEmptyField(m_field_name,  m_field_id, m_dims, vtkm::cont::Field::Association::Points, m_invalid_value); 
      }

      if(m_field_name == "HIDDEN")
      {
        //TODO: rename this field as "valid_mask" (? "invalid_mask" ?)
        //TODO: Figure out how HIDDEN works again so we can name it
        m_result.AddField(field);
        return;
      }

      vtkm::cont::UnknownArrayHandle uah_field;
      vtkm::cont::ArrayHandle<unsigned char> ah_mask;
      if (!is_empty)
      {
        std::cerr << " par_rank : " << par_rank << " is non emtpy: " << !is_empty << std::endl;
        uah_field = field.GetData();
        m_data_set.GetPointField("HIDDEN").GetData().AsArrayHandle(ah_mask);
      }
      else
      {
        std::cerr <<" uah allocate: " << std::endl;
        std::cerr <<"  ah MASK allocate: " << std::endl;
        ah_mask.AllocateAndFill(num_points, 2);
      }
      auto mask_portal = ah_mask.ReadPortal();
      std::cerr << " par_rank : " << par_rank << " after the allocate " << std::endl;

#if _DEBUG 
      std::cerr << "NUM_POINTS: " << num_points << std::endl;
#endif
      //Todo: NUM POINTS needs to be based on dims
      //Todo: determine if field point or cell
      //Todo: check if all ranks have field? 

      //local and global point ownership by rank
      std::vector<int> l_rank_mask(num_points,-1);
      std::vector<int> g_rank_mask(num_points,-1);

      //if a valid/owned point, declare your rank
      for(int j = 0; j < num_points; ++j)
      {

        std::cerr << "par rank says has num field vasl: " << par_rank << std::endl;
        if(mask_portal.Get(j) == 0)
	      {
          std::cerr << "par rank says it is valid: " << par_rank << std::endl;
          l_rank_mask[j] = par_rank;
	      }
      }

      //take Max to figure out which ranks own which points
      MPI_Allreduce(l_rank_mask.data(), g_rank_mask.data(), num_points, MPI_INT, MPI_MAX, mpi_comm);

      //combine fields
      ////send to root process
      std::cerr << "rank " << par_rank << " HEREEEEEEEEEEEEEEEEEEEEE"  << std::endl;
      if(m_field_id == 0)
      {
#if _DEBUG 
        std::cerr << "In scalar int global reduce for field: " << field.GetName() << std::endl;
#endif
        //loop through field, zero out invalid and unowned values
        scalarI ah_field = field.GetData().AsArrayHandle<scalarI>();
        int *local_field = GetVTKMPointer(ah_field);
        std::vector<int> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_INT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = (int) m_invalid_value;
            }
          }
          
          scalarI ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
          vtkm::cont::Field out_field(m_field_name,
                                      vtkm::cont::Field::Association::Points,
                                      ah_out);
          res = out_field;
        }
        else
        {
          res = field;
        }
      }//end scalarI
      else if(m_field_id == 1)
      {
#if _DEBUG 
        std::cerr << "In scalar float global reduce for field: " << field.GetName() << std::endl;
#endif
        //loop through field, zero out invalid value
        scalarF ah_field = field.GetData().AsArrayHandle<scalarF>();
        float * local_field = GetVTKMPointer(ah_field);
        std::vector<float> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,(float)0.0);
          }
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = (float) m_invalid_value;
            }
          }
          scalarF ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
          vtkm::cont::Field out_field(m_field_name,
                                      vtkm::cont::Field::Association::Points,
                                      ah_out);

          res = out_field;
        }
        else
        {
          res = field;
        }
      }//end scalarF
      else if(m_field_id == 2)
      {
#if _DEBUG 
        std::cerr << "In scalar double global reduce for field: " << field.GetName() << std::endl;
#endif
      std::cerr << "rank " << par_rank << " HEREEEEEEEEEEEEEEEEEEEEE 222222"  << std::endl;
            std::cerr << "is it this? " << std::endl;
        scalarD ah_field; 
        if(is_empty)
          ah_field.AllocateAndFill(num_points, 0.0);
        else
          ah_field = field.GetData().AsArrayHandle<scalarD>();
        //loop through field, zero out invalid value
        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
      std::cerr << "rank " << par_rank << " HEREEEEEEEEEEEEEEEEEEEEE 999999999999999"  << std::endl;
            ah_field.WritePortal().Set(i,(double)0.0);
          }
        }
        double * local_field = GetVTKMPointer(ah_field);
        std::vector<double> global_field(num_points,0.0);
        MPI_Reduce(local_field, global_field.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

          auto assoc = field.GetAssociation();
          const char* assoc_str = "Unknown";
          switch (assoc) {
            case vtkm::cont::Field::Association::Points:       assoc_str = "Points"; break;
            case vtkm::cont::Field::Association::Cells:        assoc_str = "Cells"; break;
            case vtkm::cont::Field::Association::WholeDataSet: assoc_str = "WholeDataSet"; break;
            case vtkm::cont::Field::Association::Any:         assoc_str = "Any"; break; // if present in your VTK-m
          }
          std::cerr << "rank: " << par_rank << " field assoc: " << assoc_str << std::endl;
        if(par_rank == 0)
        {
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = (double)m_invalid_value;
            }
          }
          
          scalarD ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);

          vtkm::cont::Field out_field(m_field_name,
                                      vtkm::cont::Field::Association::Points,
                                      ah_out);
          res = out_field;
        }
        else
        {
          res = field;
        }
      } //end scalarD
      else if(m_field_id == 3)
      {
        //loop through field, zero out invalid value
        vec2_32 ah_field = field.GetData().AsArrayHandle<vec2_32>();
        std::vector<float> local_x_points(num_points,(float)0.0);
        std::vector<float> local_y_points(num_points,(float)0.0);
        std::vector<float> global_x_points(num_points,(float)0.0);
        std::vector<float> global_y_points(num_points,(float)0.0);

	      //std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
            vec2_32 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
            ah_out.Allocate(num_points);
            for(int i = 0; i < num_points; ++i)
            {
              if(g_rank_mask[i] == 1)
              {
                global_x_points[i] = (float)m_invalid_value;
                global_y_points[i] = (float)m_invalid_value;
              }

              vtkm::Vec<vtkm::Float32,2> points_vec = vtkm::make_Vec(global_x_points[i],global_y_points[i]);
              ah_out.WritePortal().Set(i,points_vec);
            }

            vtkm::cont::Field out_field(m_field_name,
                                        vtkm::cont::Field::Association::Points,
                                        ah_out);
            res = out_field;
        }
        else
        {
          res = field;
        }
      }//end vec2_32
      else if(m_field_id == 4)
      {
        //loop through field, zero out invalid value
        vec2_64 ah_field = field.GetData().AsArrayHandle<vec2_64>();
        std::vector<double> local_x_points(num_points,(double)0.0);
        std::vector<double> local_y_points(num_points,(double)0.0);
        std::vector<double> global_x_points(num_points,(double)0.0);
        std::vector<double> global_y_points(num_points,(double)0.0);

	      //std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          vec2_64 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == 1)
            {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
            }
            vtkm::Vec<vtkm::Float64,2> points_vec = vtkm::make_Vec(global_x_points[i],global_y_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
          }
          vtkm::cont::Field out_field(m_field_name,
                                      vtkm::cont::Field::Association::Points,
                                      ah_out);
			      
          res = out_field;
        }
        else
          res = field;
      }//end vec2_64
      else if(m_field_id == 5)
      {
        //loop through field, zero out invalid value
        vec3_32 ah_field = field.GetData().AsArrayHandle<vec3_32>();
        std::vector<float> local_x_points(num_points,0);
        std::vector<float> local_y_points(num_points,0);
        std::vector<float> local_z_points(num_points,0);
        std::vector<float> global_x_points(num_points,0);
        std::vector<float> global_y_points(num_points,0);
        std::vector<float> global_z_points(num_points,0);
	      //std::cerr << ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";
        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
          local_z_points[i] = ah_field.ReadPortal().Get(i)[2];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          vec3_32 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == 1)
            {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
              global_z_points[i] = m_invalid_value;
            }

            vtkm::Vec<vtkm::Float32,3> points_vec = vtkm::make_Vec(global_x_points[i],
                                                                   global_y_points[i],
                                                                   global_z_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
          }
        
          vtkm::cont::Field out_field(m_field_name,
                                      vtkm::cont::Field::Association::Points,
                                      ah_out);

          res = out_field;
        }
        else
        {
          res = field;
        }
      }//end vec3_32
      else if(m_field_id == 6)
      {
        //loop through field, zero out invalid value
        vec3_64 ah_field = field.GetData().AsArrayHandle<vec3_64>();
        std::vector<double> local_x_points(num_points,0);
        std::vector<double> local_y_points(num_points,0);
        std::vector<double> local_z_points(num_points,0);
        std::vector<double> global_x_points(num_points,0);
        std::vector<double> global_y_points(num_points,0);
        std::vector<double> global_z_points(num_points,0);

      	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
          local_z_points[i] = ah_field.ReadPortal().Get(i)[2];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          vec3_64 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == 1)
            {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
              global_z_points[i] = m_invalid_value;
            }
            
            vtkm::Vec<vtkm::Float64,3> points_vec = vtkm::make_Vec(global_x_points[i],
                                                                   global_y_points[i],
                                                                   global_z_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
          }
          vtkm::cont::Field out_field(m_field_name,
                                      vtkm::cont::Field::Association::Points,
                                      ah_out);

          res = out_field;
        }
        else
        {
          res = field;
        }
      }//end vec3_64
      else
      {
      std::cerr << "rank " << par_rank << " HEREEEEEEEEEEEEEEEEEEEEE 3333333333"  << std::endl;
          m_result.AddField(field);
          return;
      }

      m_result.AddField(res);
      return;
    }
  }; //struct reduceFields

};//class globalReduceField
#endif

class LocalReduceField
{
  vtkm::cont::DataSet &m_dataset;
  vtkm::cont::Field   &m_field;
  vtkm::cont::Field   &m_mask;
  const std::string   m_field_name;
  vtkm::Float64       m_invalid_value;

public:
  LocalReduceField(vtkm::cont::DataSet &dataset, vtkm::cont::Field &field, vtkm::cont::Field &mask, const std::string &field_name, vtkm::Float64 invalid_value)
    : m_dataset(dataset),
      m_field(field),
      m_mask(mask),
      m_field_name(field_name),
      m_invalid_value(invalid_value)
  {}

  ~LocalReduceField()
  {}

  void LocalReduce()
  {
    vtkm::cont::UnknownArrayHandle uah_field = m_field.GetData();
    vtkm::cont::UnknownArrayHandle uah_local_field = m_dataset.GetField(m_field_name).GetData();

    //mask where 0 is valid adn 2 is invalid
    //holds individual domain
    vtkm::cont::ArrayHandle<unsigned char> tmp_mask;
    m_mask.GetData().AsArrayHandle(tmp_mask);

    //mask where 0 is valid adn 2 is invalid
    //holds all domains combined
    vtkm::cont::ArrayHandle<unsigned char> local_mask;
    if(m_field.IsPointField())
    {
      m_dataset.GetPointField("HIDDEN").GetData().AsArrayHandle(local_mask);
    }
    else
    {
      m_dataset.GetCellField("HIDDEN").GetData().AsArrayHandle(local_mask);
    }

    auto tmp_mask_portal = tmp_mask.ReadPortal();
    auto r_local_mask_portal = local_mask.ReadPortal();
    auto w_local_mask_portal = local_mask.WritePortal();
    
    int num_points = tmp_mask_portal.GetNumberOfValues();

    if(uah_field.CanConvert<scalarI>())
    {
      //loop through field, zero out invalid values
      scalarI tmp_data = m_field.GetData().AsArrayHandle<scalarI>();
      scalarI local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<scalarI>();
      int *tmp_field = GetVTKMPointer(tmp_data);
      int *local_field = GetVTKMPointer(local_data);

      for(int i = 0; i < num_points; ++i)
      {
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          local_field[i] = tmp_field[i];
          local_data.WritePortal().Set(i,tmp_field[i]);
          w_local_mask_portal.Set(i,0);
        }
      }
    }//end scalarI
    else if(uah_field.CanConvert<scalarF>())
    {
      //loop through field, zero out invalid values
      scalarF tmp_data = m_field.GetData().AsArrayHandle<scalarF>();
      scalarF local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<scalarF>();
      float *tmp_field = GetVTKMPointer(tmp_data);
      float *local_field = GetVTKMPointer(local_data);

      for(int i = 0; i < num_points; ++i)
      {
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          local_field[i] = tmp_field[i];
          local_data.WritePortal().Set(i,tmp_field[i]);
          w_local_mask_portal.Set(i,0);
        }
      }
    }//end scalarF
    else if(uah_field.CanConvert<scalarD>())
    {
      //loop through field, zero out invalid values
      scalarD tmp_data = m_field.GetData().AsArrayHandle<scalarD>();
      scalarD local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<scalarD>();
      double *tmp_field = GetVTKMPointer(tmp_data);
      double *local_field = GetVTKMPointer(local_data);

      for(int i = 0; i < num_points; ++i)
      {
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          local_field[i] = tmp_field[i];
          local_data.WritePortal().Set(i,tmp_field[i]);
          w_local_mask_portal.Set(i,0);
        }
      }
    } //end scalarD
    else if(uah_field.CanConvert<vec2_32>())
    {
      //loop through field, zero out invalid values
      vec2_32 tmp_data = m_field.GetData().AsArrayHandle<vec2_32>();
      vec2_32 local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<vec2_32>();

      for(int i = 0; i < num_points; ++i)
      {
        float tmp_x = tmp_data.ReadPortal().Get(i)[0];
        float tmp_y = tmp_data.ReadPortal().Get(i)[1];
        float local_x = local_data.ReadPortal().Get(i)[0];
        float local_y = local_data.ReadPortal().Get(i)[1];
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          w_local_mask_portal.Set(i,0);
          local_x = tmp_x;
          local_y = tmp_y;
          vtkm::Vec<vtkm::Float32,2> vec = vtkm::make_Vec(local_x,local_y);
          local_data.WritePortal().Set(i,vec);
        }
      }
    }//end vec2_32
    else if(uah_field.CanConvert<vec2_64>())
    {
      //loop through field, zero out invalid values
      vec2_64 tmp_data = m_field.GetData().AsArrayHandle<vec2_64>();
      vec2_64 local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<vec2_64>();

      for(int i = 0; i < num_points; ++i)
      {
        double tmp_x = tmp_data.ReadPortal().Get(i)[0];
        double tmp_y = tmp_data.ReadPortal().Get(i)[1];
        double local_x = local_data.ReadPortal().Get(i)[0];
        double local_y = local_data.ReadPortal().Get(i)[1];
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          w_local_mask_portal.Set(i,0);
          local_x = tmp_x;
          local_y = tmp_y;
          vtkm::Vec<vtkm::Float64,2> vec = vtkm::make_Vec(local_x,local_y);
          local_data.WritePortal().Set(i,vec);
        }
      }
    }//end vec2_64
    else if(uah_field.CanConvert<vec3_32>())
    {
      //loop through field, zero out invalid values
      vec3_32 tmp_data = m_field.GetData().AsArrayHandle<vec3_32>();
      vec3_32 local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<vec3_32>();

      for(int i = 0; i < num_points; ++i)
      {
        float tmp_x = tmp_data.ReadPortal().Get(i)[0];
        float tmp_y = tmp_data.ReadPortal().Get(i)[1];
        float tmp_z = tmp_data.ReadPortal().Get(i)[2];
        float local_x = local_data.ReadPortal().Get(i)[0];
        float local_y = local_data.ReadPortal().Get(i)[1];
        float local_z = local_data.ReadPortal().Get(i)[2];
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          w_local_mask_portal.Set(i,0);
          local_x = tmp_x;
          local_y = tmp_y;
          local_z = tmp_z;
          vtkm::Vec<vtkm::Float32,3> vec = vtkm::make_Vec(local_x,local_y,local_z);
          local_data.WritePortal().Set(i,vec);
        }
      }
    }//end vec3_32
    else if(uah_field.CanConvert<vec3_64>())
    {
      //loop through field, zero out invalid values
      vec3_64 tmp_data = m_field.GetData().AsArrayHandle<vec3_64>();
      vec3_64 local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<vec3_64>();

      for(int i = 0; i < num_points; ++i)
      {
        double tmp_x = tmp_data.ReadPortal().Get(i)[0];
        double tmp_y = tmp_data.ReadPortal().Get(i)[1];
        double tmp_z = tmp_data.ReadPortal().Get(i)[2];
        double local_x = local_data.ReadPortal().Get(i)[0];
        double local_y = local_data.ReadPortal().Get(i)[1];
        double local_z = local_data.ReadPortal().Get(i)[2];
        //tie breaker will be higher domain number 
	      //which we loop through as we VTKmProbe/sample
        if((tmp_mask_portal.Get(i) == 0)) //incoming domain
        {
          w_local_mask_portal.Set(i,0);
          local_x = tmp_x;
          local_y = tmp_y;
          local_z = tmp_z;
          vtkm::Vec<vtkm::Float64,3> vec = vtkm::make_Vec(local_x,local_y,local_z);
          local_data.WritePortal().Set(i,vec);
        }
      }
    }//end vec3_64
    else
    {
        return;
    }
  }; //struct reduceField
};//class localReduceField

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


#ifdef VTKH_PARALLEL
#if _DEBUG 
  std::cerr << "INPUT START" << std::endl;
  this->m_input->PrintSummary(std::cerr); 
  std::cerr << "INPUT END---------------------" << std::endl;
  vtkm::Bounds bounds = m_input->GetGlobalBounds();
  std::cerr << "GlobalBounds: " << std::endl;
  std::cerr << bounds.X.Min << " " << bounds.X.Max << " " << bounds.Y.Min << " " << bounds.Y.Max << " " << bounds.Z.Min << " " << bounds.Z.Max << std::endl; 
#endif
#endif

  std::vector<vtkm::Id> domain_ids = this->m_input->GetDomainIds(); 
  int num_domains = domain_ids.size();
  bool is_empty = this->m_input->IsEmpty();
  std::cerr << "IS EMPTY: " << is_empty << std::endl;

#if _DEBUG 
  std::cerr << "m_dims: " << m_dims[0] << " " << m_dims[1] << " " << m_dims[2] << std::endl;
  std::cerr << "m_origin: " << m_origin[0] << " " << m_origin[1] << " " << m_origin[2] << std::endl;
  std::cerr << "m_spacing: " << m_spacing[0] << " " << m_spacing[1] << " " << m_spacing[2] << std::endl;
  std::cerr << "input num cells: " << m_input->GetGlobalNumberOfCells() << std::endl; 
#ifdef VTKH_PARALLEL
  std::cerr << "par rank " << par_rank << " num domains: " << num_domains << std::endl;
#endif
#endif

//      }
//
//    }
//
//
//      std::cerr << "INPUT STARTT: "  << std::endl;
//      this->m_input->PrintSummary(std::cerr);
//      std::cerr << "INPUTT ENDDDDDDDDDDDDD" << std::endl;
//  } //end field loop

  vtkm::cont::DataSet local_res;
  for(int i = 0; i < num_domains; ++i)
  {
#ifdef VTKH_PARALLEL
    std::cerr << "rank: " << par_rank << " but now we are in num domains: " << num_domains <<std::endl;
#endif
    vtkm::cont::DataSet dom;
    
    if(this->m_input->HasDomainId(domain_ids[i]))
    {
      dom = this->m_input->GetDomainById(domain_ids[i]);
      for(const auto &field_name : m_fields)
      {
        //Uniform Grid Sample
        vtkh::vtkmProbe probe;
        probe.setBoxDims(m_dims);
        probe.setBoxOrigin(m_origin);
        probe.setBoxSpacing(m_spacing);
        probe.setInvalidValue(m_invalid_value);
        auto dataset = probe.Run(dom);
        vtkm::cont::Field tmp_field = dataset.GetField(field_name);

#if _DEBUG 
        std::cerr <<"UNIFORM GRID OUTPUT START: " << std::endl;
        dataset.PrintSummary(std::cerr);
        std::cerr <<"UNIFORM GRID OUTPUT END" << std::endl;
#endif
        vtkm::cont::Field valid_field;
        if(tmp_field.IsPointField())
        {
          vtkm::cont::Field point_field = dataset.GetPointField("HIDDEN");
          valid_field = point_field;
        }
        else
        {
          vtkm::cont::Field cell_field = dataset.GetCellField("HIDDEN");
          valid_field = cell_field;
        }

        std::string cs_name = dataset.GetCoordinateSystemName();
        if(!local_res.HasCoordinateSystem(cs_name))
        {
          local_res.CopyStructure(dataset);
          local_res.AddField(valid_field);
        }
        if(!local_res.HasField(field_name))
        {
          local_res.AddField(tmp_field);
        }
        else
        {
          vtkh::detail::LocalReduceField localreducefield(local_res,tmp_field,valid_field, field_name, m_invalid_value);
          localreducefield.LocalReduce();
        }
      }
    }
  }

#if _DEBUG 
  std::cerr <<" LOCAL RES START" << std::endl;
  local_res.PrintSummary(std::cerr);
  std::cerr <<" LOCAL RES END" << std::endl;
#endif


#ifdef VTKH_PARALLEL
  //take uniform sampled grid and reduce to root process
  vtkm::cont::DataSet reduced_output;
  reduced_output.CopyStructure(local_res);
  std::cerr << "rank is now this one : " << par_rank << std::endl;
  
  for(const auto &field_name : m_fields)
  {
  std::cerr << "rank is now this one : " << par_rank << " field name: " << field_name <<  std::endl;
    vtkm::cont::DataSet reduced;
    bool valid_field;
    vtkm::Id field_id = this->m_input->GetFieldType(field_name, valid_field);
    vtkh::detail::GlobalReduceField g_reducefield(local_res, field_name, m_invalid_value, m_dims, field_id, reduced);
    std::cerr << "rank ???? : " << par_rank << " field name: " << field_name <<  std::endl;
    g_reducefield.Reduce();
    vtkm::cont::Field reduced_field = reduced.GetField(field_name);
    reduced_output.AddField(reduced_field);
  }
  
  if(par_rank == 0)
  {
    this->m_output->AddDomain(reduced_output, 0);
  }

#if _DEBUG 
  //change to desired rank for output 
    if(par_rank == 0)
    {
      //this->m_output->AddDomain(output,0);
      //this->m_output->AddDomain(local_res,0);
      std::cerr << "FINAL OUTPUT START" << std::endl;
      this->m_output->PrintSummary(std::cerr); 
      std::cerr << "FINAL OUTPUT END---------------------" << std::endl;
    }
    std::cerr <<" PAR RANK " << par_rank << " at the very end" << std::endl;
#endif //end _DEBUG
#else //serial
  this->m_output->AddDomain(local_res,0);
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
UniformGrid::Fields(const std::vector<std::string> fields)
{
  m_fields = fields;
}

void
UniformGrid::InvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

} // namespace vtkh
