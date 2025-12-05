
#include <vtkh/filters/Sample.hpp>
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

using Scalar_i32_hnd = vtkm::cont::ArrayHandle<vtkm::Int32>;
using Scalar_f32_hnd = vtkm::cont::ArrayHandle<vtkm::Float32>;
using Scalar_f64_hnd = vtkm::cont::ArrayHandle<vtkm::Float64>;

using Vec2_f32_hnd  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>;
using Vec2_f64_hnd  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>;

using Vec3_f32_hnd  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>;
using Vec3_f64_hnd  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>;

using Vec2_f32    = vtkm::Vec<vtkm::Float32, 2>;
using Vec3_f32    = vtkm::Vec<vtkm::Float32, 3>;

using Vec2_f64    = vtkm::Vec<vtkm::Float64, 2>;
using Vec3_f64    = vtkm::Vec<vtkm::Float64, 3>;


#define _DEBUG 0

//---------------------------------------------------------------------------//
namespace vtkh
{

//---------------------------------------------------------------------------//
namespace detail
{
vtkm::cont::Field 
MakeEmptyField(std::string field_name , vtkm::Id field_id, int num_values, vtkm::cont::Field::Association assoc, vtkm::Float64 val)
{
  if(field_id == 0)
  {
    std::vector<int> v_empty(num_values, (int) val);
    Scalar_i32_hnd ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 1)
  {
    std::vector<float> v_empty(num_values, (float) val);
    Scalar_f32_hnd ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 2)
  {
    std::vector<double> v_empty(num_values, (double) val);
    Scalar_f64_hnd ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 3)
  {
    Vec2_f32_hnd ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
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
    Vec2_f64_hnd ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>();
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
    Vec3_f32_hnd ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>();
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
    Vec3_f64_hnd  ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>();
    for(int i = 0; i < num_values; ++i)
    {
      Vec3_f64 empty_vec = vtkm::make_Vec(val, val, val);
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

//---------------------------------------------------------------------------//
#ifdef VTKH_PARALLEL
class GlobalReduceField
{
  const vtkm::cont::DataSet &m_dataset;
  std::string               m_field;
  vtkm::Float64             m_invalid_value;
  const int                 m_num_points;
  const vtkm::Id            m_field_id;
  vtkm::cont::DataSet       &m_result;

public:
  //-------------------------------------------------------------------------//
  GlobalReduceField(const vtkm::cont::DataSet &dataset,
                    const std::string field,
                    const int num_points,
                    const vtkm::Id field_id,
                    const vtkm::Float64 invalid_value,
                    vtkm::cont::DataSet &result)
    : m_dataset(dataset),
      m_field(field),
      m_num_points(num_points),
      m_field_id(field_id),
      m_invalid_value(invalid_value),
      m_result(result)
  {}

  //-------------------------------------------------------------------------//
  ~GlobalReduceField()
  {}

  //-------------------------------------------------------------------------//
  void
  Reduce()
  {
    vtkm::cont::DataSet res;
    res.CopyStructure(m_dataset);
    ReduceField r_field(m_field, m_dataset, m_invalid_value, m_num_points, m_field_id, m_result);
    r_field.reduce();
    return;
  }

  //-------------------------------------------------------------------------//
  struct ReduceField
  {
    const vtkm::cont::DataSet &m_data_set;
    const std::string m_field_name;
    const vtkm::Float64 m_invalid_value;
    const int m_num_points;
    const vtkm::Id m_field_id;
    vtkm::cont::DataSet &m_result;
  
    //-----------------------------------------------------------------------//
    ReduceField(const std::string field_name,
                const vtkm::cont::DataSet &data_set,
                const vtkm::Float64 invalid_value,
                const int num_points, 
                const vtkm::Id field_id, 
                vtkm::cont::DataSet &result)
      : m_field_name(field_name),
        m_data_set(data_set),
        m_invalid_value(invalid_value),
        m_num_points(num_points),
        m_field_id(field_id),
        m_result(result)
    {}

    //-----------------------------------------------------------------------//
    void
    reduce()
    {
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
      int par_rank;
      int par_size;
      MPI_Comm_rank(mpi_comm, &par_rank);
      MPI_Comm_size(mpi_comm, &par_size);

      vtkm::cont::Field res;
      vtkm::cont::Field field;

      bool is_empty = false;
      if(m_data_set.HasField(m_field_name))
      {
        is_empty = false;
        field = m_data_set.GetField(m_field_name);
      }
      else
      {
        is_empty = true;
        field = detail::MakeEmptyField(m_field_name,  m_field_id, m_num_points, vtkm::cont::Field::Association::Points, m_invalid_value); 
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
        uah_field = field.GetData();
        m_data_set.GetPointField("HIDDEN").GetData().AsArrayHandle(ah_mask);
      }
      else
      {
        ah_mask.AllocateAndFill(m_num_points, 2);
      }

      auto mask_portal = ah_mask.ReadPortal();
      

#if _DEBUG 
      std::cerr << "NUM_POINTS: " << m_num_points << std::endl;
#endif
      //Todo: NUM POINTS needs to be based on num_points
      //Todo: determine if field point or cell
      //Todo: check if all ranks have field? 

      //local and global point ownership by rank
      std::vector<int> l_rank_mask(m_num_points,-1);
      std::vector<int> g_rank_mask(m_num_points,-1);

      //if a valid/owned point, declare your rank
      for(int j = 0; j < m_num_points; ++j)
      {
        if(mask_portal.Get(j) == 0)
        {
          l_rank_mask[j] = par_rank;
        }
      }

      //take Max to figure out which ranks own which points
      MPI_Allreduce(l_rank_mask.data(), g_rank_mask.data(), m_num_points, MPI_INT, MPI_MAX, mpi_comm);

      //combine fields
      ////send to root process
      if(m_field_id == 0)
      {
#if _DEBUG 
        std::cerr << "In scalar int global reduce for field: " << field.GetName() << std::endl;
#endif
        //loop through field, zero out invalid and unowned values
        Scalar_i32_hnd ah_field = field.GetData().AsArrayHandle<Scalar_i32_hnd>();
        int *local_field = GetVTKMPointer(ah_field);
        std::vector<int> global_field(m_num_points,0);

        for(int i = 0; i < m_num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
        }

        MPI_Reduce(local_field, global_field.data(), m_num_points, MPI_INT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < m_num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = (int) m_invalid_value;
            }
          }
          
          Scalar_i32_hnd ah_out = vtkm::cont::make_ArrayHandle(global_field.data(), 
                                                               m_num_points,
                                                               vtkm::CopyFlag::On);
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
        Scalar_f32_hnd ah_field = field.GetData().AsArrayHandle<Scalar_f32_hnd>();
        float * local_field = GetVTKMPointer(ah_field);
        std::vector<float> global_field(m_num_points,0);

        for(int i = 0; i < m_num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,(float)0.0);
          }
        }

        MPI_Reduce(local_field, global_field.data(), m_num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < m_num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = (float) m_invalid_value;
            }
          }
          Scalar_f32_hnd ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),m_num_points,vtkm::CopyFlag::On);
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
        Scalar_f64_hnd ah_field = field.GetData().AsArrayHandle<Scalar_f64_hnd>();
        //loop through field, zero out invalid value
        for(int i = 0; i < m_num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,(double)0.0);
          }
        }

        double * local_field = GetVTKMPointer(ah_field);
        std::vector<double> global_field(m_num_points,0.0);
        MPI_Reduce(local_field, global_field.data(), m_num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < m_num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = (double)m_invalid_value;
            }
          }
          
          Scalar_f64_hnd ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),m_num_points,vtkm::CopyFlag::On);

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
        Vec2_f32_hnd ah_field = field.GetData().AsArrayHandle<Vec2_f32_hnd>();
        std::vector<float> local_x_points(m_num_points,(float)0.0);
        std::vector<float> local_y_points(m_num_points,(float)0.0);
        std::vector<float> global_x_points(m_num_points,(float)0.0);
        std::vector<float> global_y_points(m_num_points,(float)0.0);

        for(int i = 0; i < m_num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), m_num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), m_num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
            Vec2_f32_hnd ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
            ah_out.Allocate(m_num_points);
            for(int i = 0; i < m_num_points; ++i)
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
        Vec2_f64_hnd ah_field = field.GetData().AsArrayHandle<Vec2_f64_hnd>();
        std::vector<double> local_x_points(m_num_points,(double)0.0);
        std::vector<double> local_y_points(m_num_points,(double)0.0);
        std::vector<double> global_x_points(m_num_points,(double)0.0);
        std::vector<double> global_y_points(m_num_points,(double)0.0);

        for(int i = 0; i < m_num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), m_num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), m_num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          Vec2_f64_hnd ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>();
          ah_out.Allocate(m_num_points);
          for(int i = 0; i < m_num_points; ++i)
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
        Vec3_f32_hnd ah_field = field.GetData().AsArrayHandle<Vec3_f32_hnd>();
        std::vector<float> local_x_points(m_num_points,0);
        std::vector<float> local_y_points(m_num_points,0);
        std::vector<float> local_z_points(m_num_points,0);
        std::vector<float> global_x_points(m_num_points,0);
        std::vector<float> global_y_points(m_num_points,0);
        std::vector<float> global_z_points(m_num_points,0);

        for(int i = 0; i < m_num_points; ++i)
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

        MPI_Reduce(local_x_points.data(), global_x_points.data(), m_num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), m_num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), m_num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          Vec3_f32_hnd ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>();
          ah_out.Allocate(m_num_points);
          for(int i = 0; i < m_num_points; ++i)
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
        Vec3_f64_hnd ah_field = field.GetData().AsArrayHandle<Vec3_f64_hnd>();
        std::vector<double> local_x_points(m_num_points,0);
        std::vector<double> local_y_points(m_num_points,0);
        std::vector<double> local_z_points(m_num_points,0);
        std::vector<double> global_x_points(m_num_points,0);
        std::vector<double> global_y_points(m_num_points,0);
        std::vector<double> global_z_points(m_num_points,0);

        for(int i = 0; i < m_num_points; ++i)
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

        MPI_Reduce(local_x_points.data(), global_x_points.data(), m_num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), m_num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), m_num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          Vec3_f64_hnd ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>();
          ah_out.Allocate(m_num_points);
          for(int i = 0; i < m_num_points; ++i)
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
          m_result.AddField(field);
          return;
      }

      m_result.AddField(res);
      return;
    }
  }; //struct reduceFields

};//class globalReduceField
#endif
//---------------------------------------------------------------------------//
class LocalReduceField
{
  vtkm::cont::DataSet &m_dataset;
  vtkm::cont::Field   &m_field;
  vtkm::cont::Field   &m_mask;
  const std::string   m_field_name;
  vtkm::Float64       m_invalid_value;

public:
  //-------------------------------------------------------------------------//
  LocalReduceField(vtkm::cont::DataSet &dataset,
                   vtkm::cont::Field &field,
                   vtkm::cont::Field &mask,
                   const std::string &field_name,
                   vtkm::Float64 invalid_value)
    : m_dataset(dataset),
      m_field(field),
      m_mask(mask),
      m_field_name(field_name),
      m_invalid_value(invalid_value)
  {}
  
  //-------------------------------------------------------------------------//
  ~LocalReduceField()
  {}
  
  //-------------------------------------------------------------------------//
  void
  LocalReduce()
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

    if(uah_field.CanConvert<Scalar_i32_hnd>())
    {
      //loop through field, zero out invalid values
      Scalar_i32_hnd tmp_data = m_field.GetData().AsArrayHandle<Scalar_i32_hnd>();
      Scalar_i32_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Scalar_i32_hnd>();
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
    }//end Scalar_i32_hnd
    else if(uah_field.CanConvert<Scalar_f32_hnd>())
    {
      //loop through field, zero out invalid values
      Scalar_f32_hnd tmp_data = m_field.GetData().AsArrayHandle<Scalar_f32_hnd>();
      Scalar_f32_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Scalar_f32_hnd>();
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
    }//end Scalar_f32_hnd
    else if(uah_field.CanConvert<Scalar_f64_hnd>())
    {
      //loop through field, zero out invalid values
      Scalar_f64_hnd tmp_data = m_field.GetData().AsArrayHandle<Scalar_f64_hnd>();
      Scalar_f64_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Scalar_f64_hnd>();
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
    } //end Scalar_f64_hnd
    else if(uah_field.CanConvert<Vec2_f32_hnd>())
    {
      //loop through field, zero out invalid values
      Vec2_f32_hnd tmp_data = m_field.GetData().AsArrayHandle<Vec2_f32_hnd>();
      Vec2_f32_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Vec2_f32_hnd>();

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
    }//end Vec2_f32_hnd
    else if(uah_field.CanConvert<Vec2_f64_hnd>())
    {
      //loop through field, zero out invalid values
      Vec2_f64_hnd tmp_data = m_field.GetData().AsArrayHandle<Vec2_f64_hnd>();
      Vec2_f64_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Vec2_f64_hnd>();

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
    }//end Vec2_f64_hnd
    else if(uah_field.CanConvert<Vec3_f32_hnd>())
    {
      //loop through field, zero out invalid values
      Vec3_f32_hnd tmp_data   = m_field.GetData().AsArrayHandle<Vec3_f32_hnd>();
      Vec3_f32_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Vec3_f32_hnd>();

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
    }//end Vec3_f32_hnd
    else if(uah_field.CanConvert<Vec3_f64_hnd>())
    {
      //loop through field, zero out invalid values
      Vec3_f64_hnd tmp_data = m_field.GetData().AsArrayHandle<Vec3_f64_hnd>();
      Vec3_f64_hnd local_data = m_dataset.GetField(m_field_name).GetData().AsArrayHandle<Vec3_f64_hnd>();

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
    }//end Vec3_f64_hnd
    else
    {
        return;
    }
  }; //struct reduceField
};//class localReduceField

} //namespace detail

//---------------------------------------------------------------------------//
Sample::Sample()
	: m_invalid_value(std::numeric_limits<double>::min())
{

}

//---------------------------------------------------------------------------//
Sample::~Sample()
{

}

//---------------------------------------------------------------------------//
void
Sample::PreExecute()
{
  Filter::PreExecute();
}

//---------------------------------------------------------------------------//
void
Sample::DoExecute()
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
  const int num_domains = domain_ids.size();

#if _DEBUG 
#ifdef VTKH_PARALLEL
  std::cerr << "par rank " << par_rank << " num domains: " << num_domains << std::endl;
#endif
#endif

  vtkm::cont::DataSet local_res;
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet dom;
    
    if(this->m_input->HasDomainId(domain_ids[i]))
    {
      dom = this->m_input->GetDomainById(domain_ids[i]);
      for(const auto &field_name : m_fields)
      {
        //Uniform Grid Sample
        vtkh::vtkmProbe probe;
        probe.setPoints(m_points_xs,m_points_ys,m_points_zs);
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
          vtkh::detail::LocalReduceField localreducefield(local_res,
                                                          tmp_field,
                                                          valid_field,
                                                          field_name,
                                                          m_invalid_value);
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
  
  for(const auto &field_name : m_fields)
  {
    vtkm::cont::DataSet reduced;
    bool valid_field;
    vtkm::Id field_id = this->m_input->GetFieldType(field_name, valid_field);
    vtkh::detail::GlobalReduceField g_reducefield(local_res, field_name, m_invalid_value, m_num_samples, field_id, reduced);
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

//---------------------------------------------------------------------------//
void
Sample::PostExecute()
{
  Filter::PostExecute();
}

//---------------------------------------------------------------------------//
std::string
Sample::GetName() const
{
  return "vtkh::Sample";
}
//---------------------------------------------------------------------------//
void
Sample::Line(int num_samples,
             double x_start,
             double y_start,
             double z_start,
             double x_end,
             double y_end,
             double z_end)

{
    m_num_samples = num_samples;

    int line_spatial_num_points = 3;
    // check for 2d case
    if(z_start == z_end)
    {
        line_spatial_num_points = 2;
    }

    // alloc array handles to hold num_samples
    // alloc xs, ys
    m_points_xs.Allocate(num_samples);
    m_points_ys.Allocate(num_samples);
    if(line_spatial_num_points ==3)
    {
      // alloc z
      m_points_zs.Allocate(num_samples);
    }
    else
    {
      m_points_zs.Allocate(0);
    }

    double line_dpos   = 1.0 / double(num_samples);

    for(int i=0;i<num_samples;i++)
    {
        double line_pos = i * line_dpos;
        double x = (x_end - x_start) * line_pos + x_start;
        double y = (y_end - y_start) * line_pos + y_start;
        m_points_xs.WritePortal().Set(i,x);
        m_points_ys.WritePortal().Set(i,y);

        if(line_spatial_num_points ==3)
        {
            double z = (z_end - z_start) * line_pos + z_start;
            m_points_zs.WritePortal().Set(i,z);
        }
    }
}

//---------------------------------------------------------------------------//
void
Sample::Box(int *num_points,
            double x_start,
            double y_start,
            double z_start,
            double x_end,
            double y_end,
            double z_end)

{
  int m_num_samples = num_points[0]*num_points[1]*num_points[2];

  // alloc array handles to hold num_samples
  // alloc xs, ys, zs
  m_points_xs.Allocate(m_num_samples);
  m_points_ys.Allocate(m_num_samples);
  m_points_zs.Allocate(m_num_samples);
  
  auto x_portal = m_points_xs.WritePortal();
  auto y_portal = m_points_ys.WritePortal();
  auto z_portal = m_points_zs.WritePortal();
  
  const int Nx = num_points[0];
  const int Ny = num_points[1];
  const int Nz = num_points[2];
  //unset *_start&*_end are both set to (max-min)/2
  const double dx = (Nx > 1) ? (x_end - x_start) / double(Nx - 1) : 0.0;
  const double dy = (Ny > 1) ? (y_end - y_start) / double(Ny - 1) : 0.0;
  const double dz = (Nz > 1) ? (z_end - z_start) / double(Nz - 1) : 0.0;

#if _DEBUG 
  std::cerr << "Nx: " << Nx << " Ny: " << Ny << " Nz: " << Nz << std::endl;
  
  std::cerr << "x_start: " << x_start << " x_end: " << x_end << std::endl;
  std::cerr << "y_start: " << y_start << " y_end: " << y_end << std::endl;
  std::cerr << "z_start: " << z_start << " z_end: " << z_end << std::endl;
  std::cerr << "dx: " << dx << " dy: " << dy << " dz: " << dz << std::endl;
#endif
  
  int idx = 0;
  for (int i = 0; i < Nx; ++i)
  {
    double x = (Nx > 1) ? (x_start + i * dx) : x_start;
    for (int j = 0; j < Ny; ++j)
    {
      double y = (Ny > 1) ? (y_start + j * dy) : y_start;
      for (int k = 0; k < Nz; ++k)
      {
        double z = (Nz > 1) ? (z_start + k * dz) : z_start;
        x_portal.Set(idx, x);
        y_portal.Set(idx, y);
        z_portal.Set(idx, z);
#if _DEBUG 
        std::cerr << "i: " << i << " j: " << j << " k: " << k << " x: " << x << " y: " << y << " z: " << z << std::endl;
        std::cerr << "idx: " << idx << std::endl;
#endif
        idx++;
      }
    }
  }
}
//---------------------------------------------------------------------------//
void
Sample::Points(vtkm::cont::ArrayHandle<vtkm::Float64> xs,
               vtkm::cont::ArrayHandle<vtkm::Float64> ys,
               vtkm::cont::ArrayHandle<vtkm::Float64> zs)
{
  m_num_samples = xs.GetNumberOfValues();
  m_points_xs = xs;
  m_points_ys = ys;
  m_points_zs = zs;
}

//---------------------------------------------------------------------------//
void
Sample::Fields(const std::vector<std::string> fields)
{
  m_fields = fields;
}

//---------------------------------------------------------------------------//
void
Sample::InvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

} // namespace vtkh
