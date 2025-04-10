
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
MakeEmptyField(std::string field_name , vtkm::Id field_id, Vec3f dims, vtkm::cont::Field::Association assoc)
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
    std::vector<int> v_empty(num_values, 0.0);
    scalarI ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 1)
  {
    std::vector<float> v_empty(num_values, 0.0);
    scalarF ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 2)
  {
    std::vector<double> v_empty(num_values, 0.0);
    scalarD ah_empty = vtkm::cont::make_ArrayHandle(v_empty.data(),num_values,vtkm::CopyFlag::On);
    vtkm::cont::Field f_empty(field_name,
                              assoc,
                              ah_empty);
    return f_empty;
  }
  else if(field_id == 3)
  {
    vec2_32 ah_empty = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
    for(int i = 0; i < num_values; ++i)
    {
      Vec2f empty_vec = vtkm::make_Vec(0.0,0.0);
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
    for(int i = 0; i < num_values; ++i)
    {
      Vec2d empty_vec = vtkm::make_Vec(0.0,0.0);
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
    for(int i = 0; i < num_values; ++i)
    {
      Vec3f empty_vec = vtkm::make_Vec(0.0,0.0,0.0);
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
      Vec3d empty_vec = vtkm::make_Vec(0.0,0.0,0.0);
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
  vtkm::Float64             m_invalid_value;

public:
  GlobalReduceField(const vtkm::cont::DataSet &dataset, const std::string &field, vtkm::Float64 &invalid_value)
    : m_dataset(dataset),
      m_field(field),
      m_invalid_value(invalid_value)
  {}
  ~GlobalReduceField()
  {}

  vtkm::cont::DataSet Reduce()
  {
    vtkm::cont::DataSet res;
    res.CopyStructure(m_dataset);
    vtkm::cont::Field field = m_dataset.GetField(m_field);
    ReduceField r_field(field, m_dataset, m_invalid_value);
    vtkm::cont::Field res_field = r_field.reduce();
    res.AddField(res_field);

    return res;
  }

  struct ReduceField
  {
    vtkm::cont::Field &m_input_field;
    const vtkm::cont::DataSet &m_data_set;
    vtkm::Float64 &m_invalid_value;
  
    ReduceField(vtkm::cont::Field &input_field,
                const vtkm::cont::DataSet &data_set, 
                vtkm::Float64 &invalid_value)
      : m_input_field(input_field),
        m_data_set(data_set),
        m_invalid_value(invalid_value)
    {}

    vtkm::cont::Field reduce()
    {
      if(m_input_field.GetName() == "HIDDEN")
      {
        return m_input_field;
      }

      vtkm::cont::Field res;
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
      int par_rank;
      int par_size;
      MPI_Comm_rank(mpi_comm, &par_rank);
      MPI_Comm_size(mpi_comm, &par_size);

      vtkm::cont::UnknownArrayHandle uah_field = m_input_field.GetData();

      vtkm::cont::ArrayHandle<unsigned char> ah_mask;
      if(m_input_field.IsPointField())
      {
        m_data_set.GetPointField("HIDDEN").GetData().AsArrayHandle(ah_mask);
      }
      else
      {
        m_data_set.GetCellField("HIDDEN").GetData().AsArrayHandle(ah_mask);
      }
      auto mask_portal = ah_mask.ReadPortal();
      int num_points = mask_portal.GetNumberOfValues();
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
        if(mask_portal.Get(j) == 0)
	      {
          l_rank_mask[j] = par_rank;
	      }
      }

      //take Max to figure out which ranks own which points
      MPI_Allreduce(l_rank_mask.data(), g_rank_mask.data(), num_points, MPI_INT, MPI_MAX, mpi_comm);

      //combine fields
      ////send to root process
      if(uah_field.CanConvert<scalarI>())
      {
        //loop through field, zero out invalid and unowned values
        scalarI ah_field = m_input_field.GetData().AsArrayHandle<scalarI>();
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
              global_field[i] = m_invalid_value;
#if _DEBUG 
              global_field[i] = par_rank*10;
#endif
            }
          }
          
          scalarI ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
          vtkm::cont::Field out_field(m_input_field.GetName(),
                                      m_input_field.GetAssociation(),
                                      ah_out);
          res = out_field;
        }
        else
        {
          res = m_input_field;
        }
      }//end scalarI
      else if(uah_field.CanConvert<scalarF>())
      {
        //loop through field, zero out invalid value
        scalarF ah_field = m_input_field.GetData().AsArrayHandle<scalarF>();
        float * local_field = GetVTKMPointer(ah_field);
        std::vector<float> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_FLOAT, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = m_invalid_value;
#if _DEBUG 
              global_field[i] = par_rank*10;
#endif
            }
          }
          scalarF ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
          vtkm::cont::Field out_field(m_input_field.GetName(),

                                      m_input_field.GetAssociation(),
                                      ah_out);

          res = out_field;
        }
        else
        {
          res = m_input_field;
        }
      }//end scalarF
      else if(uah_field.CanConvert<scalarD>())
      {
        scalarD ah_field = uah_field.AsArrayHandle<scalarD>();
        //loop through field, zero out invalid value
        for(int i = 0; i < num_points; ++i)
        {
          //if we do not own the point, set it to zero
          if(g_rank_mask[i] != par_rank)
          {
            ah_field.WritePortal().Set(i,0);
          }
        }
        double * local_field = GetVTKMPointer(ah_field);
        std::vector<double> global_field(num_points,0);
        MPI_Reduce(local_field, global_field.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        if(par_rank == 0)
        {
          for(int i = 0; i < num_points; ++i)
          {
            if(g_rank_mask[i] == -1)
            {
              global_field[i] = m_invalid_value;
#if _DEBUG 
              global_field[i] = par_rank*10;
#endif
            }
          }
          
          scalarD ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
          vtkm::cont::Field out_field(m_input_field.GetName(),
                                      m_input_field.GetAssociation(),
                                      ah_out);
          res = out_field;
        }
        else
        {
          res = m_input_field;
        }
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
                global_x_points[i] = m_invalid_value;
                global_y_points[i] = m_invalid_value;
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
        {
          res = m_input_field;
        }
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
        
        vtkm::cont::Field out_field(m_input_field.GetName(),
                                    m_input_field.GetAssociation(),
                                    ah_out);

        res = out_field;
      }
      else
      {
        res = m_input_field;
      }
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
          vtkm::cont::Field out_field(m_input_field.GetName(),
                                      m_input_field.GetAssociation(),
                                      ah_out);

          res = out_field;
        }
        else
        {
          res = m_input_field;
        }
      }//end vec3_64
      else
      {
          return m_input_field;
      }

      return res;
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

    //create mask where true == 1
//    std::vector<int> l_mask(num_points,1);
//    for(int j = 0; j < num_points; ++j)
//    {
//      //if combined domains have valid location && indiv domain has valid location
//      l_mask[j] = (tmp_mask_portal.Get(j)==0) && (r_local_mask_portal.Get(j)==0);
//      if(!l_mask[j])
//      {
//        if((r_local_mask_portal.Get(j) == 2) && (tmp_mask+portal.Get(j) == 0))
//          w_local_mask_portal.Set(j,0);
//      }
//    }

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
  vtkm::Range par_den_range= m_input->GetRange("den").ReadPortal().Get(0);
  std::cerr <<"  par rank: " << par_rank << "local range: " <<  par_den_range.Min << " " << par_den_range.Max << std::endl;
  std::cerr << "INPUT START" << std::endl;
  this->m_input->PrintSummary(std::cerr); 
  std::cerr << "INPUT END---------------------" << std::endl;
  vtkm::Range den_range= m_input->GetGlobalRange("den").ReadPortal().Get(0);
  std::cerr << "GLOBAL Rageng: " << den_range.Min << " " << den_range.Max << std::endl;
  vtkm::Bounds bounds = m_input->GetGlobalBounds();
  std::cerr << "GlobalBounds: " << std::endl;
  std::cerr << bounds.X.Min << " " << bounds.X.Max << " " << bounds.Y.Min << " " << bounds.Y.Max << " " << bounds.Z.Min << " " << bounds.Z.Max << std::endl; 
#endif
#endif


  std::vector<vtkm::Id> domain_ids = this->m_input->GetDomainIds(); 
  const int num_domains = domain_ids.size();
  //add mask to keep track of valid points after sampling
  //this->m_input->AddConstantPointField(0.0, "mask");

  //put vtkm datasets into a partitionedDS for vtkm::Merge
//  vtkm::cont::PartitionedDataSet sampled_doms;
//  for(int i = 0; i < num_domains; ++i)
//  {
//    vtkm::cont::DataSet dom;
//    
//    if(this->m_input->HasDomainId(domain_ids[i]))
//    {
//      dom = this->m_input->GetDomainById(domain_ids[i]);
//
//      sampled_doms.AppendPartition(dom);
//    }
//  }
//#ifdef VTKH_PARALLEL
//  //if there is no data, add some empty 
//  vtkm::cont::DataSet tmp_empty;
//  if(sampled_doms.GetNumberOfPartitions() == 0)
//  {
//    tmp_empty = vtkm::cont::DataSetBuilderUniform::Create(m_dims, m_origin, m_spacing);
//    bool valid_field;
//    vtkm::cont::Field::Association assoc = this->m_input->GetFieldAssociation(m_field, valid_field);
//
//    if(!valid_field)
//    {
//      this->m_output = this->m_input;
//      return;
//    }
//    
//    vtkm::Id field_id = this->m_input->GetFieldType(m_field, valid_field);
//    vtkm::cont::Field empty_field = vtkh::detail::MakeEmptyField(m_field,field_id,m_dims,assoc);
//    tmp_empty.AddField(empty_field);
//    sampled_doms.AppendPartition(tmp_empty);
//  }
//#endif

  ///
  /// Approach we can use that would not need MergeDataSets:
  ///
  /// (This will also be more memory efficient b/c we don't need to convert
  /// everything to a fused unstructured grid)
  ///
  /// create local output grid `local_res` (including masking info)
  /// create global output grid `global_res` (including masking info)
  /// create tmp local output grid `local_res_tmp` (including masking info)
  ///
  /// for each domain `d`:
  ///   reset local_res_tmp (including making info)
  ///   execute probe filter on `d` with output in `local_res_tmp`
  ///   combine results from `local_res_tmp` into `local_res`
  /// if mpi parallel:
  ///.  Use global reduce w/ `local_res` to create `global_res`
  ///
  ///.  (We know that all ranks will have something to reduce, b/c
  ///.   even if they have no domains, the still created the local
  ///    output grid)

//  vtkm::filter::multi_block::MergeDataSets mergeDataSets;
//  mergeDataSets.SetInvalidValue(m_invalid_value);
//  //return a partitiondataset
//  auto merged = mergeDataSets.Execute(sampled_doms);
//  auto result = merged.GetPartitions();
#if _DEBUG 
  std::cerr << "m_dims: " << m_dims[0] << " " << m_dims[1] << " " << m_dims[2] << std::endl;
  std::cerr << "m_origin: " << m_origin[0] << " " << m_origin[1] << " " << m_origin[2] << std::endl;
  std::cerr << "m_spacing: " << m_spacing[0] << " " << m_spacing[1] << " " << m_spacing[2] << std::endl;
  std::cerr << "input num cells: " << m_input->GetGlobalNumberOfCells() << std::endl; 
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
      //Uniform Grid Sample
      vtkh::vtkmProbe probe;
      probe.dims(m_dims);
      probe.origin(m_origin);
      probe.spacing(m_spacing);
      probe.invalidValue(m_invalid_value);
      auto dataset = probe.Run(dom);
      vtkm::cont::Field tmp_field = dataset.GetField(m_field);

#if _DEBUG 
      std::cerr <<"UNIFORM GRID OUTPUT: " << std::endl;
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
      if(!local_res.HasField(m_field))
      {
        local_res.AddField(tmp_field);
      }
      else
      {
        vtkh::detail::LocalReduceField localreducefield(local_res,tmp_field,valid_field, m_field, m_invalid_value);
	localreducefield.LocalReduce();
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
  vtkh::detail::GlobalReduceField g_reducefields(local_res, m_field, m_invalid_value);
  auto output = g_reducefields.Reduce();
#if _DEBUG 
  //change to desired rank for output 
  if(par_rank == 0)
  {
    //this->m_output->AddDomain(output,0);
    this->m_output->AddDomain(local_res,0);
    vtkm::Range output_den_range= this->m_output->GetRange("den").ReadPortal().Get(0);
    std::cerr <<"  par rank: " << par_rank << "output range: " <<  output_den_range.Min << " " << output_den_range.Max << std::endl;
    std::cerr << "par rank: " << par_rank << " output num cells: " << m_output->GetGlobalNumberOfCells() << std::endl; 
    std::cerr << "FINAL OUTPUT START" << std::endl;
    this->m_output->PrintSummary(std::cerr); 
    std::cerr << "FINAL OUTPUT END---------------------" << std::endl;
  }
#endif
  if(par_rank == 0)
  {
    this->m_output->AddDomain(output,0);
  }
#else
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
UniformGrid::Field(const std::string field)
{
  m_field = field;
}

void
UniformGrid::InvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

} // namespace vtkh
