#ifndef ASCENT_DISPATCH_HPP
#define ASCENT_DISPATCH_HPP

#include "ascent_memory_manager.hpp"
#include "ascent_memory_interface.hpp"
#include "ascent_array.hpp"
#include "ascent_raja_policies.hpp"
#include "ascent_execution.hpp"

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

static inline int cell_shape(const std::string shape_type)
{
  int shape_id = 0;
  if(shape_type == "tri")
  {
      shape_id = 5;
  }
  else if(shape_type == "quad")
  {
      shape_id = 9;
  }
  else if(shape_type == "tet")
  {
      shape_id = 10;
  }
  else if(shape_type == "hex")
  {
      shape_id = 12;
  }
  else if(shape_type == "point")
  {
      shape_id = 1;
  }
  else if(shape_type == "line")
  {
      shape_id = 3;
  }
  else
  {
    ASCENT_ERROR("Unsupported cell type "<<shape_type);
  }
  return shape_id;
}

ASCENT_EXEC
static int num_indices(const int shape_id)
{
  int indices = 0;
  if(shape_id == 5)
  {
    indices = 3;
  }
  else if(shape_id == 9)
  {
    indices = 4;
  }
  else if(shape_id == 10)
  {
    indices = 4;
  }
  else if(shape_id == 12)
  {
    indices = 8;
  }
  else if(shape_id == 1)
  {
    indices = 1;
  }
  else if(shape_id == 3)
  {
    indices = 2;
  }
  return indices;
}

template<typename CoordsType, typename ConnType>
struct UnstructuredMesh
{

  MemoryAccessor<CoordsType> m_coords_x;
  MemoryAccessor<CoordsType> m_coords_y;
  MemoryAccessor<CoordsType> m_coords_z;
  MemoryAccessor<ConnType> m_conn;
  const int m_shape_type;
  const int m_dims;
  const int m_num_indices;
  const int m_num_cells;

  UnstructuredMesh() = delete;
  UnstructuredMesh(const std::string mem_space,
                   MemoryInterface<CoordsType> &coords,
                   MemoryInterface<ConnType> &conn,
                   const int shape_type,
                   const int dims)
    : m_coords_x(coords.accessor(mem_space, "x")),
      m_coords_y(coords.accessor(mem_space, "y")),
      m_coords_z(dims == 3 ? coords.accessor(mem_space, "z") :
                             // just use a dummy in this case
                             coords.accessor(mem_space, "x")),
      m_conn(conn.accessor(mem_space)),
      m_shape_type(shape_type),
      m_dims(dims),
      m_num_indices(num_indices(shape_type)),
      m_num_cells(m_conn.m_size / m_num_indices)
  {
  }

  // TODO: some sort of error checking mechinism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    const int offset = cell_index * m_num_indices;
    for(int i = 0; i < m_num_indices; ++i)
    {
      indices[i] = static_cast<int>(m_conn[offset + i]);
    }
  }

  ASCENT_EXEC
  void cell_vertex(const int vert_id, double vertex[3]) const
  {
    vertex[0] = static_cast<double>(m_coords_x[vert_id]);
    vertex[1] = static_cast<double>(m_coords_y[vert_id]);
    if(m_dims == 3)
    {
      vertex[2] = static_cast<double>(m_coords_z[vert_id]);
    }
    else
    {
      vertex[2] = 0.;
    }
  }

};


template<typename Function, typename Exec>
void
dispatch_memory_mesh(const conduit::Node &n_coords,
                     const conduit::Node &n_topo,
                     Function &func,
                     const Exec &exec)
{
  const std::string mem_space = Exec::memory_space;
  const std::string mesh_type = n_topo["type"].as_string();

  if(mesh_type == "unstructured")
  {
    const int shape_type = cell_shape(n_topo["elements/shape"].as_string());
    const int dims = n_coords["values"].number_of_children();

    if(dims < 2 || dims > 3)
    {
      std::cout<<"Bad dims "<<dims<<"\n";
      // TODO: log error
    }

    const std::string conn_path = "elements/connectivity";
    // figure out the types of coords
    if(is_conduit_type<conduit::float32>(n_coords["values/x"]))
    {
      MemoryInterface<conduit::float32> coords(n_coords);

      if(is_conduit_type<conduit::int32>(n_topo[conn_path]))
      {
        MemoryInterface<conduit::int32> conn(n_topo, conn_path);
        UnstructuredMesh<conduit::float32,conduit::int32> mesh(mem_space,
                                                               coords,
                                                               conn,
                                                               shape_type,
                                                               dims);
        func(mesh,exec);
      }
      else if(is_conduit_type<conduit::int64>(n_topo[conn_path]))
      {
        MemoryInterface<conduit::int64> conn(n_topo, conn_path);
        UnstructuredMesh<conduit::float32,conduit::int64> mesh(mem_space,
                                                               coords,
                                                               conn,
                                                               shape_type,
                                                               dims);
        func(mesh,exec);
      }
      else
      {
        std::cout<<"bad topo "<<n_topo[conn_path].to_summary_string()<<"\n";
        // TODO: log error
      }
    }
    else if(is_conduit_type<conduit::float64>(n_coords["values/x"]))
    {
      MemoryInterface<conduit::float64> coords(n_coords);

      if(is_conduit_type<conduit::int32>(n_topo[conn_path]))
      {
        MemoryInterface<conduit::int32> conn(n_topo, conn_path);
        UnstructuredMesh<conduit::float64,conduit::int32> mesh(mem_space,
                                                               coords,
                                                               conn,
                                                               shape_type,
                                                               dims);
        func(mesh,exec);
      }
      else if(is_conduit_type<conduit::int64>(n_topo[conn_path]))
      {
        MemoryInterface<conduit::int64> conn(n_topo, conn_path);
        UnstructuredMesh<conduit::float64,conduit::int64> mesh(mem_space,
                                                               coords,
                                                               conn,
                                                               shape_type,
                                                               dims);
        func(mesh,exec);
      }
      else
      {
        std::cout<<"bad topo "<<n_topo[conn_path].to_summary_string()<<"\n";
        // TODO: log error
      }
    }
    else
    {
      //std::cout<<"bad coords "<<n_coords.to_summary_string()<<" \n";
      n_coords.schema().print();
      n_coords.print();
      std::cout<<"bad coords "<<n_coords.to_summary_string()<<" \n";
      // TODO: log error
    }

  }
}

// TODO could make this a variadic template, maybe
template<typename Function>
void
exec_dispatch_mesh(const conduit::Node &n_coords,
                   const conduit::Node &n_topo,
                   Function &func)
{

  const std::string exec_policy = ExecutionManager::execution();

  std::cout<<"Mesh Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    dispatch_memory_mesh(n_coords,n_topo, func, exec);
  }
#if defined(ASCENT_USE_OPENMP)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    dispatch_memory_mesh(n_coords,n_topo, func, exec);
  }
#endif
#ifdef ASCENT_USE_CUDA
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    dispatch_memory_mesh(n_coords,n_topo, func, exec);
  }
#endif
  else
  {
    //TODO: log error this could hang things
    ASCENT_ERROR("Execution dispatch: unsupported execution policy "<<
                  exec_policy);
  }
}

template<typename Function, typename T>
void
exec_dispatch_array(Array<T> &array, Function &func)
{
  const std::string exec_policy = ExecutionManager::execution();

  std::cout<<"Array Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    func(array, exec);
  }
#if defined(ASCENT_USE_OPENMP)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    func(array, exec);
  }
#endif
#ifdef ASCENT_USE_CUDA
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    func(array, exec);
  }
#endif
  else
  {
    //TODO: log error this could hang things
    ASCENT_ERROR("Execution dispatch: unsupported execution policy "<<
                  exec_policy);
  }
}

template<typename Function>
void
exec_dispatch(Function &func)
{
  const std::string exec_policy = ExecutionManager::execution();

  std::cout<<"Exec only policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    func(exec);
  }
#if defined(ASCENT_USE_OPENMP)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    func(exec);
  }
#endif
#ifdef ASCENT_USE_CUDA
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    func(exec);
  }
#endif
  else
  {
    //TODO: log error this could hang things
    ASCENT_ERROR("Execution dispatch: unsupported execution policy "<<
                  exec_policy);
  }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
