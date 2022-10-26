#ifndef ASCENT_DISPATCH_HPP
#define ASCENT_DISPATCH_HPP

#include "ascent_memory_manager.hpp"
#include "ascent_memory_interface.hpp"
#include "ascent_meshes.hpp"
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
      n_coords.schema().print();
      n_coords.print();
      std::cout<<"bad coords "<<n_coords.to_summary_string()<<" \n";
      // TODO: log error
    }
    // end unstructured mesh
  }
  else if(mesh_type == "uniform")
  {

    UniformMesh mesh(n_coords);
    func(mesh,exec);
  }
  else if(mesh_type == "rectilinear")
  {
    const int dims = n_coords["values"].number_of_children();
    if(dims < 2 || dims > 3)
    {
      std::cout<<"Bad dims "<<dims<<"\n";
      // TODO: log error
    }

    // figure out the types of coords
    if(is_conduit_type<conduit::float32>(n_coords["values/x"]))
    {
      MemoryInterface<conduit::float32> coords(n_coords);
      RectilinearMesh<conduit::float32> mesh(mem_space,
                                             coords,
                                             dims);
      func(mesh,exec);
    }
    else if(is_conduit_type<conduit::float64>(n_coords["values/x"]))
    {
      MemoryInterface<conduit::float32> coords(n_coords);
      RectilinearMesh<conduit::float32> mesh(mem_space,
                                             coords,
                                             dims);
      func(mesh,exec);
    }
    else
    {
      std::cout<<"Bad coordinates type rectilinear\n";
    }
  }
  else if(mesh_type == "structured")
  {
    const int dims = n_coords["values"].number_of_children();
    if(dims < 2 || dims > 3)
    {
      std::cout<<"Bad dims "<<dims<<"\n";
      // TODO: log error
    }
    int point_dims[3] = {0,0,0};
    point_dims[0] = n_topo["element/dims/i"].to_int32();
    point_dims[1] = n_topo["element/dims/j"].to_int32();
    if(dims == 3)
    {
      if(!n_topo.has_path("element/dims/k"))
      {
        std::cout<<"Coordinate system disagrees with element dims\n";
      }
      else
      {
        point_dims[2] = n_topo["element/dims/k"].to_int32();
      }
    }

    // figure out the types of coords
    if(is_conduit_type<conduit::float32>(n_coords["values/x"]))
    {
      MemoryInterface<conduit::float32> coords(n_coords);
      StructuredMesh<conduit::float32> mesh(mem_space,
                                            coords,
                                            dims,
                                            point_dims);
      func(mesh,exec);
    }
    else if(is_conduit_type<conduit::float64>(n_coords["values/x"]))
    {
      MemoryInterface<conduit::float32> coords(n_coords);
      StructuredMesh<conduit::float32> mesh(mem_space,
                                            coords,
                                            dims,
                                            point_dims);
      func(mesh,exec);
    }
    else
    {
      std::cout<<"Bad coordinates type structured\n";
    }
  }
  else
  {
    std::cout<<"mesh type not implemented:  "<<mesh_type<<"\n";
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

  //std::cout<<"Mesh Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    dispatch_memory_mesh(n_coords,n_topo, func, exec);
  }
#if defined(ASCENT_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    dispatch_memory_mesh(n_coords,n_topo, func, exec);
  }
#endif
#if defined(ASCENT_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    dispatch_memory_mesh(n_coords,n_topo, func, exec);
  }
#endif
#if defined(ASCENT_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
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

  //std::cout<<"Array Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    func(array, exec);
  }
#if defined(ASCENT_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    func(array, exec);
  }
#endif
#if defined(ASCENT_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    func(array, exec);
  }
#endif
#if defined(ASCENT_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
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

  //std::cout<<"Exec only policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    func(exec);
  }
#if defined(ASCENT_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    func(exec);
  }
#endif
#if defined(ASCENT_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    func(exec);
  }
#endif
#if defined(ASCENT_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
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
