//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef ASCENT_BLUEPRINT_DEVICE_DISPATCH_HPP
#define ASCENT_BLUEPRINT_DEVICE_DISPATCH_HPP

#include "ascent_execution_policies.hpp"
#include "ascent_execution_manager.hpp"
#include "ascent_memory_manager.hpp"
#include "ascent_array.hpp"

#include "ascent_blueprint_type_utils.hpp"
#include "ascent_blueprint_device_mesh_objects.hpp"

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


//-----------------------------------------------------------------------------
// exec_dispatch_{type} variants:
//   exec_dispatch_function
//   exec_dispatch_array
//   exec_dispatch_mcarray_component
//   exec_dispatch_mesh
//
//   exec_dispatch_two_leaves <-- bad name!
//
//-----------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////
// TODO NEEDS A BETTER NAME !!!!!
// TODO THIS NEEDS TO BE RAJAFIED
template<typename Function>
conduit::Node
exec_dispatch_two_leaves(const conduit::Node &values0,
                         const conduit::Node &values1,
                         const Function &func)
{
  // check for single component scalar
  int num_children0 = values0.number_of_children();
  int num_children1 = values1.number_of_children();
  if(num_children0 > 1 || num_children1 > 1)
  {
    ASCENT_ERROR("exec_dispatch_two_leaves: Internal error: expected leaf arrays.");
  }
  const conduit::Node &vals0 = num_children0 == 0 ? values0 : values0.child(0);
  const conduit::Node &vals1 = num_children1 == 0 ? values1 : values1.child(0);

  conduit::Node res;
  const int num_vals0 = vals0.dtype().number_of_elements();
  const int num_vals1 = vals1.dtype().number_of_elements();

  if(vals0.dtype().is_float32())
  {
    const conduit::float32 *ptr0 =  vals0.as_float32_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64()) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("exec_dispatch_two_leaves: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else if(vals0.dtype().is_float64())
  {
    const conduit::float64 *ptr0 =  vals0.as_float64_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64()) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("exec_dispatch_two_leaves: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else if(vals0.dtype().is_int32())
  {
    const conduit::int32 *ptr0 =  vals0.as_int32_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64()) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("exec_dispatch_two_leaves: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else if(vals0.dtype().is_int64())
  {
    const conduit::int64 *ptr0 =  vals0.as_int64_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64()) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("exec_dispatch_two_leaves: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else
  {
    ASCENT_ERROR("exec_dispatch_two_leavesh: unsupported array type for array0: "<<
                  values0.schema().to_string());
  }
  return res;
}
////////////////////////////////////////////////////////////////////////////////////


//-----------------------------------------------------------------------------
template<typename Function, typename Exec>
conduit::Node exec_dispatch_mcarray_component(const conduit::Node &node,
                                              const std::string &component,
                                              const Function &func,
                                              const Exec &exec)
{
  const std::string mem_space = Exec::memory_space;

  conduit::Node res;
  if(mcarray_is_float32(node))
  {
    MCArray<conduit::float32> farray(node);
    DeviceAccessor<conduit::float32> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else if(mcarray_is_float64(node))
  {
    MCArray<conduit::float64> farray(node);
    DeviceAccessor<conduit::float64> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else if(mcarray_is_int32(node))
  {
    MCArray<conduit::int32> farray(node);
    DeviceAccessor<conduit::int32> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else if(mcarray_is_int64(node))
  {
    MCArray<conduit::int64> farray(node);
    DeviceAccessor<conduit::int64> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else
  {
    ASCENT_ERROR("exec_dispatch_mcarray_component: unsupported type "<<
                  node.schema().to_string());
  }
  return res;
}



//-----------------------------------------------------------------------------
template<typename Function>
conduit::Node
exec_dispatch_mcarray_component(const conduit::Node &node,
                                const std::string &component,
                                const Function &func)
{

  conduit::Node res;
  const std::string exec_policy = ExecutionManager::execution_policy();
  //std::cout<<"Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    res = exec_dispatch_mcarray_component(node, component, func, exec);
  }
#if defined(ASCENT_RAJA_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    res = exec_dispatch_mcarray_component(node, component, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    res = exec_dispatch_mcarray_component(node, component, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
    res = exec_dispatch_mcarray_component(node, component, func, exec);
  }
#endif
  else
  {
    ASCENT_ERROR("exec_dispatch_mcarray_component: unsupported execution policy "<<
                  exec_policy);
  }
  return res;
}


//-----------------------------------------------------------------------------
template<typename Function, typename Exec>
void
exec_dispatch_mesh(const conduit::Node &n_coords,
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
      MCArray<conduit::float32> coords(n_coords["values"]);

      if(is_conduit_type<conduit::int32>(n_topo[conn_path]))
      {
        MCArray<conduit::int32> conn(n_topo[conn_path]);
        UnstructuredMesh<conduit::float32,conduit::int32> mesh(mem_space,
                                                               coords,
                                                               conn,
                                                               shape_type,
                                                               dims);
        func(mesh,exec);
      }
      else if(is_conduit_type<conduit::int64>(n_topo[conn_path]))
      {
        MCArray<conduit::int64> conn(n_topo[conn_path]);
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
      MCArray<conduit::float64> coords(n_coords["values"]);

      if(is_conduit_type<conduit::int32>(n_topo[conn_path]))
      {
        MCArray<conduit::int32> conn(n_topo[conn_path]);
        UnstructuredMesh<conduit::float64,conduit::int32> mesh(mem_space,
                                                               coords,
                                                               conn,
                                                               shape_type,
                                                               dims);
        func(mesh,exec);
      }
      else if(is_conduit_type<conduit::int64>(n_topo[conn_path]))
      {
        MCArray<conduit::int64> conn(n_topo[conn_path]);
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
      MCArray<conduit::float32> x_coords(n_coords["values/x"]);
      MCArray<conduit::float32> y_coords(n_coords["values/y"]);
      std::string zvalue_path = "values/z";

      if(!n_coords.has_path("values/z"))
      {
        zvalue_path = "values/x";
      }
      MCArray<conduit::float32> z_coords(n_coords[zvalue_path]);

      RectilinearMesh<conduit::float32> mesh(mem_space,
                                             x_coords,
                                             y_coords,
                                             z_coords,
                                             dims);
      func(mesh,exec);
    }
    else if(is_conduit_type<conduit::float64>(n_coords["values/x"]))
    {
      MCArray<conduit::float64> x_coords(n_coords["values/x"]);
      MCArray<conduit::float64> y_coords(n_coords["values/y"]);
      std::string zvalue_path = "values/z";

      if(!n_coords.has_path("values/z"))
      {
        zvalue_path = "values/x";
      }
      MCArray<conduit::float64> z_coords(n_coords[zvalue_path]);

      RectilinearMesh<conduit::float64> mesh(mem_space,
                                             x_coords,
                                             y_coords,
                                             z_coords,
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
      MCArray<conduit::float32> coords(n_coords["values"]);
      StructuredMesh<conduit::float32> mesh(mem_space,
                                            coords,
                                            dims,
                                            point_dims);
      func(mesh,exec);
    }
    else if(is_conduit_type<conduit::float64>(n_coords["values/x"]))
    {
      MCArray<conduit::float32> coords(n_coords["values"]);
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
    std::cout<<"exec_dispatch_mesh: mesh type not implemented:  "<<mesh_type<<"\n";
  }


}

//-----------------------------------------------------------------------------
// TODO could make this a variadic template, maybe
template<typename Function>
void
exec_dispatch_mesh(const conduit::Node &n_coords,
                   const conduit::Node &n_topo,
                   Function &func)
{

  const std::string exec_policy = ExecutionManager::execution_policy();

  //std::cout<<"Mesh Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    exec_dispatch_mesh(n_coords,n_topo, func, exec);
  }
#if defined(ASCENT_RAJA_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    exec_dispatch_mesh(n_coords,n_topo, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    exec_dispatch_mesh(n_coords,n_topo, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
    exec_dispatch_mesh(n_coords,n_topo, func, exec);
  }
#endif
  else
  {
    //TODO: log error this could hang things
    ASCENT_ERROR("exec_dispatch_mesh: unsupported execution policy "<<
                  exec_policy);
  }
}

//-----------------------------------------------------------------------------
//dispatch memory for a derived field (DF) binary operation
template<typename Function, typename Exec>
conduit::Node
dispatch_memory_binary_df(const conduit::Node &l_field,
                          const conduit::Node &r_field,
                          std::string component,
                          const Function &func,
                          const Exec &exec)
{
    const std::string mem_space = Exec::memory_space;

    conduit::Node res;
    if(field_is_float32(l_field))
    {
        if(!field_is_float32(r_field))
        {
          ASCENT_ERROR("Type dispatch: mismatch array types\n"<<
                       l_field.schema().to_string() <<
                       "\n vs. \n" <<
                       r_field.schema().to_string());
        }

        MCArray<conduit::float32> l_farray(l_field["values"]);
        MCArray<conduit::float32> r_farray(r_field["values"]);
        DeviceAccessor<conduit::float32> l_accessor = l_farray.accessor(mem_space, component);
        DeviceAccessor<conduit::float32> r_accessor = r_farray.accessor(mem_space, component);
        res = func(l_accessor, r_accessor, exec);

    }
    else if(field_is_float64(l_field))
    {
        if(!field_is_float64(r_field))
        {
            ASCENT_ERROR("Type dispatch: mismatch array types\n"<<
                         l_field.schema().to_string() <<
                          "\n vs. \n" <<
                              r_field.schema().to_string());
        }

        MCArray<conduit::float64> l_farray(l_field["values"]);
        MCArray<conduit::float64> r_farray(r_field["values"]);
        DeviceAccessor<conduit::float64>  l_accessor = l_farray.accessor(mem_space, component);
        DeviceAccessor<conduit::float64>  r_accessor = r_farray.accessor(mem_space, component);
        res = func(l_accessor, r_accessor, exec);
    }
    else if(field_is_int32(l_field))
    {
        if(!field_is_int32(r_field))
        {
            ASCENT_ERROR("Type dispatch: mismatch array types\n"<<
                         l_field.schema().to_string() <<
                         "\n vs. \n" <<
                             r_field.schema().to_string());
        }

        MCArray<conduit::int32> l_farray(l_field["values"]);
        MCArray<conduit::int32> r_farray(r_field["values"]);
        DeviceAccessor<conduit::int32>  l_accessor = l_farray.accessor(mem_space, component);
        DeviceAccessor<conduit::int32>  r_accessor = r_farray.accessor(mem_space, component);
        res = func(l_accessor, r_accessor, exec);
    }
    else if(field_is_int64(l_field))
    {

        if(!field_is_int64(r_field))
        {
          ASCENT_ERROR("Type dispatch: mismatch array types\n"<<
                       l_field.schema().to_string() <<
                       "\n vs. \n" <<
                       r_field.schema().to_string());
        }

        MCArray<conduit::int64> l_farray(l_field["values"]);
        MCArray<conduit::int64> r_farray(r_field["values"]);
        DeviceAccessor<conduit::int64>  l_accessor = l_farray.accessor(mem_space, component);
        DeviceAccessor<conduit::int64>  r_accessor = r_farray.accessor(mem_space, component);
        res = func(l_accessor, r_accessor, exec);
    }
    else
    {
        ASCENT_ERROR("Type dispatch: unsupported array type "<<
                      l_field.schema().to_string());
    }

    return res;
}


//-----------------------------------------------------------------------------
//dispatch memory for a derived field (DF) unary operation
template<typename Function, typename Exec>
conduit::Node
dispatch_memory_unary_df(const conduit::Node &field,
                          const double &val,
                          std::string component,
                          const Function &func,
                          const Exec &exec)
{
    const std::string mem_space = Exec::memory_space;

    conduit::Node res;
    if(field_is_float32(field))
    {

        MCArray<conduit::float32> farray(field["values"]);
        DeviceAccessor<conduit::float32> accessor = farray.accessor(mem_space, component);
        res = func(accessor, val, exec);
    }
    else if(field_is_float64(field))
    {
        MCArray<conduit::float64> farray(field["values"]);
        DeviceAccessor<conduit::float64>  accessor = farray.accessor(mem_space, component);
        res = func(accessor, val, exec);
    }
    else if(field_is_int32(field))
    {
        MCArray<conduit::int32> farray(field["values"]);
        DeviceAccessor<conduit::int32>  accessor = farray.accessor(mem_space, component);
        res = func(accessor, val, exec);
    }
    else if(field_is_int64(field))
    {

        MCArray<conduit::int64> farray(field["values"]);
        DeviceAccessor<conduit::int64>  accessor = farray.accessor(mem_space, component);
        res = func(accessor, val, exec);
    }
    else
    {
        ASCENT_ERROR("Type dispatch: unsupported array type "<<
                      field.schema().to_string());
    }

    return res;
}

template<typename Function>
conduit::Node
exec_dispatch_binary_df(const conduit::Node &l_field,
                        const conduit::Node &r_field,
                        std::string component,
                        const Function &func)
{

  conduit::Node res;
  const std::string exec_policy = ExecutionManager::execution_policy();
  //std::cout<<"Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    res = dispatch_memory_binary_df(l_field, r_field, component, func, exec);
  }
#if defined(ASCENT_RAJA_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    res = dispatch_memory_binary_df(l_field, r_field, component, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    res = dispatch_memory_binary_df(l_field, r_field, component, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
    res = dispatch_memory_binary_df(l_field, r_field, component, func, exec);
  }
#endif
  else
  {
    ASCENT_ERROR("Execution dispatch: unsupported execution policy "<<
                  exec_policy);
  }
  return res;
}


template<typename Function>
conduit::Node
exec_dispatch_unary_df(const conduit::Node &field,
		       const double &val,
                       std::string component,
                       const Function &func)
{

  conduit::Node res;
  const std::string exec_policy = ExecutionManager::execution_policy();
  //std::cout<<"Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    res = dispatch_memory_unary_df(field, val, component, func, exec);
  }
#if defined(ASCENT_RAJA_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    res = dispatch_memory_unary_df(field, val, component, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    res = dispatch_memory_unary_df(field, val, component, func, exec);
  }
#endif
#if defined(ASCENT_RAJA_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
    res = dispatch_memory_unary_df(field, val, component, func, exec);
  }
#endif
  else
  {
    ASCENT_ERROR("Execution dispatch: unsupported execution policy "<<
                  exec_policy);
  }
  return res;
}

//-----------------------------------------------------------------------------
template<typename Function, typename T>
void
exec_dispatch_array(Array<T> &array, Function &func)
{
  const std::string exec_policy = ExecutionManager::execution_policy();

  //std::cout<<"Array Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    func(array, exec);
  }
#if defined(ASCENT_RAJA_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    func(array, exec);
  }
#endif
#if defined(ASCENT_RAJA_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    func(array, exec);
  }
#endif
#if defined(ASCENT_RAJA_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
    func(array, exec);
  }
#endif
  else
  {
    //TODO: log error this could hang things
    ASCENT_ERROR("exec_dispatch_array: unsupported execution policy "<<
                  exec_policy);
  }
}

//-----------------------------------------------------------------------------
template<typename Function>
void
exec_dispatch_function(Function &func)
{
  const std::string exec_policy = ExecutionManager::execution_policy();

  //std::cout<<"Exec only policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    func(exec);
  }
#if defined(ASCENT_RAJA_OPENMP_ENABLED)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    func(exec);
  }
#endif
#if defined(ASCENT_RAJA_CUDA_ENABLED)
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    func(exec);
  }
#endif
#if defined(ASCENT_RAJA_HIP_ENABLED)
  else if(exec_policy == "hip")
  {
    HipExec exec;
    func(exec);
  }
#endif
  else
  {
    //TODO: log error this could hang things
    ASCENT_ERROR("exec_dispatch_function: unsupported execution policy "<<
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
