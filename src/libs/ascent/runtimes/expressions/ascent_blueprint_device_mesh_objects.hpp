//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef ASCENT_BLUEPRINT_DEVICE_MESH_OBJECTS_HPP
#define ASCENT_BLUEPRINT_DEVICE_MESH_OBJECTS_HPP

#include <conduit.hpp>
#include "ascent_execution_policies.hpp"
#include <ascent_logging.hpp>
#include "ascent_memory_manager.hpp"
#include "ascent_blueprint_device_mesh_objects.hpp"
#include "ascent_array.hpp"
#include "ascent_execution_policies.hpp"

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


using index_t = conduit::index_t;

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// begin is_conduit_type
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
template<typename T>
inline bool is_conduit_type(const conduit::Node &values);

//---------------------------------------------------------------------------//
template<>
inline bool is_conduit_type<conduit::float64>(const conduit::Node &values)
{
  return values.dtype().is_float64();
}

//---------------------------------------------------------------------------//
template<>
inline bool is_conduit_type<conduit::float32>(const conduit::Node &values)
{
  return values.dtype().is_float32();
}

//---------------------------------------------------------------------------//
template<>
inline bool is_conduit_type<conduit::int32>(const conduit::Node &values)
{
  return values.dtype().is_int32();
}

//---------------------------------------------------------------------------//
template<>
inline bool is_conduit_type<conduit::int64>(const conduit::Node &values)
{
  return values.dtype().is_int64();
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// end is_conduit_type
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// begin conduit_ptr
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
template<typename T>
inline T* conduit_ptr(const conduit::Node &values);

//---------------------------------------------------------------------------//
template<>
inline conduit::float64 * conduit_ptr<conduit::float64>(const conduit::Node &values)
{
  return const_cast<conduit::float64*>(values.as_float64_ptr());
}

//---------------------------------------------------------------------------//
template<>
inline conduit::float32 * conduit_ptr<conduit::float32>(const conduit::Node &values)
{
  return const_cast<conduit::float32*>(values.as_float32_ptr());
}

//---------------------------------------------------------------------------//
template<>
inline conduit::int32 * conduit_ptr<conduit::int32>(const conduit::Node &values)
{
  return const_cast<conduit::int32*>(values.as_int32_ptr());
}
//---------------------------------------------------------------------------//
template<>
inline conduit::int64 * conduit_ptr<conduit::int64>(const conduit::Node &values)
{
  return const_cast<conduit::int64*>(values.as_int64_ptr());
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// end conduit_ptr
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
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


//---------------------------------------------------------------------------//
// Vec
//---------------------------------------------------------------------------//
template<typename T, int S>
class Vec
{
public:
  T m_data[S];

  ASCENT_EXEC const T &operator[] (const int &i) const
  {
    return m_data[i];
  }

  ASCENT_EXEC T &operator[] (const int &i)
  {
    return m_data[i];
  }
};

//---------------------------------------------------------------------------//
// DeviceAccessor
//---------------------------------------------------------------------------//
template<typename T>
struct DeviceAccessor
{
  const T *m_values;
  const index_t m_size;
  const index_t m_offset;
  const index_t m_stride;

  //==---------------------------------------------------------------------==//
  DeviceAccessor(const T *values, const conduit::DataType &dtype)
    : m_values(values),
      m_size(dtype.number_of_elements()),
      // conduit strides and offsets are in terms of bytes
      m_offset(dtype.offset() / sizeof(T)),
      m_stride(dtype.stride() / sizeof(T))
  {

  }

  //==---------------------------------------------------------------------==//
  ASCENT_EXEC
  T operator[](const index_t index) const
  {
    return m_values[m_offset + m_stride * index];
  }
protected:
  DeviceAccessor(){};
};

//---------------------------------------------------------------------------//
// MCArray
//---------------------------------------------------------------------------//
//-- Note: This supports both MCArrays and Leaf Arrays!
//---------------------------------------------------------------------------//
template<typename T>
class MCArray
{
private:
  // src node
  const conduit::Node  &m_src_node;
  // used if we need to make any host or device copies
  conduit::Node        m_tmps;
   // num comps
  int                  m_components;
  // sizes for each comp
  std::vector<index_t> m_sizes;

public:
  //==---------------------------------------------------------------------==//
  // No default constructor
  MCArray() = delete;

  //==---------------------------------------------------------------------==//
  // Main constructor
  MCArray(const conduit::Node &node)
    : m_src_node(node)
  {
    int children = m_src_node.number_of_children();
    if(children == 0 || children == 1)
    {
      m_components = 1;
    }
    else
    {
      m_components = children;
    }

    m_sizes.resize(m_components);

    bool types_match = true;
    if(children == 0)
    {
      types_match = is_conduit_type<T>(m_src_node);
      m_sizes[0] = m_src_node.dtype().number_of_elements();
    }
    else
    {
      for(int i = 0; i < children; ++i)
      {
        types_match &= is_conduit_type<T>(m_src_node.child(i));
        m_sizes[i] = m_src_node.child(i).dtype().number_of_elements();
      }
    }

    if(!types_match)
    {
      std::string schema = m_src_node.schema().to_yaml();
      ASCENT_ERROR("MCArray type does not match conduit type: "<<schema);
    }
  }

  //==---------------------------------------------------------------------==//
  index_t size(int component) const
  {
    return m_sizes[component];
  }

  //==---------------------------------------------------------------------==//
  int components() const
  {
    return m_components;
  }

  //==---------------------------------------------------------------------==//
  T value(const index_t idx, const std::string &component)
  {
    int comp_idx = resolve_component(component);
    // TODO: log error component and index
    //if(idx < 0 || idx >= m_sizes[comp_idx])
    //{
    //  std::cout<<"[MCarray] Invalid index "<<idx<<" size "<<m_sizes[comp_idx]<<"\n";
    //}
    return value(idx, comp_idx);
  }

  //==---------------------------------------------------------------------==//
  T value(const index_t idx, int component)
  {
    // TODO: log error component and index
    //if(idx < 0 || idx >= m_sizes[component])
    //{
    //  std::cout<<"[MCarray] Invalid index "<<idx<<" size "<<m_sizes[component]<<"\n";
    //}
    std::string path;
    const T * ptr = raw_ptr(component,path);
    // elemen_index is extremely missleading. Its actually the byte offset to
    // the element. 2 hours to learn this
    T val;
    index_t el_idx = 0;
    if(path != "")
    {
        el_idx = m_src_node[path].dtype().element_index(idx) / sizeof(T);
    }
    else
    {
        el_idx = m_src_node.dtype().element_index(idx) / sizeof(T);
    }

    if(DeviceMemory::is_device_ptr(ptr))
    {
      #if defined(ASCENT_CUDA_ENABLED)
          cudaMemcpy (&val, &ptr[el_idx], sizeof (T), cudaMemcpyDeviceToHost);
      #elif defined(ASCENT_HIP_ENABLED)
          hipMemcpy (&val, &ptr[el_idx], sizeof (T), hipMemcpyDeviceToHost);
      #endif
    }
    else
    {
      val = ptr[el_idx];
    }
    return val;
  }

  //==---------------------------------------------------------------------==//
  std::string component_path(int component)
  {
    std::string leaf_path;
    if(component < 0 || component >= m_components)
    {
      ASCENT_ERROR("Invalid component "<<component<<" number of components "<<m_components);
    }

    const int children = m_src_node.number_of_children();

    leaf_path = "";
    if(children > 0)
    {
      leaf_path = m_src_node.child(component).name();
    }
    return leaf_path;
  }

  //==---------------------------------------------------------------------==//
  // get the raw pointer used by conduit and return the path
  const T *raw_ptr(int component, std::string &leaf_path)
  {
    leaf_path = component_path(component);
    if(leaf_path != "")
    {
        return conduit_ptr<T>(m_src_node[leaf_path]);
    }
    else
    {
        return conduit_ptr<T>(m_src_node);
    }
  }

  //==---------------------------------------------------------------------==//
  int resolve_component(const std::string &component)
  {
    int component_idx = 0;

    // im going to allow blank names for component 0 since
    // an mcarray is ambiguous with a single component
    if(m_components == 1 && component == "")
    {
      return component_idx;
    }
    else
    {
      const int children = m_src_node.number_of_children();
      bool match = false;
      for(int i = 0; i < children; ++i)
      {
        if(component == m_src_node.child(i).name())
        {
          component_idx = i;
          match = true;
          break;
        }
      }
      if(!match)
      {
        ASCENT_ERROR("No component named '"<<component<<"'");
      }
    }
    return component_idx;
  }

  //==---------------------------------------------------------------------==//
  const T *device_ptr_const(int component)
  {
    std::string leaf_path;
    const T * ptr = raw_ptr(component,leaf_path);

#if defined(ASCENT_DEVICE_ENABLED)
    if(DeviceMemory::is_device_ptr(ptr))
    {
      //std::cout<<"already a gpu pointer\n";
      return ptr;
    }
    else
    {
      std::string d_path = "device";
      if(leaf_path != "")
      {
          d_path += "_" + leaf_path;
      }

      //std::cout<<"leaf_path '"<<leaf_path<<"' device _path '"<<d_path<<"'\n";
      //std::cout<<"size "<<m_sizes[component]<<"\n";
      if(!m_tmps.has_path(d_path))
      {
        //std::cout<<"Creating device pointer\n";
        conduit::Node &n_device = m_tmps[d_path];
        n_device.set_allocator(AllocationManager::conduit_device_allocator_id());
        //std::cout<<"setting...\n";
        n_device.set(ptr, m_sizes[component]);
        //std::cout<<"set\n";
      }
      //else std::cout<<"already device_values\n";
      return conduit_ptr<T>(m_tmps[d_path]);
    }
#else
    //std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }

  //==---------------------------------------------------------------------==//
  const T *host_ptr_const(int component)
  {
    std::string leaf_path;
    const T * ptr = raw_ptr(component,leaf_path);

#if defined(ASCENT_DEVICE_ENABLED)
    bool is_unified;
    bool is_gpu;
    DeviceMemory::is_device_ptr(ptr,is_gpu, is_unified);
    bool is_host_accessible =  !is_gpu || (is_gpu && is_unified);
    
    if(is_unified)
    {
      //std::cout<<"Unified\n";
    }

    if(is_gpu)
    {
      //std::cout<<"gpu\n";
    }

    if(is_host_accessible)
    {
      //std::cout<<"already a host pointer\n";
      return ptr;
    }
    else
    {
      std::string h_path = "host";
      if(leaf_path != "")
      {
          h_path += "_" + leaf_path;
      }

      if(!m_tmps.has_path(h_path))
      {
        //std::cout<<"Creating host pointer\n";
        conduit::Node &n_host = m_tmps[h_path];
        n_host.set_allocator(AllocationManager::conduit_host_allocator_id());
        n_host.set(ptr, m_sizes[component]);
      }
      //else std::cout<<"already host_values\n";
      return conduit_ptr<T>(m_tmps[h_path]);
    }
#else
    //std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }

  //==---------------------------------------------------------------------==//
  // can make string param that specifies the device
  // TODO: this will not be the way to access ptrs going forward
  const T *ptr_const(const std::string &location = "device")
  {
    if(location != "host" && location != "device")
    {
      ASCENT_ERROR("Invalid location: '"<<location<<"'");
    }

    std::string leaf_path;
    if(location == "device")
    {
      return device_ptr_const(0);
    }
    else
    {
      return host_ptr_const(0);
    }
  }

  //==---------------------------------------------------------------------==//
  DeviceAccessor<T> accessor(const std::string location = "device",
                             const std::string comp = "")
  {
    if(location != "device" && location != "host")
    {
      ASCENT_ERROR("Bad location string '"<<location<<"'");
    }
    const int children = m_src_node.number_of_children();
    int comp_idx = 0;
    if(comp == "")
    {
      if(m_components != 1)
      {
        ASCENT_ERROR("Ambiguous component: node has more than one component but no"<<
                     " component was specified");
      }
    }
    else
    {
      comp_idx = resolve_component(comp);
    }

    std::string leaf_path = component_path(comp_idx);
    //std::cout<<"giving out access "<<leaf_path<<" "<<location<<"\n";

    const T* ptr = location == "device" ? device_ptr_const(comp_idx) : host_ptr_const(comp_idx);
    if(leaf_path != "")
    {
        return DeviceAccessor<T>(ptr, m_src_node[leaf_path].dtype());
    }
    else
    {
        return DeviceAccessor<T>(ptr, m_src_node.dtype());
    }
  }

};

//---------------------------------------------------------------------------//
ASCENT_EXEC
void
mesh_logical_index_2d(int *idx, const int vert_index, const int *dims)
{
  idx[0] = vert_index % dims[0];
  idx[1] = vert_index / dims[0];
}

//---------------------------------------------------------------------------//
ASCENT_EXEC
void
mesh_logical_index_3d(int *idx, const int vert_index, const int *dims)
{
  idx[0] = vert_index % dims[0];
  idx[1] = (vert_index / dims[0]) % dims[1];
  idx[2] = vert_index / (dims[0] * dims[1]);
}

//---------------------------------------------------------------------------//
ASCENT_EXEC
void structured_cell_indices(const int cell_index,
                             const Vec<int,3> &point_dims,
                             const int dims, // 2d or 3d
                             int indices[8])
{
  const int element_dims[3] = {point_dims[0]-1,
                               point_dims[1]-1,
                               point_dims[2]-1};
  int element_index[3];
  if(dims == 2)
  {
    mesh_logical_index_2d(element_index, cell_index, element_dims);

    indices[0] = element_index[1] * point_dims[0] + element_index[0];
    indices[1] = indices[0] + 1;
    indices[2] = indices[1] + point_dims[0];
    indices[3] = indices[2] - 1;
  }
  else
  {
    mesh_logical_index_3d(element_index, cell_index, element_dims);

    indices[0] =
        (element_index[2] * point_dims[1] + element_index[1])
         * point_dims[0] + element_index[0];
    indices[1] = indices[0] + 1;
    indices[2] = indices[1] + point_dims[1];
    indices[3] = indices[2] - 1;
    indices[4] = indices[0] + point_dims[0] * point_dims[2];
    indices[5] = indices[4] + 1;
    indices[6] = indices[5] + point_dims[1];
    indices[7] = indices[6] - 1;
  }
}

//---------------------------------------------------------------------------//
// UniformMesh
//---------------------------------------------------------------------------//
struct UniformMesh
{
  Vec<int,3> m_point_dims;
  Vec<double,3> m_origin;
  Vec<double,3> m_spacing;
  int m_dims;
  int m_num_cells;
  int m_num_indices;
  int m_num_points;

  //==---------------------------------------------------------------------==//
  UniformMesh() = delete;

  //==---------------------------------------------------------------------==//
  UniformMesh(const conduit::Node &n_coords)
  {
    const conduit::Node &n_dims = n_coords["dims"];
    // assume we have a valid dataset
    m_point_dims[0] = n_dims["i"].to_int();
    m_point_dims[1] = n_dims["j"].to_int();
    m_point_dims[2] = 1;
    m_dims = 2;
    // check for 3d
    if(n_dims.has_path("k"))
    {
      m_point_dims[2] = n_dims["k"].to_int();
      m_dims = 3;
    }

    m_origin[0] = 0.0;
    m_origin[1] = 0.0;
    m_origin[2] = 0.0;

    m_spacing[0] = 1.0;
    m_spacing[1] = 1.0;
    m_spacing[2] = 1.0;

    if(n_coords.has_child("origin"))
    {
      const conduit::Node &n_origin = n_coords["origin"];

      if(n_origin.has_child("x"))
      {
        m_origin[0] = n_origin["x"].to_float64();
      }

      if(n_origin.has_child("y"))
      {
        m_origin[1] = n_origin["y"].to_float64();
      }

      if(n_origin.has_child("z"))
      {
        m_origin[2] = n_origin["z"].to_float64();
      }
    }

    if(n_coords.has_path("spacing"))
    {
      const conduit::Node &n_spacing = n_coords["spacing"];

      if(n_spacing.has_path("dx"))
      {
        m_spacing[0] = n_spacing["dx"].to_float64();
      }

      if(n_spacing.has_path("dy"))
      {
        m_spacing[1] = n_spacing["dy"].to_float64();
      }

      if(n_spacing.has_path("dz"))
      {
        m_spacing[2] = n_spacing["dz"].to_float64();
      }
    }

    m_num_points = m_point_dims[0] * m_point_dims[1];
    m_num_cells = (m_point_dims[0] - 1) *(m_point_dims[1] - 1);
    if(m_dims == 3)
    {
      m_num_cells *= m_point_dims[2] - 1;
      m_num_points = m_point_dims[2];
    }

    if(m_dims == 3)
    {
      m_num_indices = 8;
    }
    else
    {
      m_num_indices = 4;
    }


  }

  //==---------------------------------------------------------------------==//
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    structured_cell_indices(cell_index,
                            m_point_dims,
                            m_dims,
                            indices);
  }

  //==---------------------------------------------------------------------==//
  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    int logical_idx[3];
    int pdims[3] = {m_point_dims[0],
                    m_point_dims[1],
                    m_point_dims[2]};
    if(m_dims == 2)
    {
      mesh_logical_index_2d(logical_idx, vert_id, pdims);
    }
    else
    {
      mesh_logical_index_3d(logical_idx, vert_id, pdims);
    }

    vert[0] = m_origin[0] + logical_idx[0] * m_spacing[0];
    vert[1] = m_origin[1] + logical_idx[1] * m_spacing[1];
    if(m_dims == 3)
    {
      vert[2] = m_origin[2] + logical_idx[2] * m_spacing[2];
    }
    else
    {
      vert[2] = 0.;
    }

  }

};

//---------------------------------------------------------------------------//
// StructuredMesh
//---------------------------------------------------------------------------//
template<typename CoordsType>
struct StructuredMesh
{

  DeviceAccessor<CoordsType> m_coords_x;
  DeviceAccessor<CoordsType> m_coords_y;
  DeviceAccessor<CoordsType> m_coords_z;
  const int m_dims;
  const Vec<int,3> m_point_dims;
  const int m_num_indices;
  const int m_num_cells;
  const int m_num_points;

  //==---------------------------------------------------------------------==//
  StructuredMesh() = delete;

  //==---------------------------------------------------------------------==//
  StructuredMesh(const std::string mem_space,
                 MCArray<CoordsType> &coords,
                 const int dims,
                 const int point_dims[3])
    : m_coords_x(coords.accessor(mem_space, "x")),
      m_coords_y(coords.accessor(mem_space, "y")),
      m_coords_z(dims == 3 ? coords.accessor(mem_space, "z") :
                             // just use a dummy in this case
                             coords.accessor(mem_space, "x")),
      m_dims(dims),
      m_point_dims({{point_dims[0],
                     point_dims[1],
                     point_dims[2]}}),
      m_num_indices(m_dims == 2 ? 4 : 8),
      m_num_cells(m_dims == 2 ? (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1)
                              : (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1) *
                                (m_point_dims[2] - 1)),
      m_num_points(m_dims == 2 ? (m_point_dims[0]) *
                                 (m_point_dims[1])
                               : (m_point_dims[0]) *
                                 (m_point_dims[1]) *
                                 (m_point_dims[2]))
  {
  }

  //==---------------------------------------------------------------------==//
  // TODO: some sort of error checking mechanism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    structured_cell_indices(cell_index,
                            m_point_dims,
                            m_dims,
                            indices);
  }

  //==---------------------------------------------------------------------==//
  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    vert[0] = static_cast<double>(m_coords_x[vert_id]);
    vert[1] = static_cast<double>(m_coords_y[vert_id]);
    if(m_dims == 3)
    {
      vert[2] = static_cast<double>(m_coords_z[vert_id]);
    }
    else
    {
      vert[2] = 0.;
    }
  }

};

//---------------------------------------------------------------------------//
// RectilinearMesh
//---------------------------------------------------------------------------//
template<typename CoordsType>
struct RectilinearMesh
{

  DeviceAccessor<CoordsType> m_coords_x;
  DeviceAccessor<CoordsType> m_coords_y;
  DeviceAccessor<CoordsType> m_coords_z;
  const int m_dims;
  const Vec<int,3> m_point_dims;
  const int m_num_indices;
  const int m_num_cells;
  const int m_num_points;

  //==---------------------------------------------------------------------==//
  RectilinearMesh() = delete;

  //==---------------------------------------------------------------------==//
  RectilinearMesh(const std::string mem_space,
                  MCArray<CoordsType> &x_coords,
                  MCArray<CoordsType> &y_coords,
                  MCArray<CoordsType> &z_coords,
                  const int dims)
    : m_coords_x(x_coords.accessor(mem_space)),
      m_coords_y(x_coords.accessor(mem_space)),
      m_coords_z(z_coords.accessor(mem_space)),
      // m_coords_z(dims == 3 ? z_coords.accessor(mem_space) :
      //                        // just use a dummy in this case
      //                        coords.accessor(mem_space)),
      m_dims(dims),
      m_point_dims({{(int)m_coords_x.m_size,
                     (int)m_coords_y.m_size,
                     (int)m_coords_z.m_size}}),
      m_num_indices(m_dims == 2 ? 4 : 8),
      m_num_cells(m_dims == 2 ? (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1)
                              : (m_point_dims[0] - 1) *
                                (m_point_dims[1] - 1) *
                                (m_point_dims[2] - 1)),
      m_num_points(m_dims == 2 ? (m_point_dims[0]) *
                                 (m_point_dims[1])
                               : (m_point_dims[0]) *
                                 (m_point_dims[1]) *
                                 (m_point_dims[2]))
  {
  }

  //==---------------------------------------------------------------------==//
  // TODO: some sort of error checking mechanism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    structured_cell_indices(cell_index,
                            m_point_dims,
                            m_dims,
                            indices);
  }

  //==---------------------------------------------------------------------==//
  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    int logical_idx[3];
    int pdims[3] = {m_point_dims[0],
                    m_point_dims[1],
                    m_point_dims[2]};
    if(m_dims == 2)
    {
      mesh_logical_index_2d(logical_idx, vert_id, pdims);
    }
    else
    {
      mesh_logical_index_3d(logical_idx, vert_id, pdims);
    }

    vert[0] = static_cast<double>(m_coords_x[logical_idx[0]]);
    vert[1] = static_cast<double>(m_coords_y[logical_idx[1]]);
    if(m_dims == 3)
    {
      vert[2] = static_cast<double>(m_coords_z[logical_idx[2]]);
    }
    else
    {
      vert[2] = 0.;
    }
  }

};

//---------------------------------------------------------------------------//
// UnstructuredMesh
//---------------------------------------------------------------------------//
template<typename CoordsType, typename ConnType>
struct UnstructuredMesh
{

  DeviceAccessor<CoordsType> m_coords_x;
  DeviceAccessor<CoordsType> m_coords_y;
  DeviceAccessor<CoordsType> m_coords_z;
  DeviceAccessor<ConnType> m_conn;
  const int m_shape_type;
  const int m_dims;
  const int m_num_indices;
  const int m_num_cells;
  const int m_num_points;

  //==---------------------------------------------------------------------==//
  UnstructuredMesh() = delete;

  //==---------------------------------------------------------------------==//
  UnstructuredMesh(const std::string mem_space,
                   MCArray<CoordsType> &coords,
                   MCArray<ConnType> &conn,
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
      m_num_cells(m_conn.m_size / m_num_indices),
      m_num_points(m_coords_x.m_size)

  {
  }

  //==---------------------------------------------------------------------==//
  // TODO: some sort of error checking mechanism
  ASCENT_EXEC
  void cell_indices(const int cell_index, int indices[8]) const
  {
    const int offset = cell_index * m_num_indices;
    for(int i = 0; i < m_num_indices; ++i)
    {
      indices[i] = static_cast<int>(m_conn[offset + i]);
    }
  }

  //==---------------------------------------------------------------------==//
  ASCENT_EXEC
  void vertex(const int vert_id, double vert[3]) const
  {
    vert[0] = static_cast<double>(m_coords_x[vert_id]);
    vert[1] = static_cast<double>(m_coords_y[vert_id]);
    if(m_dims == 3)
    {
      vert[2] = static_cast<double>(m_coords_z[vert_id]);
    }
    else
    {
      vert[2] = 0.;
    }
  }

};



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
