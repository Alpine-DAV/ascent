#ifndef ASCENT_FIELD_ARRAY_HPP
#define ASCENT_FIELD_ARRAY_HPP

#include <conduit.hpp>
#include "ascent_raja_policies.hpp"
#include "ascent_memory_manager.hpp"
#include <ascent_logging.hpp>

namespace ascent
{

using index_t = conduit::index_t;

template<typename T>
inline bool is_conduit_type(const conduit::Node &values);

template<>
inline bool is_conduit_type<conduit::float64>(const conduit::Node &values)
{
  return values.dtype().is_float64();
}

template<>
inline bool is_conduit_type<conduit::float32>(const conduit::Node &values)
{
  return values.dtype().is_float32();
}

template<>
inline bool is_conduit_type<conduit::int32>(const conduit::Node &values)
{
  return values.dtype().is_int32();
}

template<>
inline bool is_conduit_type<conduit::int64>(const conduit::Node &values)
{
  return values.dtype().is_int64();
}

template<typename T>
inline T* conduit_ptr(conduit::Node &values);

template<>
inline conduit::float64 * conduit_ptr<conduit::float64>(conduit::Node &values)
{
  return values.as_float64_ptr();
}

template<>
inline conduit::float32 * conduit_ptr<conduit::float32>(conduit::Node &values)
{
  return values.as_float32_ptr();
}

template<>
inline conduit::int32 * conduit_ptr<conduit::int32>(conduit::Node &values)
{
  return values.as_int32_ptr();
}

template<>
inline conduit::int64 * conduit_ptr<conduit::int64>(conduit::Node &values)
{
  return values.as_int64_ptr();
}

// TODO: int64 index and size?
template<typename T>
struct ScalarAccess
{
  const T *m_values;
  const int m_size;
  ASCENT_EXEC
  T operator[](const int index)
  {
    return m_values[index];
  }
};

template<typename T>
struct ArrayAccess
{
  const T *m_values;
  const index_t m_size;
  const index_t m_offset;
  const index_t m_stride;

  ArrayAccess(const T *values, const index_t size, const index_t offset, const index_t stride)
    : m_values(values),
      m_size(size),
      m_offset(offset),
      m_stride(stride)
  {
  }

  ASCENT_EXEC
  T operator[](const index_t index)
  {
    return m_values[m_offset + m_stride * index];
  }
protected:
  ArrayAccess(){};
};

// rename to scalar array and don't support mcarrays
// TODO: if we allow non-const access then we need to
//       keep track of what is dirty
template<typename T>
class FieldArray
{
private:
  int m_components;
  conduit::Node &m_field;
  // path to manage memory
  const std::string m_path;
  std::vector<index_t> m_sizes;

public:
  // no default constructor
  FieldArray() = delete;

  FieldArray(const conduit::Node &field, const std::string path = "values")
    : m_field(const_cast<conduit::Node&>(field)),
      m_path(path)
  {
    if(!field.has_path(path))
    {
      ASCENT_ERROR("Array: does have path '"<<path<<"' "<<field.schema().to_yaml());
    }

    int children = m_field[m_path].number_of_children();
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
      types_match = is_conduit_type<T>(m_field[m_path]);
      m_sizes[0] = m_field[m_path].dtype().number_of_elements();
    }
    else
    {
      for(int i = 0; i < children; ++i)
      {
        types_match &= is_conduit_type<T>(m_field[m_path].child(i));
        m_sizes[i] = m_field[m_path].child(i).dtype().number_of_elements();
      }
    }

    if(!types_match)
    {
      std::string schema = m_field.schema().to_yaml();
      ASCENT_ERROR("Field type does not match conduit type: "<<schema);
    }
  }

  index_t size(int component) const
  {
    return m_sizes[component];
  }

  int components() const
  {
    return m_components;
  }

  T value(const index_t idx, const std::string component)
  {
    int comp_idx = resolve_component(component);
    return value(idx, comp_idx);
  }

  T value(const index_t idx, int component)
  {
    std::string path;
    const T * ptr = raw_ptr(component,path);
    index_t el_idx = m_field[path].dtype().element_index(idx);
    T val;
#ifdef ASCENT_USE_CUDA
    if(is_gpu_ptr(ptr))
    {
      cudaMemcpy (&val, &ptr[el_idx], sizeof (T), cudaMemcpyDeviceToHost);
    }
    else
    {
      val = ptr[el_idx];
    }
#else
    val = ptr[el_idx];
#endif
    return val;
  }

  // get the raw pointer used by conduit and return the path
  const T *raw_ptr(int component, std::string &leaf_path)
  {
    if(component < 0 || component >= m_components)
    {
      ASCENT_ERROR("Invalid component "<<component<<" number of components "<<m_components);
    }

    const int children = m_field[m_path].number_of_children();

    leaf_path = m_path;
    if(children > 0)
    {
      leaf_path = m_path + "/" + m_field[m_path].child(component).name();
    }

    const T * ptr = conduit_ptr<T>(m_field[leaf_path]);
    return ptr;
  }

  int resolve_component(const std::string component)
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
      const int children = m_field["values"].number_of_children();
      bool match = false;
      for(int i = 0; i < children; ++i)
      {
        if(component == m_field["values"].child(i).name())
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

  const T *device_ptr_const(const std::string component)
  {
  }

  const T *device_ptr_const(int component)
  {

    std::string leaf_path;
    const T * ptr = raw_ptr(component,leaf_path);


#ifdef ASCENT_USE_CUDA
    if(is_gpu_ptr(ptr))
    {
      std::cout<<"already a gpu pointer\n";
      return ptr;
    }
    else
    {
      std::string d_path = "device_"+leaf_path;
      std::cout<<"leaf_path '"<<leaf_path<<"' device _path '"<<d_path<<"'\n";
      std::cout<<"size "<<m_sizes[component]<<"\n";
      if(!m_field.has_path(d_path))
      {
        std::cout<<"Creating device pointer\n";
        conduit::Node &n_device = m_field[d_path];
        n_device.set_allocator(AllocationManager::conduit_device_allocator_id());
        std::cout<<"setting...\n";
        n_device.set(ptr, m_sizes[component]);
        std::cout<<"set\n";
      }
      else std::cout<<"already device_values\n";
      return conduit_ptr<T>(m_field[d_path]);
    }
#else
    std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }

  const T *host_ptr_const(int component)
  {
    std::string leaf_path;
    const T * ptr = raw_ptr(component,leaf_path);

#ifdef ASCENT_USE_CUDA
    if(!is_gpu_ptr(ptr))
    {
      std::cout<<"already a host pointer\n";
      return ptr;
    }
    else
    {
      std::string h_path = "host_" + leaf_path;
      if(!m_field.has_path(h_path))
      {
        std::cout<<"Creating host pointer\n";
        conduit::Node &n_host = m_field[h_path];
        n_host.set_allocator(AllocationManager::conduit_host_allocator_id());
        n_host.set(ptr, m_sizes[component]);
      }
      else std::cout<<"already host_values\n";
      return conduit_ptr<T>(m_field[h_path]);
    }
#else
    std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }

  // can make string param that specifies the device
  // TODO: this will not be the way to access ptrs going forward
  const T *ptr_const(const std::string location = "device")
  {
    if(location != "host" && location != "device")
    {
      ASCENT_ERROR("Invalid location: '"<<location<<"'");
    }

    std::cout<<"SDKF:SDKFH\n";

    if(location == "device")
    {
      return device_ptr_const(0);
    }
    else
    {
      return host_ptr_const(0);
    }
  }

};


} // namespace ascent
#endif
