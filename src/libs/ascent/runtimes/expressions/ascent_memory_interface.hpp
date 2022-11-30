#ifndef ASCENT_MEMORY_INTERFACE_HPP
#define ASCENT_MEMORY_INTERFACE_HPP

#include <conduit.hpp>
#include "ascent_execution_policies.hpp"
#include "ascent_memory_manager.hpp"
#include <ascent_logging.hpp>

namespace ascent
{

// TODO conduit::index_t ?
using index_t = conduit::int64;

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

template<typename T>
struct MemoryAccessor
{
  const T *m_values;
  const index_t m_size;
  const index_t m_offset;
  const index_t m_stride;

  MemoryAccessor(const T *values, const conduit::DataType &dtype)
    : m_values(values),
      m_size(dtype.number_of_elements()),
      // conduit strides and offsets are in terms of bytes
      m_offset(dtype.offset() / sizeof(T)),
      m_stride(dtype.stride() / sizeof(T))
  {
  }

  ASCENT_EXEC
  T operator[](const index_t index) const
  {
    return m_values[m_offset + m_stride * index];
  }
protected:
  MemoryAccessor(){};
};

// rename to scalar array and don't support mcarrays
// TODO: if we allow non-const access then we need to
//       keep track of what is dirty
template<typename T>
class MemoryInterface
{
private:
  int m_components;
  conduit::Node &m_field;
  // path to manage memory
  const std::string m_path;
  std::vector<index_t> m_sizes;

public:
  // no default constructor
  MemoryInterface() = delete;

  MemoryInterface(const conduit::Node &field, const std::string &path = "values")
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

  T value(const index_t idx, const std::string &component)
  {
    int comp_idx = resolve_component(component);
    // TODO: log error component and index
    //if(idx < 0 || idx >= m_sizes[comp_idx])
    //{
    //  std::cout<<"[Memory Interface] Invalid index "<<idx<<" size "<<m_sizes[comp_idx]<<"\n";
    //}
    return value(idx, comp_idx);
  }

  T value(const index_t idx, int component)
  {
    // TODO: log error component and index
    //if(idx < 0 || idx >= m_sizes[component])
    //{
    //  std::cout<<"[Memory Interface] Invalid index "<<idx<<" size "<<m_sizes[component]<<"\n";
    //}
    std::string path;
    const T * ptr = raw_ptr(component,path);
    // elemen_index is extremely missleading. Its actually the byte offset to
    // the element. 2 hours to learn this
    index_t el_idx = m_field[path].dtype().element_index(idx) / sizeof(T);
    T val;

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

  std::string component_path(int component)
  {
    std::string leaf_path;
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
    return leaf_path;
  }

  // get the raw pointer used by conduit and return the path
  const T *raw_ptr(int component, std::string &leaf_path)
  {
    leaf_path = component_path(component);
    const T * ptr = conduit_ptr<T>(m_field[leaf_path]);
    return ptr;
  }

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
      const int children = m_field[m_path].number_of_children();
      bool match = false;
      for(int i = 0; i < children; ++i)
      {
        if(component == m_field[m_path].child(i).name())
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
      std::string d_path = "device_"+leaf_path;

      //std::cout<<"leaf_path '"<<leaf_path<<"' device _path '"<<d_path<<"'\n";
      //std::cout<<"size "<<m_sizes[component]<<"\n";
      if(!m_field.has_path(d_path))
      {
        //std::cout<<"Creating device pointer\n";
        conduit::Node &n_device = m_field[d_path];
        n_device.set_allocator(AllocationManager::conduit_device_allocator_id());
        //std::cout<<"setting...\n";
        n_device.set(ptr, m_sizes[component]);
        //std::cout<<"set\n";
      }
      //else std::cout<<"already device_values\n";
      return conduit_ptr<T>(m_field[d_path]);
    }
#else
    //std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }

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
      std::string h_path = "host_" + leaf_path;
      if(!m_field.has_path(h_path))
      {
        //std::cout<<"Creating host pointer\n";
        conduit::Node &n_host = m_field[h_path];
        n_host.set_allocator(AllocationManager::conduit_host_allocator_id());
        n_host.set(ptr, m_sizes[component]);
      }
      //else std::cout<<"already host_values\n";
      return conduit_ptr<T>(m_field[h_path]);
    }
#else
    //std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }

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

  MemoryAccessor<T> accessor(const std::string location = "device", const std::string comp = "")
  {
    if(location != "device" && location != "host")
    {
      ASCENT_ERROR("Bad location string '"<<location<<"'");
    }
    const int children = m_field[m_path].number_of_children();
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
    return MemoryAccessor<T>(ptr, m_field[leaf_path].dtype());
  }

};


} // namespace ascent
#endif
