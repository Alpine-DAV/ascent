#ifndef ASCENT_FIELD_ARRAY_HPP
#define ASCENT_FIELD_ARRAY_HPP

#include <conduit.hpp>
#include "ascent_memory_manager.hpp"
#include <ascent_logging.hpp>

namespace ascent
{

template<typename T>
bool is_conduit_type(const conduit::Node &values);

template<>
bool is_conduit_type<conduit::float64>(const conduit::Node &values)
{
  return values.dtype().is_float64();
}

template<>
bool is_conduit_type<conduit::float32>(const conduit::Node &values)
{
  return values.dtype().is_float32();
}

template<>
bool is_conduit_type<conduit::int32>(const conduit::Node &values)
{
  return values.dtype().is_int32();
}

template<typename T>
T* conduit_ptr(conduit::Node &values);

template<>
conduit::float64 * conduit_ptr<conduit::float64>(conduit::Node &values)
{
  return values.as_float64_ptr();
}

template<>
conduit::float32 * conduit_ptr<conduit::float32>(conduit::Node &values)
{
  return values.as_float32_ptr();
}

template<>
conduit::int32 * conduit_ptr<conduit::int32>(conduit::Node &values)
{
  return values.as_int32_ptr();
}

template<>
conduit::int64 * conduit_ptr<conduit::int64>(conduit::Node &values)
{
  return values.as_int64_ptr();
}

template<typename T>
class FieldArray
{
private:
  int m_components;
  conduit::Node &m_field;
public:
  FieldArray(const conduit::Node &field)
    : m_field(const_cast<conduit::Node&>(field))
  {
    int children = m_field["values"].number_of_children();
    if(children == 0 || children == 1)
    {
      m_components = 1;
    }
    else
    {
      m_components = children;
    }

    bool types_match = true;
    if(children == 0)
    {
      types_match = is_conduit_type<T>(m_field["values"]);
    }
    else
    {
      for(int i = 0; i < children; ++i)
      {
        types_match &= is_conduit_type<T>(m_field["values"].child(i));
      }
    }

    if(!types_match)
    {
      ASCENT_ERROR("Field type does not match conduit type");
    }
  }

  conduit::index_t size() const
  {
    // NO
    return m_field["values"].dtype().number_of_elements();
  }

  // can make string param that specifies the device
  const T *ptr_const()
  {
    std::cout<<"SDKF:SDKFH\n";
    // This is not complete at all
    // assuming that children == 0
    const T * ptr = conduit_ptr<T>(m_field["values"]);
    const conduit::index_t size  = m_field["values"].dtype().number_of_elements();
#ifdef ASCENT_USE_CUDA
    if(is_gpu_ptr(ptr))
    {
      std::cout<<"already a gpu pointer\n";
      return ptr;
    }
    else
    {
      if(!m_field.has_path("device_values"))
      {
        std::cout<<"Creating device pointer\n";
        conduit::Node &n_device = m_field["deivce_values"];
        n_device.set_allocator(AllocationManager::conduit_device_allocator_id());
        n_device.set(ptr, size);
      }
      else std::cout<<"already device_values\n";
      return conduit_ptr<T>(m_field["device_values"]);
    }
#else
    std::cout<<"just returning ptr\n";
    return ptr;
#endif
  }
};

} // namespace ascent
#endif
