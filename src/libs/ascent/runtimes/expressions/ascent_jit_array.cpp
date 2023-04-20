//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_jit_array.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_jit_array.hpp"
#include <ascent_logging.hpp>

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

namespace detail
{

std::vector<std::string>
component_names(const int num_components)
{
  std::vector<std::string> component_names{};
  if(num_components > 1)
  {
    for(int i = 0; i < num_components; ++i)
    {
      component_names.push_back(std::string(1, 'x' + i));
    }
  }
  return component_names;
}

} // namespace detail

std::string
ArrayCode::index(const std::string &idx,
                 const std::string name,
                 const std::ptrdiff_t offset,
                 const std::ptrdiff_t stride,
                 const size_t pointer_size) const
{
  if(offset % pointer_size != 0)
  {
    ASCENT_ERROR(
        "StridedArray: The offset of "
        << offset
        << " bytes does not divide into the array's pointer size which is "
        << pointer_size << " bytes.");
  }
  if(stride % pointer_size != 0)
  {
    ASCENT_ERROR(
        "StridedArray: The stride of "
        << stride
        << " bytes does not divide into the array's pointer size which is "
        << pointer_size << " bytes.");
  }

  const int pointer_offset = offset / pointer_size;
  const int pointer_stride = stride / pointer_size;
  // don't want to clutter the code with 0 + 1 * x
  const std::string offset_str =
      (pointer_offset == 0 ? "" : std::to_string(pointer_offset) + " + ");
  const std::string stride_str =
      (pointer_stride == 1 ? "" : std::to_string(pointer_stride) + " * ");
  return name + "[" + offset_str + stride_str + idx + "]";
}

std::string
ArrayCode::index(const std::string &array_name,
                 const std::string &idx,
                 const int component) const
{
  // each component has a pointer, get the pointer name
  std::string pointer_name;

  std::ptrdiff_t offset;
  std::ptrdiff_t stride;
  size_t pointer_size;
  const auto array_it = array_map.find(array_name);
  if(array_it == array_map.end())
  {
    // array is not in the map meaning it was created in the kernel
    // by default that means it's interleaved...
    // This is not even used for temporary_field because their schema is put
    // into array_map explicitely
    pointer_name = array_name;
    if(component == -1)
    {
      offset = 0;
      stride = 8;
      pointer_size = 8;
    }
    else
    {
      offset = 8 * component;
      stride = 8;
      pointer_size = 8;
    }
  }
  else
  {
    const int num_components = array_it->second.schema.number_of_children();
    if(component == -1)
    {
      // check that the array only has one component
      if(num_components != 0)
      {
        ASCENT_ERROR("ArrayCode could not get the index of array '"
                     << array_name << "' because it has " << num_components
                     << " components and no component was specified.");
      }
      offset = array_it->second.schema.dtype().offset();
      stride = array_it->second.schema.dtype().stride();
      pointer_size = array_it->second.schema.dtype().element_bytes();

      pointer_name = array_name;
    }
    else
    {
      if(component >= num_components)
      {
        ASCENT_ERROR("ArrayCode could not get component "
                     << component << " of an array which only has "
                     << num_components << " components.");
      }
      const conduit::Schema &component_schema =
          array_it->second.schema.child(component);
      offset = component_schema.dtype().offset();
      stride = component_schema.dtype().stride();
      pointer_size = component_schema.dtype().element_bytes();

      if(array_it->second.codegen_array)
      {
        pointer_name = component_schema.name();
      }
      else
      {
        pointer_name = array_name + "_" + component_schema.name();
      }
    }
  }

  return index(idx, pointer_name, offset, stride, pointer_size);
}

std::string
ArrayCode::index(const std::string &array_name,
                 const std::string &idx,
                 const std::string &component) const
{

  const auto array_it = array_map.find(array_name);
  if(array_it == array_map.end())
  {
    ASCENT_ERROR("Cannot get the component '"
                 << component << "' of array '" << array_name
                 << "' because its schema was not found. Try using an integer "
                    "component instead of a string component.");
  }
  return index(array_name, idx, array_it->second.schema.child_index(component));
}

void
schemaFactory(const std::string &schema_type,
              const conduit::DataType::TypeID type_id,
              const size_t component_size,
              const std::vector<std::string> &component_names,
              conduit::Schema &out_schema)
{
  if(component_names.size() == 0)
  {
    out_schema.set(conduit::DataType(type_id, component_size));
  }
  else
  {
    if(schema_type == "contiguous")
    {
      for(size_t i = 0; i < component_names.size(); ++i)
      {
        const auto element_bytes = conduit::DataType::default_bytes(type_id);
        out_schema[component_names[i]].set(
            conduit::DataType(type_id,
                              component_size,
                              component_size * element_bytes * i,
                              element_bytes,
                              element_bytes,
                              conduit::Endianness::DEFAULT_ID));
      }
    }
    else if(schema_type == "interleaved")
    {
      for(size_t i = 0; i < component_names.size(); ++i)
      {
        const auto element_bytes = conduit::DataType::default_bytes(type_id);
        out_schema[component_names[i]].set(
            conduit::DataType(type_id,
                              component_size,
                              element_bytes * i,
                              element_bytes * component_names.size(),
                              element_bytes,
                              conduit::Endianness::DEFAULT_ID));
      }
    }
    else
    {
      ASCENT_ERROR("schemaFactory: Unknown schema type '" << schema_type
                                                          << "'.");
    }
  }
}

void
schemaFactory(const std::string &schema_type,
              const conduit::DataType::TypeID type_id,
              const size_t component_size,
              const int num_components,
              conduit::Schema &out_schema)
{
  std::vector<std::string> component_names = detail::component_names(num_components);
  schemaFactory(schema_type, type_id, component_size, component_names, out_schema);
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
