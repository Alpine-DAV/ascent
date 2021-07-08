//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_derived_jit.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_derived_jit.hpp"
#include "ascent_array.hpp"
#include "ascent_blueprint_architect.hpp"
#include "ascent_blueprint_topologies.hpp"
#include "ascent_expressions_ast.hpp"

#include <ascent_data_logger.hpp>
#include <ascent_logging.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <occa.hpp>

#ifdef ASCENT_CUDA_ENABLED
#include <occa/modes/cuda/utils.hpp>
#endif

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

std::string
type_string(const conduit::DataType &dtype)
{
  std::string type;
  if(dtype.is_float64())
  {
    type = "double";
  }
  else if(dtype.is_float32())
  {
    type = "float";
  }
  else if(dtype.is_int32())
  {
    type = "int";
  }
  else if(dtype.is_int64())
  {
    type = "long";
  }
  else if(dtype.is_unsigned_integer())
  {
    type = "unsigned long";
  }
  else
  {
    ASCENT_ERROR("JIT: unknown argument type: " << dtype.to_string());
  }
  return type;
}

//-----------------------------------------------------------------------------
// -- Array Allocation Functions
//-----------------------------------------------------------------------------
//{{{

using slice_t = std::tuple<size_t, size_t, size_t>;

void
get_occa_mem(std::vector<Array<unsigned char>> &buffers,
             const std::vector<slice_t> &slices,
             std::vector<occa::memory> &occa)
{
  occa::device &device = occa::getDevice();
  ASCENT_DATA_OPEN("copy to device");
  const std::string mode = device.mode();

  // I think the valid modes are: "Serial" "OpenMP", "CUDA:
  const size_t num_slices = slices.size();
  occa.resize(num_slices);

  for(size_t i = 0; i < num_slices; ++i)
  {
    ASCENT_DATA_OPEN("array_" + std::to_string(i));
    flow::Timer device_array_timer;
    Array<unsigned char> &buf = buffers[std::get<0>(slices[i])];
    size_t buf_offset = std::get<1>(slices[i]);
    size_t buf_size = std::get<2>(slices[i]);
    if(mode == "Serial" || mode == "OpenMP")
    {
      unsigned char *ptr = buf.get_host_ptr();
      occa[i] = occa::cpu::wrapMemory(
          device, ptr + buf_offset, buf_size * sizeof(unsigned char));
    }
#ifdef ASCENT_CUDA_ENABLED
    else if(mode == "CUDA")
    {
      unsigned char *ptr = buf.get_device_ptr();
      occa[i] = occa::cuda::wrapMemory(
          device, ptr + buf_offset, buf_size * sizeof(unsigned char));
    }
#endif
    else
    {
      ASCENT_ERROR("Unknow occa mode " << mode);
    }
    ASCENT_DATA_ADD("bytes", buf_size);
    ASCENT_DATA_CLOSE();
  }
  ASCENT_DATA_CLOSE();
}

// pack/allocate the host array using a contiguous schema
void
host_realloc_array(const conduit::Node &src_array,
                   const conduit::Schema &dest_schema,
                   conduit::Node &dest_array)
{
  // This check isn't strong enough, it will pass if the objects have different
  // child names
  if(!dest_schema.compatible(src_array.schema()))
  {
    ASCENT_ERROR("JIT: failed to allocate host array because the source and "
                 "destination schemas are incompatible.");
  }
  if(src_array.schema().equals(dest_schema))
  {
    // we don't need to copy the data since it's already the schema we want
    dest_array.set_external(src_array);
  }
  else
  {
    // reallocate to a better schema
    dest_array.set(dest_schema);
    dest_array.update_compatible(src_array);

    // src_array.info().print();
    // dest_array.info().print();
  }
}

// temporaries will always be one chunk of memory
void
device_alloc_temporary(const std::string &array_name,
                       const conduit::Schema &dest_schema,
                       conduit::Node &args,
                       std::vector<Array<unsigned char>> &array_memories,
                       std::vector<slice_t> &slices,
                       unsigned char *host_ptr)
{
  ASCENT_DATA_OPEN("temp Array");

  const auto size = dest_schema.total_bytes_compact();
  Array<unsigned char> mem;
  if(host_ptr == nullptr)
  {
    mem.resize(size);
  }
  else
  {
    // instead of using a temporary array for output and copying, we can point
    // Array<> at the destination conduit array
    mem.set(host_ptr, size);
  }
  array_memories.push_back(mem);
  slices.push_back(slice_t(array_memories.size() - 1, 0, size));

  if(dest_schema.number_of_children() == 0)
  {
    const std::string param =
        detail::type_string(dest_schema.dtype()) + " *" + array_name;
    args[param + "/index"] = slices.size() - 1;
  }
  else
  {
    for(const std::string &component : dest_schema.child_names())
    {
      const std::string param =
          detail::type_string(dest_schema[component].dtype()) + " *" +
          array_name + "_" + component;
      args[param + "/index"] = slices.size() - 1;
    }
  }

  ASCENT_DATA_ADD("bytes", dest_schema.total_bytes_compact());
  ASCENT_DATA_CLOSE();
}

void
device_alloc_array(const conduit::Node &array,
                   const conduit::Schema &dest_schema,
                   conduit::Node &args,
                   std::vector<Array<unsigned char>> &array_memories,
                   std::vector<slice_t> &slices)
{
  ASCENT_DATA_OPEN("input Array");
  conduit::Node res_array;
  flow::Timer host_array_timer;
  host_realloc_array(array, dest_schema, res_array);
  ASCENT_DATA_ADD("bytes", res_array.total_bytes_compact());
  flow::Timer device_array_timer;
  if(array.number_of_children() == 0)
  {
    unsigned char *start_ptr =
        static_cast<unsigned char *>(const_cast<void *>(res_array.data_ptr()));

    const auto size = res_array.total_bytes_compact();
    Array<unsigned char> mem;
    mem.set(start_ptr, size);

    const std::string param =
        "const " + detail::type_string(array.dtype()) + " *" + array.name();
    array_memories.push_back(mem);
    slices.push_back(slice_t(array_memories.size() - 1, 0, size));
    args[param + "/index"] = slices.size() - 1;
  }
  else
  {
    std::set<MemoryRegion> memory_regions;
    for(const std::string &component : res_array.child_names())
    {
      const conduit::Node &n_component = res_array[component];
      const void *start_ptr = res_array[component].data_ptr();
      const size_t size = n_component.dtype().spanned_bytes();
      MemoryRegion memory_region(start_ptr, size);
      const auto inserted = memory_regions.insert(memory_region);
      // if overlaps, union two regions
      if(!inserted.second)
      {
        auto hint = inserted.first;
        hint++;
        const unsigned char *min_start =
            std::min(inserted.first->start,
                     memory_region.start,
                     std::less<const unsigned char *>());
        const unsigned char *max_end =
            std::max(inserted.first->end,
                     memory_region.end,
                     std::less<const unsigned char *>());
        MemoryRegion unioned_region(min_start, max_end);
        memory_regions.erase(inserted.first);
        memory_regions.insert(hint, unioned_region);
      }
    }
    //
    // Matt: I feel like this is overkill, although I don't claim to know
    //       what this code fully does. Right now, its either block copyable
    //       or not. It should not matter if the block is in some larger region.
    //       I am just a cave man.
    //
    for(const std::string &component : res_array.child_names())
    {
      const conduit::Node &n_component = res_array[component];
      const void *start_ptr = res_array[component].data_ptr();
      const size_t size = n_component.dtype().spanned_bytes();
      MemoryRegion sub_region(start_ptr, size);
      const auto full_region_it = memory_regions.find(sub_region);
      if(!full_region_it->allocated)
      {
        full_region_it->allocated = true;
        Array<unsigned char> mem;
        unsigned char *data_ptr =
            const_cast<unsigned char *>(full_region_it->start);
        mem.set(data_ptr, full_region_it->end - full_region_it->start);
        array_memories.push_back(mem);
        full_region_it->index = array_memories.size() - 1;
      }
      const std::string param = "const " +
                                detail::type_string(n_component.dtype()) +
                                " *" + array.name() + "_" + component;
      // make a slice, push it and use that to support cases where
      // we have multiple pointers inside one allocation
      slices.push_back(slice_t(full_region_it->index,
                               static_cast<const unsigned char *>(start_ptr) -
                                   full_region_it->start,
                               size));
      args[param + "/index"] = slices.size() - 1;
    }
  }
  ASCENT_DATA_CLOSE();
}
//}}}

std::string
indent_code(const std::string &input_code, const int num_spaces)
{
  std::stringstream ss(input_code);
  std::string line;
  std::unordered_set<std::string> lines;
  std::string output_code;
  // num_spaces is the starting indentation level
  std::string indent(num_spaces, ' ');
  while(std::getline(ss, line))
  {
    if(line == "{")
    {
      output_code += indent + line + "\n";
      indent += "  ";
    }
    else if(line == "}")
    {
      try
      {
        indent = indent.substr(2);
      }
      catch(const std::out_of_range &e)
      {
        ASCENT_ERROR("Could not indent string:\n" << input_code);
      }
      output_code += indent + line + "\n";
    }
    else
    {
      output_code += indent + line + "\n";
    }
  }
  return output_code;
}
};

//-----------------------------------------------------------------------------
// -- MemoryRegion
//-----------------------------------------------------------------------------
//{{{
MemoryRegion::MemoryRegion(const void *start, const void *end)
    : start(static_cast<const unsigned char *>(start)),
      end(static_cast<const unsigned char *>(end)), allocated(false)
{
}

MemoryRegion::MemoryRegion(const void *start, const size_t size)
    : start(static_cast<const unsigned char *>(start)), end(this->start + size),
      allocated(false)
{
}

bool
MemoryRegion::operator<(const MemoryRegion &other) const
{
  return std::less<const void *>()(end, other.start);
}
//}}}

//-----------------------------------------------------------------------------
// -- ArrayCode
//-----------------------------------------------------------------------------
// {{{
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
// }}}

//-----------------------------------------------------------------------------
// -- MathCode
//-----------------------------------------------------------------------------
// {{{
void
MathCode::determinant_2x2(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &res_name,
                          const bool declare) const
{
  code.insert((declare ? "const double " : "") + res_name + " = " + a +
              "[0] * " + b + "[1] - " + b + "[0] * " + a + "[1];\n");
}
void
MathCode::determinant_3x3(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &c,
                          const std::string &res_name,
                          const bool declare) const
{
  code.insert((declare ? "const double " : "") + res_name + " = " + a +
              "[0] * (" + b + "[1] * " + c + "[2] - " + c + "[1] * " + b +
              "[2]) - " + a + "[1] * (" + b + "[0] * " + c + "[2] - " + c +
              "[0] * " + b + "[2]) + " + a + "[2] * (" + b + "[0] * " + c +
              "[1] - " + c + "[0] * " + b + "[1]);\n");
}

void
MathCode::vector_subtract(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &res_name,
                          const int num_components,
                          const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[" + std::to_string(num_components) +
                "];\n");
  }
  for(int i = 0; i < num_components; ++i)
  {
    code.insert(res_name + "[" + std::to_string(i) + "] = " + a + "[" +
                std::to_string(i) + "] - " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

void
MathCode::vector_add(InsertionOrderedSet<std::string> &code,
                     const std::string &a,
                     const std::string &b,
                     const std::string &res_name,
                     const int num_components,
                     const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[" + std::to_string(num_components) +
                "];\n");
  }
  for(int i = 0; i < num_components; ++i)
  {
    code.insert(res_name + "[" + std::to_string(i) + "] = " + a + "[" +
                std::to_string(i) + "] + " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

void
MathCode::cross_product(InsertionOrderedSet<std::string> &code,
                        const std::string &a,
                        const std::string &b,
                        const std::string &res_name,
                        const int num_components,
                        const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[3];\n");
  }
  if(num_components == 3)
  {
    code.insert(res_name + "[0] = " + a + "[1] * " + b + "[2] - " + a +
                "[2] * " + b + "[1];\n");
    code.insert(res_name + "[1] = " + a + "[2] * " + b + "[0] - " + a +
                "[0] * " + b + "[2];\n");
  }
  else if(num_components == 2)
  {
    code.insert(res_name + "[0] = 0;\n");
    code.insert(res_name + "[1] = 0;\n");
  }
  else
  {
    ASCENT_ERROR("cross_product is not implemented for vectors '"
                 << a << "' and '" << b << "' with " << num_components
                 << "components.");
  }
  code.insert(res_name + "[2] = " + a + "[0] * " + b + "[1] - " + a + "[1] * " +
              b + "[0];\n");
}

void
MathCode::dot_product(InsertionOrderedSet<std::string> &code,
                      const std::string &a,
                      const std::string &b,
                      const std::string &res_name,
                      const int num_components,
                      const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[" + std::to_string(num_components) +
                "];\n");
  }
  for(int i = 0; i < num_components; ++i)
  {
    code.insert(res_name + "[" + std::to_string(i) + "] = " + a + "[" +
                std::to_string(i) + "] * " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

void
MathCode::magnitude(InsertionOrderedSet<std::string> &code,
                    const std::string &a,
                    const std::string &res_name,
                    const int num_components,
                    const bool declare) const
{
  if(num_components == 3)
  {
    code.insert((declare ? "const double " : "") + res_name + " = sqrt(" + a +
                "[0] * " + a + "[0] + " + a + "[1] * " + a + "[1] + " + a +
                "[2] * " + a + "[2]);\n");
  }
  else if(num_components == 2)
  {
    code.insert((declare ? "const double " : "") + res_name + " = sqrt(" + a +
                "[0] * " + a + "[0] + " + a + "[1] * " + a + "[1]);\n");
  }
  else
  {
    ASCENT_ERROR("magnitude for vector '" << a << "' of size " << num_components
                                          << " is not implemented.");
  }
}

void
MathCode::array_avg(InsertionOrderedSet<std::string> &code,
                    const int length,
                    const std::string &array_name,
                    const std::string &res_name,
                    const bool declare) const
{
  std::stringstream array_avg;
  array_avg << "(";
  for(int i = 0; i < length; ++i)
  {
    if(i != 0)
    {
      array_avg << " + ";
    }
    array_avg << array_name + "[" << i << "]";
  }
  array_avg << ") / " << length;
  code.insert((declare ? "const double " : "") + res_name + " = " +
              array_avg.str() + ";\n");
}

// average value of a component given an array of vectors
void
MathCode::component_avg(InsertionOrderedSet<std::string> &code,
                        const int length,
                        const std::string &array_name,
                        const std::string &coord,
                        const std::string &res_name,
                        const bool declare) const
{
  const int component = coord[0] - 'x';
  std::stringstream comp_avg;
  comp_avg << "(";
  for(int i = 0; i < length; ++i)
  {
    if(i != 0)
    {
      comp_avg << " + ";
    }
    comp_avg << array_name + "[" << i << "][" << component << "]";
  }
  comp_avg << ") / " << length;
  code.insert((declare ? "const double " : "") + res_name + " = " +
              comp_avg.str() + ";\n");
}

// }}}

//-----------------------------------------------------------------------------
// -- TopologyCode
//-----------------------------------------------------------------------------
// {{{
TopologyCode::TopologyCode(const std::string &topo_name,
                           const conduit::Node &domain,
                           const ArrayCode &array_code)
    : topo_name(topo_name), domain(domain), array_code(array_code), math_code()
{
  const conduit::Node &n_topo = domain["topologies/" + topo_name];
  const std::string coords_name = n_topo["coordset"].as_string();
  this->topo_type = n_topo["type"].as_string();
  this->num_dims = topo_dim(topo_name, domain);
  if(topo_type == "unstructured")
  {
    this->shape =
        domain["topologies/" + topo_name + "/elements/shape"].as_string();
    if(shape == "polygonal")
    {
      // multiple shapes
      this->shape_size = -1;
    }
    else if(shape == "polyhedral")
    {
      const std::string &subelement_shape =
          domain["topologies/" + topo_name + "/subelements/shape"].as_string();
      if(subelement_shape != "polygonal")
      {
        // shape_size becomes the number of vertices for the subelements
        this->shape_size = get_num_vertices(shape);
      }
      else
      {
        this->shape_size = -1;
      }
    }
    else
    {
      // single shape
      this->shape_size = get_num_vertices(shape);
    }
  }
  else
  {
    // uniform, rectilinear, structured
    this->shape_size = static_cast<int>(std::pow(2, num_dims));
  }
}

void
TopologyCode::element_idx(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type == "uniform" || topo_type == "rectilinear" ||
     topo_type == "structured")
  {
    code.insert({"int " + topo_name + "_element_idx[" +
                     std::to_string(num_dims) + "];\n",
                 topo_name + "_element_idx[0] = item % (" + topo_name +
                     "_dims_i - 1);\n"});

    if(num_dims >= 2)
    {
      code.insert(topo_name + "_element_idx[1] = (item / (" + topo_name +
                  "_dims_i - 1)) % (" + topo_name + "_dims_j - 1);\n");
    }
    if(num_dims == 3)
    {
      code.insert(topo_name + "_element_idx[2] = item / ((" + topo_name +
                  "_dims_i - 1) * (" + topo_name + "_dims_j - 1));\n");
    }
  }
  else
  {
    ASCENT_ERROR("element_idx for unstructured is not implemented.");
  }
}

// vertices are ordered in the VTK format
// https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
// I'm also assuming the x,y,z axis shown to the left of VTK_HEXAHEDRON
void
TopologyCode::structured_vertices(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type != "uniform" && topo_type != "rectilinear" &&
     topo_type != "structured")
  {
    ASCENT_ERROR("The function structured_vertices only supports uniform, "
                 "rectilinear, and structured topologies.");
  }
  element_idx(code);

  // vertex indices
  code.insert("int " + topo_name + "_vertices[" + std::to_string(shape_size) +
              "];\n");
  if(num_dims == 1)
  {
    code.insert(
        {topo_name + "_vertices[0] = " + topo_name + "_element_idx[0];\n",
         topo_name + "_vertices[1] = " + topo_name + "_vertices[0] + 1;\n"});
  }
  else if(num_dims == 2)
  {
    code.insert(
        {topo_name + "_vertices[0] = " + topo_name + "_element_idx[1] * " +
             topo_name + "_dims_i + " + topo_name + "_element_idx[0];\n",
         topo_name + "_vertices[1] = " + topo_name + "_vertices[0] + 1;\n",
         topo_name + "_vertices[2] = " + topo_name + "_vertices[1] + " +
             topo_name + "_dims_i;\n",
         topo_name + "_vertices[3] = " + topo_name + "_vertices[2] - 1;\n"});
  }
  else if(num_dims == 3)
  {
    code.insert({
        topo_name + "_vertices[0] = (" + topo_name + "_element_idx[2] * " +
            topo_name + "_dims_j + " + topo_name + "_element_idx[1]) * " +
            topo_name + "_dims_i + " + topo_name + "_element_idx[0];\n",
        topo_name + "_vertices[1] = " + topo_name + "_vertices[0] + 1;\n",
        topo_name + "_vertices[2] = " + topo_name + "_vertices[1] + " +
            topo_name + "_dims_i;\n",
        topo_name + "_vertices[3] = " + topo_name + "_vertices[2] - 1;\n",
        topo_name + "_vertices[4] = " + topo_name + "_vertices[0] + " +
            topo_name + "_dims_i * " + topo_name + "_dims_j;\n",
        topo_name + "_vertices[5] = " + topo_name + "_vertices[4] + 1;\n",
        topo_name + "_vertices[6] = " + topo_name + "_vertices[5] + " +
            topo_name + "_dims_i;\n",
        topo_name + "_vertices[7] = " + topo_name + "_vertices[6] - 1;\n",
    });
  }
}

void
TopologyCode::structured_vertex_locs(
    InsertionOrderedSet<std::string> &code) const
{
  if(topo_type != "structured")
  {
    ASCENT_ERROR("The function structured_vertex_locs only supports structured "
                 "topologies.");
  }
  structured_vertices(code);
  code.insert("double " + topo_name + "_vertex_locs[" +
              std::to_string(shape_size) + "][" + std::to_string(num_dims) +
              "];\n");
  for(int i = 0; i < shape_size; ++i)
  {
    vertex_xyz(code,
               array_code.index(topo_name + "_vertices", std::to_string(i)),
               false,
               array_code.index(topo_name + "_vertex_locs", std::to_string(i)),
               false);
  }
}

void
TopologyCode::unstructured_vertices(InsertionOrderedSet<std::string> &code,
                                    const std::string &index_name) const
{
  if(topo_type != "unstructured")
  {
    ASCENT_ERROR(
        "The function unstructured_vertices only supports unstructured "
        "topologies.");
  }
  if(shape_size == -1)
  {
    // TODO generate vertices array for multi-shapes case, it's variable length
    // so might have to find the max shape size before hand and pass it in
  }
  else
  {
    // single shape
    // inline the for-loop
    code.insert("int " + topo_name + "_vertices[" + std::to_string(shape_size) +
                "];\n");
    for(int i = 0; i < shape_size; ++i)
    {
      code.insert(topo_name + "_vertices[" + std::to_string(i) +
                  "] = " + topo_name + "_connectivity[" + index_name + " * " +
                  std::to_string(shape_size) + " + " + std::to_string(i) +
                  "];\n");
    }
  }
}

void
TopologyCode::unstructured_vertex_locs(InsertionOrderedSet<std::string> &code,
                                       const std::string &index_name) const
{
  if(topo_type != "unstructured")
  {
    ASCENT_ERROR(
        "The function unstructured_vertex_locs only supports unstructured "
        "topologies.");
  }
  unstructured_vertices(code, index_name);
  if(shape_size == -1)
  {
    // multiple shapes
    code.insert({"int " + topo_name + "_shape_size = " + topo_name + "_sizes[" +
                     index_name + "];\n",
                 "int " + topo_name + "_offset = " + topo_name + "_offsets[" +
                     index_name + "];\n",
                 "double " + topo_name + "_vertex_locs[" + topo_name +
                     "_shape_size][" + std::to_string(num_dims) + "];\n"});

    InsertionOrderedSet<std::string> for_loop;
    for_loop.insert(
        {"for(int i = 0; i < " + topo_name + "_shape_size; ++i)\n", "{\n"});
    vertex_xyz(for_loop,
               array_code.index(topo_name + "_connectivity",
                                topo_name + "_offset + i"),
               false,
               array_code.index(topo_name + "_vertex_locs", "i"),
               false);
    for_loop.insert("}\n");
    code.insert(for_loop.accumulate());
  }
  else
  {
    // single shape
    code.insert("double " + topo_name + "_vertex_locs[" +
                std::to_string(shape_size) + "][" + std::to_string(num_dims) +
                "];\n");
    for(int i = 0; i < shape_size; ++i)
    {
      vertex_xyz(
          code,
          array_code.index(topo_name + "_vertices", std::to_string(i)),
          false,
          array_code.index(topo_name + "_vertex_locs", std::to_string(i)),
          false);
    }
  }
}

void
TopologyCode::element_coord(InsertionOrderedSet<std::string> &code,
                            const std::string &coord,
                            const std::string &index_name,
                            const std::string &res_name,
                            const bool declare) const
{
  // if the logical index is provided, don't regenerate it
  std::string my_index_name;
  if(index_name.empty() &&
     (topo_type == "uniform" || topo_type == "rectilinear" ||
      topo_type == "structured"))
  {
    element_idx(code);
    my_index_name =
        topo_name + "_element_idx[" + std::to_string(coord[0] - 'x') + "]";
  }
  else
  {
    my_index_name = index_name;
  }
  if(topo_type == "uniform")
  {
    code.insert((declare ? "const double " : "") + res_name + " = " +
                topo_name + "_origin_" + coord + +" + (" + my_index_name +
                " + 0.5) * " + topo_name + "_spacing_d" + coord + ";\n");
  }
  else if(topo_type == "rectilinear")
  {
    code.insert(
        (declare ? "const double " : "") + res_name + " = (" +
        array_code.index(topo_name + "_coords", my_index_name, coord) + " + " +
        array_code.index(topo_name + "_coords", my_index_name + " + 1", coord) +
        ") / 2.0;\n");
  }
  else if(topo_type == "structured")
  {
    structured_vertex_locs(code);
    math_code.component_avg(
        code, shape_size, topo_name + "_vertex_locs", coord, res_name, declare);
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertex_locs(code);
    if(shape_size == -1)
    {
      // multiple shapes
      // This will generate 3 for loops if we want to calculate element_xyz
      // If this is an issue we can make a special case for it in
      // element_xyz
      InsertionOrderedSet<std::string> for_loop;
      for_loop.insert(
          {"for(int i = 0; i < " + topo_name + "_shape_size; ++i)\n", "{\n"});
      math_code.component_avg(for_loop,
                              shape_size,
                              topo_name + "_vertex_locs",
                              coord,
                              res_name,
                              declare);
      for_loop.insert("}\n");
      code.insert(for_loop.accumulate());
    }
    else
    {
      // single shape
      for(int i = 0; i < shape_size; ++i)
      {
        math_code.component_avg(code,
                                shape_size,
                                topo_name + "_vertex_locs",
                                coord,
                                res_name,
                                declare);
      }
    }
  }
  else
  {
    ASCENT_ERROR("Cannot get element_coord for topology of type '" << topo_type
                                                                   << "'.");
  }
}

void
TopologyCode::element_xyz(InsertionOrderedSet<std::string> &code) const
{
  code.insert("double " + topo_name + "_element_loc[" +
              std::to_string(num_dims) + "];\n");
  if(topo_type == "uniform" || topo_type == "rectilinear" ||
     topo_type == "structured" || topo_type == "unstructured")
  {
    element_coord(code, "x", "", topo_name + "_element_loc[0]", false);
    if(num_dims >= 2)
    {
      element_coord(code, "y", "", topo_name + "_element_loc[1]", false);
    }
    if(num_dims == 3)
    {
      element_coord(code, "z", "", topo_name + "_element_loc[2]", false);
    }
  }
  else
  {
    ASCENT_ERROR("Cannot get element location for unstructured topology with "
                 << num_dims << " dimensions.");
  }
}

void
TopologyCode::vertex_idx(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type == "uniform" || topo_type == "rectilinear" ||
     topo_type == "structured")
  {
    code.insert(
        {"int " + topo_name + "_vertex_idx[" + std::to_string(num_dims) +
             "];\n",
         topo_name + "_vertex_idx[0] = item % (" + topo_name + "_dims_i);\n"});
    if(num_dims >= 2)
    {
      code.insert(topo_name + "_vertex_idx[1] = (item / (" + topo_name +
                  "_dims_i)) % (" + topo_name + "_dims_j);\n");
    }
    if(num_dims == 3)
    {
      code.insert(topo_name + "_vertex_idx[2] = item / ((" + topo_name +
                  "_dims_i) * (" + topo_name + "_dims_j));\n");
    }
  }
  else
  {
    // vertex_idx is just item for explicit (unstructured)
    // vertex_idx[0] = item
    // vertex_idx[1] = item
    // vertex_idx[2] = item
    ASCENT_ERROR("vertex_idx does not need to be calculated for unstructured "
                 "topologies.");
  }
}

void
TopologyCode::vertex_coord(InsertionOrderedSet<std::string> &code,
                           const std::string &coord,
                           const std::string &index_name,
                           const std::string &res_name,
                           const bool declare) const
{
  std::string my_index_name;
  if(index_name.empty())
  {
    if(topo_name == "uniform" || topo_name == "rectilinear")
    {
      vertex_idx(code);
      my_index_name =
          topo_name + "_vertex_idx[" + std::to_string(coord[0] - 'x') + "]";
    }
    else
    {
      my_index_name = "item";
    }
  }
  else
  {
    my_index_name = index_name;
  }
  if(topo_type == "uniform")
  {
    code.insert((declare ? "const double " : "") + res_name + " = " +
                topo_name + "_origin_" + coord + " + " + my_index_name + " * " +
                topo_name + "_spacing_d" + coord + ";\n");
  }
  else
  {
    code.insert((declare ? "const double " : "") + res_name + " = " +
                array_code.index(topo_name + "_coords", my_index_name, coord) +
                ";\n");
  }
}

void
TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code,
                         const std::string &index_name,
                         const bool index_array,
                         const std::string &res_name,
                         const bool declare) const
{
  if(declare)
  {
    code.insert("double " + res_name + "[" + std::to_string(num_dims) + "];\n");
  }
  vertex_coord(code,
               "x",
               index_name + (index_array ? "[0]" : ""),
               res_name + "[0]",
               false);
  if(num_dims >= 2)
  {
    vertex_coord(code,
                 "y",
                 index_name + (index_array ? "[1]" : ""),
                 res_name + "[1]",
                 false);
  }
  if(num_dims == 3)
  {
    vertex_coord(code,
                 "z",
                 index_name + (index_array ? "[2]" : ""),
                 res_name + "[2]",
                 false);
  }
}

void
TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type == "uniform" || topo_type == "rectilinear")
  {
    vertex_idx(code);
    vertex_xyz(
        code, topo_name + "_vertex_idx", true, topo_name + "_vertex_loc");
  }
  else if(topo_type == "structured" || topo_type == "unstructured")
  {
    vertex_xyz(code, "item", false, topo_name + "_vertex_loc");
  }
}

// get rectilinear spacing for a cell
void
TopologyCode::dxdydz(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type != "rectilinear")
  {
    ASCENT_ERROR("Function dxdydz only works on rectilinear topologies.");
  }
  element_idx(code);
  code.insert("const double " + topo_name + "_dx = " +
              array_code.index(topo_name + "_coords",
                               topo_name + "_element_idx[0] + 1",
                               "x") +
              " - " +
              array_code.index(
                  topo_name + "_coords", topo_name + "_element_idx[0]", "x") +
              ";\n");
  if(num_dims >= 2)
  {
    code.insert("const double " + topo_name + "_dy = " +
                array_code.index(topo_name + "_coords",
                                 topo_name + "_element_idx[1] + 1",
                                 "y") +
                " - " +
                array_code.index(
                    topo_name + "_coords", topo_name + "_element_idx[1]", "y") +
                ";\n");
  }
  if(num_dims == 3)
  {
    code.insert("const double " + topo_name + "_dz = " +
                array_code.index(topo_name + "_coords",
                                 topo_name + "_element_idx[2] + 1",
                                 "z") +
                " - " +
                array_code.index(
                    topo_name + "_coords", topo_name + "_element_idx[2]", "z") +
                ";\n");
  }
}

// https://www.osti.gov/servlets/purl/632793 (14)
// switch vertices 2 and 3 and vertices 6 and 7 to match vtk order
void
TopologyCode::hexahedral_volume(InsertionOrderedSet<std::string> &code,
                                const std::string &vertex_locs,
                                const std::string &res_name) const
{
  math_code.vector_subtract(
      code, vertex_locs + "[6]", vertex_locs + "[0]", res_name + "_6m0", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[1]", vertex_locs + "[0]", res_name + "_1m0", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[2]", vertex_locs + "[5]", res_name + "_2m5", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[4]", vertex_locs + "[0]", res_name + "_4m0", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[5]", vertex_locs + "[7]", res_name + "_5m7", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[3]", vertex_locs + "[0]", res_name + "_3m0", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[7]", vertex_locs + "[2]", res_name + "_7m2", 3);
  // can save 4 flops if we use the fact that 6m0 is always the first column
  // of the determinant
  math_code.determinant_3x3(code,
                            res_name + "_6m0",
                            res_name + "_1m0",
                            res_name + "_2m5",
                            res_name + "_det0");
  math_code.determinant_3x3(code,
                            res_name + "_6m0",
                            res_name + "_4m0",
                            res_name + "_5m7",
                            res_name + "_det1");
  math_code.determinant_3x3(code,
                            res_name + "_6m0",
                            res_name + "_3m0",
                            res_name + "_7m2",
                            res_name + "_det2");
  code.insert("const double " + res_name + " = (" + res_name + "_det0 + " +
              res_name + "_det1 + " + res_name + "_det2) / 6.0;\n");
}

// ||(p3-p0) (p2-p0) (p1-p0)|| / 6
void
TopologyCode::tetrahedral_volume(InsertionOrderedSet<std::string> &code,
                                 const std::string &vertex_locs,
                                 const std::string &res_name) const
{
  math_code.vector_subtract(
      code, vertex_locs + "[1]", vertex_locs + "[0]", res_name + "_1m0", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[2]", vertex_locs + "[0]", res_name + "_2m0", 3);
  math_code.vector_subtract(
      code, vertex_locs + "[3]", vertex_locs + "[0]", res_name + "_3m0", 3);
  math_code.determinant_3x3(code,
                            res_name + "_3m0",
                            res_name + "_2m0",
                            res_name + "_1m0",
                            res_name + "_det");
  code.insert("const double " + res_name + " = " + res_name + "_det / 6.0;\n");
}

// 1/2 * |(p2 - p0) X (p3 - p1)|
void
TopologyCode::quadrilateral_area(InsertionOrderedSet<std::string> &code,
                                 const std::string &p0,
                                 const std::string &p1,
                                 const std::string &p2,
                                 const std::string &p3,
                                 const std::string &res_name) const
{
  math_code.vector_subtract(code, p2, p0, res_name + "_2m0", num_dims);
  math_code.vector_subtract(code, p3, p1, res_name + "_3m1", num_dims);
  math_code.cross_product(code,
                          res_name + "_2m0",
                          res_name + "_3m1",
                          res_name + "_cross",
                          num_dims);
  math_code.magnitude(code, res_name + "_cross", res_name + "_cross_mag", 3);
  code.insert("const double " + res_name + " = " + res_name +
              "_cross_mag / 2.0;\n");
}

void
TopologyCode::quadrilateral_area(InsertionOrderedSet<std::string> &code,
                                 const std::string &vertex_locs,
                                 const std::string &res_name) const
{
  quadrilateral_area(code,
                     vertex_locs + "[0]",
                     vertex_locs + "[1]",
                     vertex_locs + "[2]",
                     vertex_locs + "[3]",
                     res_name);
}

// 1/2 * |(p1 - p0) X (p2 - p0)|
void
TopologyCode::triangle_area(InsertionOrderedSet<std::string> &code,
                            const std::string &p0,
                            const std::string &p1,
                            const std::string &p2,
                            const std::string &res_name) const
{
  math_code.vector_subtract(code, p1, p0, res_name + "_1m0", num_dims);
  math_code.vector_subtract(code, p2, p0, res_name + "_2m0", num_dims);
  math_code.cross_product(code,
                          res_name + "_1m0",
                          res_name + "_2m0",
                          res_name + "_cross",
                          num_dims);
  math_code.magnitude(code, res_name + "_cross", res_name + "_cross_mag", 3);
  code.insert("const double " + res_name + " = " + res_name +
              "_cross_mag / 2.0;\n");
}

void
TopologyCode::triangle_area(InsertionOrderedSet<std::string> &code,
                            const std::string &vertex_locs,
                            const std::string &res_name) const
{
  triangle_area(code,
                vertex_locs + "[0]",
                vertex_locs + "[1]",
                vertex_locs + "[2]",
                res_name);
}

// http://index-of.co.uk/Game-Development/Programming/Graphics%20Gems%205.pdf
// k is # vertices, h = (k-1)//2, l = 0 if k is odd, l = k-1 if k is even
// 2A = sum_{i=1}^{h - 1}((P_2i - P_0) X (P_2i+1 - P_2i-1)) +
// (P_2h - P_0) X (P_l - P_2h-1)
void
TopologyCode::polygon_area_vec(InsertionOrderedSet<std::string> &code,
                               const std::string &vertex_locs,
                               const std::string &res_name) const
{
  code.insert({"double " + res_name + "_vec[3];\n",
               res_name + "_vec[0] = 0;\n",
               res_name + "_vec[1] = 0;\n",
               res_name + "_vec[2] = 0;\n",
               "const int " + res_name + "_h = (" + topo_name +
                   "_shape_size - 1) / 2;\n"});
  InsertionOrderedSet<std::string> for_loop;
  for_loop.insert({"for(int i = 1; i < " + res_name + "_h; ++i)\n", "{\n"});
  math_code.vector_subtract(for_loop,
                            vertex_locs + "[2 * i]",
                            vertex_locs + "[0]",
                            res_name + "_2im0",
                            num_dims);
  math_code.vector_subtract(for_loop,
                            vertex_locs + "[2 * i + 1]",
                            vertex_locs + "[2 * i - 1]",
                            res_name + "_2ip1_m_2im1",
                            num_dims);
  math_code.cross_product(for_loop,
                          res_name + "_2im0",
                          res_name + "_2ip1_m_2im1",
                          res_name + "_cross",
                          num_dims);
  math_code.vector_add(for_loop,
                       res_name + "_vec",
                       res_name + "_cross",
                       res_name + "_vec",
                       3,
                       false);
  for_loop.insert("}\n");
  code.insert(for_loop.accumulate());
  code.insert({"int " + res_name + "_last = ((" + topo_name +
               "_shape_size & 1) ^ 1) * (" + topo_name +
               "_shape_size - 1);\n"});
  math_code.vector_subtract(code,
                            vertex_locs + "[2 * " + res_name + "_h]",
                            vertex_locs + "[0]",
                            res_name + "_2hm0",
                            num_dims);
  math_code.vector_subtract(code,
                            vertex_locs + "[" + res_name + "_last]",
                            vertex_locs + "[2 * " + res_name + "_h - 1]",
                            res_name + "_l_m_2hm1",
                            num_dims);
  math_code.cross_product(code,
                          res_name + "_2hm0",
                          res_name + "_l_m_2hm1",
                          res_name + "_cross",
                          num_dims);
  math_code.vector_add(code,
                       res_name + "_vec",
                       res_name + "_cross",
                       res_name + "_vec",
                       3,
                       false);
}

void
TopologyCode::polygon_area(InsertionOrderedSet<std::string> &code,
                           const std::string &vertex_locs,
                           const std::string &res_name) const
{
  polygon_area_vec(code, vertex_locs, res_name);
  math_code.magnitude(code, res_name + "_vec", res_name + "_vec_mag", 3);
  code.insert("const double " + res_name + " = " + res_name +
              "_vec_mag / 2.0;\n");
}

// TODO this doesn't work because A_j needs to point outside the polyhedron
// (i.e. vertices need to be ordered counter-clockwise when looking from
// outside)
// m is number of faces
// 1/6 * sum_{j=0}^{m-1}(P_j . 2*A_j)
// P_j is some point on face j A_j is the area vector of face j
/*
void
TopologyCode::polyhedron_volume(InsertionOrderedSet<std::string> &code,
                                const std::string &vertex_locs,
                                const std::string &res_name) const
{
  code.insert({"double " + res_name + "_vec[3];\n",
               res_name + "_vec[0] = 0;\n",
               res_name + "_vec[1] = 0;\n",
               res_name + "_vec[2] = 0;\n",
               "int " + topo_name + "_polyhedral_shape_size = " + topo_name +
                   "_polyhedral_sizes[item];\n",
               "int " + topo_name + "_polyhedral_offset = " +
                   array_code.index(topo_name + "_polyhedral_offsets", "item")
+
                   ";\n"});

  InsertionOrderedSet<std::string> for_loop;
  for_loop.insert(
      {"for(int j = 0; j < " + topo_name + "_polyhedral_shape_size; ++j)\n",
       "{\n"});
  unstructured_vertex_locs(for_loop,
                        array_code.index(topo_name +
"_polyhedral_connectivity", topo_name + "_polyhedral_offset + j"));
  polygon_area_vec(for_loop, vertex_locs, res_name + "_face");
  math_code.dot_product(for_loop,
                        vertex_locs + "[4]",
                        res_name + "_face_vec",
                        res_name + "_dot",
                        num_dims);
  math_code.vector_add(for_loop,
                       res_name + "_vec",
                       res_name + "_dot",
                       res_name + "_vec",
                       num_dims,
                       false);
  for_loop.insert("}\n");
  code.insert(for_loop.accumulate());
  math_code.magnitude(code, res_name + "_vec", res_name + "_vec_mag",
num_dims); code.insert("double " + res_name + " = " + res_name + "_vec_mag
/ 6.0;\n");
}
*/

void
TopologyCode::volume(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type == "uniform")
  {
    code.insert("const double " + topo_name + "_volume = " + topo_name +
                "_spacing_dx * " + topo_name + "_spacing_dy * " + topo_name +
                "_spacing_dz;\n");
  }
  else if(topo_type == "rectilinear")
  {
    dxdydz(code);
    code.insert("const double " + topo_name + "_volume = " + topo_name +
                "_dx * " + topo_name + "_dy * " + topo_name + "_dz;\n");
  }
  else if(topo_type == "structured")
  {
    structured_vertex_locs(code);
    hexahedral_volume(code, topo_name + "_vertex_locs", topo_name + "_volume");
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertex_locs(code);
    if(shape == "hex")
    {
      hexahedral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(shape == "tet")
    {
      tetrahedral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    // else if(shape == "polyhedral")
    // {
    //   polyhedron_volume(
    //       code, topo_name + "_vertex_locs", topo_name + "_volume");
    // }
  }
}

void
TopologyCode::hexahedral_surface_area(InsertionOrderedSet<std::string> &code,
                                      const std::string &vertex_locs,
                                      const std::string &res_name) const
{
  // negative x face
  quadrilateral_area(code,
                     vertex_locs + "[4]",
                     vertex_locs + "[0]",
                     vertex_locs + "[3]",
                     vertex_locs + "[7]",
                     res_name + "_nx");
  // positive x face
  quadrilateral_area(code,
                     vertex_locs + "[1]",
                     vertex_locs + "[5]",
                     vertex_locs + "[6]",
                     vertex_locs + "[2]",
                     res_name + "_px");
  quadrilateral_area(code,
                     vertex_locs + "[4]",
                     vertex_locs + "[5]",
                     vertex_locs + "[1]",
                     vertex_locs + "[0]",
                     res_name + "_ny");
  quadrilateral_area(code,
                     vertex_locs + "[3]",
                     vertex_locs + "[2]",
                     vertex_locs + "[6]",
                     vertex_locs + "[7]",
                     res_name + "_py");
  quadrilateral_area(code,
                     vertex_locs + "[0]",
                     vertex_locs + "[1]",
                     vertex_locs + "[2]",
                     vertex_locs + "[3]",
                     res_name + "_nz");
  quadrilateral_area(code,
                     vertex_locs + "[4]",
                     vertex_locs + "[5]",
                     vertex_locs + "[6]",
                     vertex_locs + "[7]",
                     res_name + "_pz");
  code.insert("const double " + res_name + " = " + res_name + "_nx + " +
              res_name + "_px + " + res_name + "_ny + " + res_name + "_py + " +
              topo_name + "_area_nz + " + res_name + "_pz;\n");
}

void
TopologyCode::tetrahedral_surface_area(InsertionOrderedSet<std::string> &code,
                                       const std::string &vertex_locs,
                                       const std::string &res_name) const
{
  triangle_area(code,
                vertex_locs + "[0]",
                vertex_locs + "[2]",
                vertex_locs + "[1]",
                res_name + "_f0");
  triangle_area(code,
                vertex_locs + "[0]",
                vertex_locs + "[3]",
                vertex_locs + "[2]",
                res_name + "_f1");
  triangle_area(code,
                vertex_locs + "[0]",
                vertex_locs + "[1]",
                vertex_locs + "[3]",
                res_name + "_f2");
  triangle_area(code,
                vertex_locs + "[1]",
                vertex_locs + "[2]",
                vertex_locs + "[3]",
                res_name + "_f3");
  code.insert("const double " + res_name + " = " + res_name + "_f0 + " +
              res_name + "_f1 + " + res_name + "_f2 + " + res_name + "_f3;\n");
}

void
TopologyCode::area(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type == "uniform")
  {
    if(num_dims == 2)
    {
      code.insert("const double " + topo_name + "_area = " + topo_name +
                  "_spacing_dx * " + topo_name + "_spacing_dy;\n");
    }
    else
    {
      code.insert("const double " + topo_name + "_area = " + topo_name +
                  "_spacing_dx;\n");
    }
  }
  else if(topo_type == "rectilinear")
  {
    dxdydz(code);
    if(num_dims == 2)
    {
      code.insert("const double " + topo_name + "_area = " + topo_name +
                  "_dx * " + topo_name + "_dy;\n");
    }
    else
    {
      code.insert("const double " + topo_name + "_area = " + topo_name +
                  "_dx;\n");
    }
  }
  else if(topo_type == "structured")
  {
    structured_vertex_locs(code);
    if(num_dims == 2)
    {
      quadrilateral_area(code, topo_name + "_vertex_locs", topo_name + "_area");
    }
    else if(num_dims == 1)
    {
      math_code.vector_subtract(code,
                                topo_name + "vertex_locs[1]",
                                topo_name + "vertex_locs[0]",
                                topo_name + "_area",
                                1);
    }
    else
    {
      ASCENT_ERROR("area is not implemented for structured topologies with "
                   << num_dims << " dimensions.");
    }
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertex_locs(code);
    if(shape == "quad")
    {
      quadrilateral_area(code, topo_name + "_vertex_locs", topo_name + "_area");
    }
    else if(shape == "tri")
    {
      triangle_area(code, topo_name + "_vertex_locs", topo_name + "_area");
    }
    else if(shape == "polygonal")
    {
      polygon_area(code, topo_name + "_vertex_locs", topo_name + "_area");
    }
    else
    {
      ASCENT_ERROR("area for unstructured topology with shape '"
                   << shape << "' is not implemented.");
    }
  }
}

void
TopologyCode::surface_area(InsertionOrderedSet<std::string> &code) const
{
  if(topo_type == "uniform")
  {
    code.insert("const double " + topo_name + "_area = 2.0 * (" + topo_name +
                "_spacing_dx * " + topo_name + "_spacing_dy + " + topo_name +
                "_spacing_dx * " + topo_name + "_spacing_dz + " + topo_name +
                "_spacing_dy * " + topo_name + "_spacing_dz);\n");
  }
  else if(topo_type == "rectilinear")
  {
    dxdydz(code);
    code.insert("const double " + topo_name + "_area = 2.0 * (" + topo_name +
                "_dx * " + topo_name + "_dy + " + topo_name + "_dx * " +
                topo_name + "_dz + " + topo_name + "_dy * " + topo_name +
                "_dz);\n");
  }
  else if(topo_type == "structured")
  {
    structured_vertex_locs(code);
    hexahedral_surface_area(
        code, topo_name + "_vertex_locs", topo_name + "_area");
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertex_locs(code);
    if(shape == "hex")
    {
      hexahedral_surface_area(
          code, topo_name + "_vertex_locs", topo_name + "_area");
    }
    else if(shape == "tet")
    {
      tetrahedral_surface_area(
          code, topo_name + "_vertex_locs", topo_name + "_area");
    }
    // else if(shape == "polyhedral")
    // {
    //   polyhedron_surface_area(
    //       code, topo_name + "_vertex_locs", topo_name + "_area");
    // }
    else
    {
      ASCENT_ERROR("area for unstructured topology with shape '"
                   << shape << "' is not implemented.");
    }
  }
}
// }}}

//-----------------------------------------------------------------------------
// -- Packing Functions
//-----------------------------------------------------------------------------
// {{{
bool
is_compact_interleaved(const conduit::Node &array)
{
  const unsigned char *min_start = nullptr;
  const unsigned char *max_end = nullptr;
  if(array.number_of_children() == 0)
  {
    min_start = static_cast<const unsigned char *>(array.data_ptr());
    max_end = min_start + array.dtype().spanned_bytes();
  }
  else
  {
    for(const std::string &component : array.child_names())
    {
      const conduit::Node &n_component = array[component];
      if(min_start == nullptr || max_end == nullptr)
      {
        min_start = static_cast<const unsigned char *>(n_component.data_ptr());
        max_end = min_start + array.dtype().spanned_bytes();
      }
      else
      {
        const unsigned char *new_start =
            static_cast<const unsigned char *>(n_component.data_ptr());
        min_start =
            std::min(min_start, new_start, std::less<const unsigned char *>());

        max_end = std::max(max_end,
                           new_start + n_component.dtype().spanned_bytes(),
                           std::less<const unsigned char *>());
      }
    }
  }
  return (max_end - min_start) == array.total_bytes_compact();
}

// num_components should be 0 if you don't want an mcarray
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

// Compacts an array (generates a contigous schema) so that only one
// allocation is needed. Code generation will read this schema from
// array_code. The array in args is a set_external to the original data so we
// can copy to the device later.
void
pack_array(const conduit::Node &array,
           const std::string &name,
           conduit::Node &args,
           ArrayCode &array_code)
{
  args[name].set_external(array);
  if(array.is_compact() ||
     is_compact_interleaved(array) /* || array is on the device */)
  {
    // copy the existing schema
    array_code.array_map.insert(
        std::make_pair(name, SchemaBool(array.schema(), false)));
  }
  else
  {
    const int num_components = array.number_of_children();
    size_t size;
    conduit::DataType::TypeID type_id;
    if(num_components == 0)
    {
      type_id = static_cast<conduit::DataType::TypeID>(array.dtype().id());
      size = array.dtype().number_of_elements();
    }
    else
    {
      type_id =
          static_cast<conduit::DataType::TypeID>(array.child(0).dtype().id());
      size = array.child(0).dtype().number_of_elements();
    }

    conduit::Schema s;
    // TODO for now, if we need to copy we copy to contiguous
    schemaFactory("contiguous", type_id, size, array.child_names(), s);
    array_code.array_map.insert(std::make_pair(name, SchemaBool(s, false)));
  }
}

void
pack_topology(const std::string &topo_name,
              const conduit::Node &domain,
              conduit::Node &args,
              ArrayCode &array_code)
{
  const conduit::Node &topo = domain["topologies/" + topo_name];
  const std::string &topo_type = topo["type"].as_string();
  const conduit::Node &coords =
      domain["coordsets/" + topo["coordset"].as_string()];
  const size_t num_dims = topo_dim(topo_name, domain);
  if(topo_type == "uniform")
  {
    for(size_t i = 0; i < num_dims; ++i)
    {
      const std::string dim = std::string(1, 'i' + i);
      const std::string coord = std::string(1, 'x' + i);
      args[topo_name + "_dims_" + dim] = coords["dims"].child(i);
      args[topo_name + "_spacing_d" + coord] = coords["spacing"].child(i);
      args[topo_name + "_origin_" + coord] = coords["origin"].child(i);
    }
  }
  else if(topo_type == "rectilinear")
  {
    for(size_t i = 0; i < num_dims; ++i)
    {
      const std::string dim = std::string(1, 'i' + i);
      args[topo_name + "_dims_" + dim] =
          coords["values"].child(i).dtype().number_of_elements();
    }
    pack_array(coords["values"], topo_name + "_coords", args, array_code);
  }
  else if(topo_type == "structured")
  {
    for(size_t i = 0; i < num_dims; ++i)
    {
      const std::string dim = std::string(1, 'i' + i);
      args[topo_name + "_dims_" + dim] =
          topo["elements/dims"].child(i).to_int64() + 1;
    }
    pack_array(coords["values"], topo_name + "_coords", args, array_code);
  }
  else if(topo_type == "unstructured")
  {
    const conduit::Node &elements = topo["elements"];

    pack_array(coords["values"], topo_name + "_coords", args, array_code);
    pack_array(elements["connectivity"],
               topo_name + "_connectivity",
               args,
               array_code);

    const std::string &shape = elements["shape"].as_string();
    if(shape == "polygonal")
    {
      pack_array(elements["sizes"], topo_name + "_sizes", args, array_code);
      pack_array(elements["offsets"], topo_name + "_offsets", args, array_code);
    }
    else if(shape == "polyhedral")
    {
      // TODO polyhedral needs to pack additional things
    }
    else
    {
      // single shape
      // no additional packing
    }
  }
}
// }}}

//-----------------------------------------------------------------------------
// -- FieldCode
//-----------------------------------------------------------------------------
// {{{
FieldCode::FieldCode(const std::string &field_name,
                     const std::string &association,
                     const std::shared_ptr<const TopologyCode> topo_code,
                     const ArrayCode &arrays,
                     const int num_components,
                     const int component)
    : field_name(field_name), association(association),
      num_components(num_components), component(component), array_code(arrays),
      topo_code(topo_code), math_code()
{
  if(association != "element" && association != "vertex")
  {
    ASCENT_ERROR("FieldCode: unknown association '" << association << "'.");
  }
}

// get the flat index from index_name[3]
// used for structured topologies
void
FieldCode::field_idx(InsertionOrderedSet<std::string> &code,
                     const std::string &index_name,
                     const std::string &association,
                     const std::string &res_name,
                     const bool declare) const
{
  std::string res;
  if(declare)
  {
    res += "const int ";
  }
  res += res_name + " = " + index_name + "[0]";
  if(topo_code->num_dims >= 2)
  {
    res += " + " + index_name + "[1] * (" + topo_code->topo_name + "_dims_i";
    if(association == "element")
    {
      res += " - 1";
    }
    res += ")";
  }
  if(topo_code->num_dims == 3)
  {
    res += " + " + index_name + "[2] * (" + topo_code->topo_name + "_dims_i";
    if(association == "element")
    {
      res += " - 1";
    }
    res += ") * (" + topo_code->topo_name + "_dims_j";
    if(association == "element")
    {
      res += " - 1";
    }
    res += ")";
  }
  res += ";\n";
  code.insert(res);
}

// field values at the vertices of an element
void
FieldCode::element_vertex_values(InsertionOrderedSet<std::string> &code,
                                 const std::string &res_name,
                                 const int component,
                                 const bool declare) const
{
  if(topo_code->topo_type == "unstructured")
  {
    topo_code->unstructured_vertices(code);
    if(topo_code->shape_size == -1)
    {
      // multiple shapes
      ASCENT_ERROR("element_vertex_values is not implemented for multi-shape "
                   "unstructured topologies");
      // TODO see unstructured_vertices
      return;
    }
  }
  else
  {
    topo_code->structured_vertices(code);
  }
  if(declare)
  {
    code.insert("double " + res_name + "[" +
                std::to_string(topo_code->shape_size) + "];\n");
  }
  // structured and single-shape unstructured use the same code
  for(int i = 0; i < topo_code->shape_size; ++i)
  {
    const std::string &vertex =
        array_code.index(topo_code->topo_name + "_vertices", std::to_string(i));
    code.insert(array_code.index(res_name, std::to_string(i)) + " = " +
                array_code.index(field_name, vertex, component) + ";\n");
  }
}

// https://github.com/visit-dav/visit/blob/f835d5132bdf7c6c8da09157ff86541290675a6f/src/avt/Expressions/General/avtGradientExpression.C#L1417
// gradient mapping : vtk mapping
// 1 : 0
// 2 : 1
// 3 : 2
// 0 : 3
void
FieldCode::quad_gradient(InsertionOrderedSet<std::string> &code,
                         const std::string &res_name) const
{
  // xi = .5 * (x[3] + x[0] - x[1] - x[2]);
  // xj = .5 * (x[0] + x[1] - x[2] - x[3]);

  // yi = .5 * (y[3] + y[0] - y[1] - y[2]);
  // yj = .5 * (y[0] + y[1] - y[2] - y[3]);

  // vi = .5 * (v[3] + v[0] - v[1] - v[2]);
  // vj = .5 * (v[0] + v[1] - v[2] - v[3]);
  const std::string vertex_locs = topo_code->topo_name + "_vertex_locs";
  const std::string vertices = topo_code->topo_name + "_vertices";
  const std::string vertex_values = res_name + "_vertex_values";
  element_vertex_values(code, vertex_values, component, true);
  code.insert(
      {"double " + res_name + "_x[3];\n",
       res_name + "_x[0] = .5 * (" + vertex_locs + "[3][0] + " + vertex_locs +
           "[0][0] - " + vertex_locs + "[1][0] - " + vertex_locs + "[2][0]);\n",
       res_name + "_x[1] = .5 * (" + vertex_locs + "[0][0] + " + vertex_locs +
           "[1][0] - " + vertex_locs + "[2][0] - " + vertex_locs + "[3][0]);\n",
       "double " + res_name + "_y[3];\n",
       res_name + "_y[0] = .5 * (" + vertex_locs + "[3][1] + " + vertex_locs +
           "[0][1] - " + vertex_locs + "[1][1] - " + vertex_locs + "[2][1]);\n",
       res_name + "_y[1] = .5 * (" + vertex_locs + "[0][1] + " + vertex_locs +
           "[1][1] - " + vertex_locs + "[2][1] - " + vertex_locs + "[3][1]);\n",
       "double " + res_name + "_v[3];\n",
       res_name + "_v[0] = .5 * (" + array_code.index(vertex_values, "3") +
           " + " + array_code.index(vertex_values, "0") + " - " +
           array_code.index(vertex_values, "1") + " - " +
           array_code.index(vertex_values, "2") + ");\n",
       res_name + "_v[1] = .5 * (" + array_code.index(vertex_values, "0") +
           " + " + array_code.index(vertex_values, "1") + " - " +
           array_code.index(vertex_values, "2") + " - " +
           array_code.index(vertex_values, "3") + ");\n"});
  math_code.determinant_2x2(
      code, res_name + "_x", res_name + "_y", res_name + "_area");
  code.insert("const double " + res_name + "_inv_vol = 1.0 / (tiny + " +
              res_name + "_area);\n");
  math_code.determinant_2x2(
      code, res_name + "_v", res_name + "_y", res_name + "[0]", false);
  code.insert(res_name + "[0] *= " + res_name + "_inv_vol;\n");
  math_code.determinant_2x2(
      code, res_name + "_x", res_name + "_v", res_name + "[1]", false);
  code.insert(res_name + "[1] *= " + res_name + "_inv_vol;\n");
  code.insert(res_name + "[2] = 0;\n");
}

// https://github.com/visit-dav/visit/blob/f835d5132bdf7c6c8da09157ff86541290675a6f/src/avt/Expressions/General/avtGradientExpression.C#L1511
// gradient mapping : vtk mapping
// 0 : 3
// 1 : 0
// 2 : 1
// 3 : 2
// 4 : 7
// 5 : 4
// 6 : 5
// 7 : 6
void
FieldCode::hex_gradient(InsertionOrderedSet<std::string> &code,
                        const std::string &res_name) const
{
  // assume vertex locations are populated (either structured or unstructured
  // hexes)
  // clang-format off
  // xi = .25 * ( (x[3] + x[0] + x[7] + x[4]) - (x[2] + x[1] + x[5] + x[6]) );
  // xj = .25 * ( (x[0] + x[1] + x[5] + x[4]) - (x[3] + x[2] + x[6] + x[7]) );
  // xk = .25 * ( (x[7] + x[4] + x[5] + x[6]) - (x[3] + x[0] + x[1] + x[2]) );

  // yi = .25 * ( (y[3] + y[0] + y[7] + y[4]) - (y[2] + y[1] + y[5] + y[6]) );
  // yj = .25 * ( (y[0] + y[1] + y[5] + y[4]) - (y[3] + y[2] + y[6] + y[7]) );
  // yk = .25 * ( (y[7] + y[4] + y[5] + y[6]) - (y[3] + y[0] + y[1] + y[2]) );

  // zi = .25 * ( (z[3] + z[0] + z[7] + z[4]) - (z[2] + z[1] + z[5] + z[6]) );
  // zj = .25 * ( (z[0] + z[1] + z[5] + z[4]) - (z[3] + z[2] + z[6] + z[7]) );
  // zk = .25 * ( (z[7] + z[4] + z[5] + z[6]) - (z[3] + z[0] + z[1] + z[2]) );

  // vi = .25 * ( (v[3] + v[0] + v[7] + v[4]) - (v[2] + v[1] + v[5] + v[6]) );
  // vj = .25 * ( (v[0] + v[1] + v[5] + v[4]) - (v[3] + v[2] + v[6] + v[7]) );
  // vk = .25 * ( (v[7] + v[4] + v[5] + v[6]) - (v[3] + v[0] + v[1] + v[2]) );
  // clang-format on
  const std::string vertex_locs = topo_code->topo_name + "_vertex_locs";
  const std::string vertices = topo_code->topo_name + "_vertices";
  const std::string vertex_values = res_name + "_vertex_values";
  element_vertex_values(code, vertex_values, component, true);
  code.insert({
      "double " + res_name + "_x[3];\n",
      res_name + "_x[0] = .25 * ( (" + vertex_locs + "[3][0] + " + vertex_locs +
          "[0][0] + " + vertex_locs + "[7][0] + " + vertex_locs +
          "[4][0]) - (" + vertex_locs + "[2][0] + " + vertex_locs +
          "[1][0] + " + vertex_locs + "[5][0] + " + vertex_locs +
          "[6][0]) );\n",
      res_name + "_x[1] = .25 * ( (" + vertex_locs + "[0][0] + " + vertex_locs +
          "[1][0] + " + vertex_locs + "[5][0] + " + vertex_locs +
          "[4][0]) - (" + vertex_locs + "[3][0] + " + vertex_locs +
          "[2][0] + " + vertex_locs + "[6][0] + " + vertex_locs +
          "[7][0]) );\n",
      res_name + "_x[2] = .25 * ( (" + vertex_locs + "[7][0] + " + vertex_locs +
          "[4][0] + " + vertex_locs + "[5][0] + " + vertex_locs +
          "[6][0]) - (" + vertex_locs + "[3][0] + " + vertex_locs +
          "[0][0] + " + vertex_locs + "[1][0] + " + vertex_locs +
          "[2][0]) );\n",
      "double " + res_name + "_y[3];\n",
      res_name + "_y[0] = .25 * ( (" + vertex_locs + "[3][1] + " + vertex_locs +
          "[0][1] + " + vertex_locs + "[7][1] + " + vertex_locs +
          "[4][1]) - (" + vertex_locs + "[2][1] + " + vertex_locs +
          "[1][1] + " + vertex_locs + "[5][1] + " + vertex_locs +
          "[6][1]) );\n",
      res_name + "_y[1] = .25 * ( (" + vertex_locs + "[0][1] + " + vertex_locs +
          "[1][1] + " + vertex_locs + "[5][1] + " + vertex_locs +
          "[4][1]) - (" + vertex_locs + "[3][1] + " + vertex_locs +
          "[2][1] + " + vertex_locs + "[6][1] + " + vertex_locs +
          "[7][1]) );\n",
      res_name + "_y[2] = .25 * ( (" + vertex_locs + "[7][1] + " + vertex_locs +
          "[4][1] + " + vertex_locs + "[5][1] + " + vertex_locs +
          "[6][1]) - (" + vertex_locs + "[3][1] + " + vertex_locs +
          "[0][1] + " + vertex_locs + "[1][1] + " + vertex_locs +
          "[2][1]) );\n",
      "double " + res_name + "_z[3];\n",
      res_name + "_z[0] = .25 * ( (" + vertex_locs + "[3][2] + " + vertex_locs +
          "[0][2] + " + vertex_locs + "[7][2] + " + vertex_locs +
          "[4][2]) - (" + vertex_locs + "[2][2] + " + vertex_locs +
          "[1][2] + " + vertex_locs + "[5][2] + " + vertex_locs +
          "[6][2]) );\n",
      "double " + res_name + "_z[3];\n",
      res_name + "_z[1] = .25 * ( (" + vertex_locs + "[0][2] + " + vertex_locs +
          "[1][2] + " + vertex_locs + "[5][2] + " + vertex_locs +
          "[4][2]) - (" + vertex_locs + "[3][2] + " + vertex_locs +
          "[2][2] + " + vertex_locs + "[6][2] + " + vertex_locs +
          "[7][2]) );\n",
      res_name + "_z[2] = .25 * ( (" + vertex_locs + "[7][2] + " + vertex_locs +
          "[4][2] + " + vertex_locs + "[5][2] + " + vertex_locs +
          "[6][2]) - (" + vertex_locs + "[3][2] + " + vertex_locs +
          "[0][2] + " + vertex_locs + "[1][2] + " + vertex_locs +
          "[2][2]) );\n",
      "double " + res_name + "_v[3];\n",
      res_name + "_v[0] = .25 * ( (" + array_code.index(vertex_values, "3") +
          " + " + array_code.index(vertex_values, "0") + " + " +
          array_code.index(vertex_values, "7") + " + " +
          array_code.index(vertex_values, "4") + ") - (" +
          array_code.index(vertex_values, "2") + " + " +
          array_code.index(vertex_values, "1") + " + " +
          array_code.index(vertex_values, "5") + " + " +
          array_code.index(vertex_values, "6") + ") );\n",
      res_name + "_v[1] = .25 * ( (" + array_code.index(vertex_values, "0") +
          " + " + array_code.index(vertex_values, "1") + " + " +
          array_code.index(vertex_values, "5") + " + " +
          array_code.index(vertex_values, "4") + ") - (" +
          array_code.index(vertex_values, "3") + " + " +
          array_code.index(vertex_values, "2") + " + " +
          array_code.index(vertex_values, "6") + " + " +
          array_code.index(vertex_values, "7") + ") );\n",
      res_name + "_v[2] = .25 * ( (" + array_code.index(vertex_values, "7") +
          " + " + array_code.index(vertex_values, "4") + " + " +
          array_code.index(vertex_values, "5") + " + " +
          array_code.index(vertex_values, "6") + ") - (" +
          array_code.index(vertex_values, "3") + " + " +
          array_code.index(vertex_values, "0") + " + " +
          array_code.index(vertex_values, "1") + " + " +
          array_code.index(vertex_values, "2") + ") );\n",
  });
  math_code.determinant_3x3(code,
                            res_name + "_x",
                            res_name + "_y",
                            res_name + "_z",
                            res_name + "_vol");
  code.insert("const double " + res_name + "_inv_vol = 1.0 / (tiny + " +
              res_name + "_vol);\n");
  math_code.determinant_3x3(code,
                            res_name + "_v",
                            res_name + "_y",
                            res_name + "_z",
                            res_name + "[0]",
                            false);
  code.insert(res_name + "[0] *= " + res_name + "_inv_vol;\n");
  math_code.determinant_3x3(code,
                            res_name + "_v",
                            res_name + "_z",
                            res_name + "_x",
                            res_name + "[1]",
                            false);
  code.insert(res_name + "[1] *= " + res_name + "_inv_vol;\n");
  math_code.determinant_3x3(code,
                            res_name + "_v",
                            res_name + "_x",
                            res_name + "_y",
                            res_name + "[2]",
                            false);
  code.insert(res_name + "[2] *= " + res_name + "_inv_vol;\n");
}

// if_body is executed if the target element/vertex (e.g. upper, lower, current)
// is within the mesh boundary otherwise else_body is executed
void
FieldCode::visit_current(InsertionOrderedSet<std::string> &code,
                         const std::string &index_name,
                         const std::string &if_body,
                         const std::string &else_body,
                         const int dim) const
{
  InsertionOrderedSet<std::string> if_code;
  if_code.insert({"if(" + index_name + "[" + std::to_string(dim) + "] > 0 && " +
                      index_name + "[" + std::to_string(dim) + "] < " +
                      topo_code->topo_name + "_dims_" +
                      std::string(1, 'i' + dim) +
                      (association == "element" ? " - 1" : "") + ")\n",
                  "{\n"});
  if_code.insert(if_body);
  if_code.insert("}\n");
  InsertionOrderedSet<std::string> else_code;
  if(!else_body.empty())
  {
    else_code.insert({"else\n", "{\n"});
    else_code.insert(else_body);
    else_code.insert("}\n");
  }
  code.insert(if_code.accumulate() + else_code.accumulate());
}

// visit_upper and visit_lower assume that the index_name is within the
// bounds of the mesh
void
FieldCode::visit_upper(InsertionOrderedSet<std::string> &code,
                       const std::string &index_name,
                       const std::string &if_body,
                       const std::string &else_body,
                       const int dim) const
{
  InsertionOrderedSet<std::string> if_code;
  if_code.insert({"if(" + index_name + "[" + std::to_string(dim) + "] < " +
                      topo_code->topo_name + "_dims_" +
                      std::string(1, 'i' + dim) + " - " +
                      (association == "element" ? "2" : "1") + ")\n",
                  "{\n"});
  if_code.insert(index_name + "[" + std::to_string(dim) + "] += 1;\n");
  if_code.insert(if_body);
  if_code.insert(index_name + "[" + std::to_string(dim) + "] -= 1;\n");
  if_code.insert("}\n");
  InsertionOrderedSet<std::string> else_code;
  if(!else_body.empty())
  {
    else_code.insert({"else\n", "{\n"});
    else_code.insert(else_body);
    else_code.insert("}\n");
  }
  code.insert(if_code.accumulate() + else_code.accumulate());
}

void
FieldCode::visit_lower(InsertionOrderedSet<std::string> &code,
                       const std::string &index_name,
                       const std::string &if_body,
                       const std::string &else_body,
                       const int dim) const
{
  InsertionOrderedSet<std::string> if_code;
  if_code.insert(
      {"if(" + index_name + "[" + std::to_string(dim) + "] > 0)\n", "{\n"});
  if_code.insert(index_name + "[" + std::to_string(dim) + "] -= 1;\n");
  if_code.insert(if_body);
  if_code.insert(index_name + "[" + std::to_string(dim) + "] += 1;\n");
  if_code.insert("}\n");
  InsertionOrderedSet<std::string> else_code;
  if(!else_body.empty())
  {
    else_code.insert({"else\n", "{\n"});
    else_code.insert(else_body);
    else_code.insert("}\n");
  }
  code.insert(if_code.accumulate() + else_code.accumulate());
}

void
FieldCode::gradient(InsertionOrderedSet<std::string> &code) const
{
  const std::string gradient_name =
      field_name + (component == -1 ? "" : "_" + std::to_string(component)) +
      "_gradient";
  code.insert("double " + gradient_name + "[3];\n");

  // handle hex and quad gradients elsewhere
  if(association == "vertex" && (topo_code->topo_type == "structured" ||
                                 topo_code->topo_type == "unstructured"))
  {
    code.insert("double " + gradient_name + "[3];\n");
    code.insert("const double tiny = 1.e-37;\n");
    if(topo_code->topo_type == "structured")
    {
      topo_code->structured_vertex_locs(code);
      if(topo_code->num_dims == 3)
      {
        hex_gradient(code, gradient_name);
      }
      else if(topo_code->num_dims == 2)
      {
        quad_gradient(code, gradient_name);
      }
      else
      {
        ASCENT_ERROR("Gradient is not implemented for 1D structured meshes.");
      }
    }
    else if(topo_code->topo_type == "unstructured")
    {
      topo_code->unstructured_vertex_locs(code);
      if(topo_code->shape == "hex")
      {
        hex_gradient(code, gradient_name);
      }
      else if(topo_code->shape == "quad")
      {
        quad_gradient(code, gradient_name);
      }
      else
      {
        ASCENT_ERROR("Gradient of unstructured vertex associated fields only "
                     "works on hex and quad shapes. The given shape was '"
                     << topo_code->shape << "'.");
      }
    }
    return;
  }

  // handle uniforma and rectilinear gradients
  if(topo_code->topo_type != "uniform" && topo_code->topo_type != "rectilinear")
  {
    ASCENT_ERROR("Unsupported topo_type: '"
                 << topo_code->topo_type
                 << "'. Gradient is not implemented for unstructured "
                    "topologies nor structured element associated fields.");
  }

  if(association == "element")
  {
    topo_code->element_idx(code);
  }
  else if(association == "vertex")
  {
    topo_code->vertex_idx(code);
  }
  const std::string index_name =
      topo_code->topo_name + "_" + association + "_idx";

  const std::string upper = gradient_name + "_upper";
  const std::string lower = gradient_name + "_lower";
  code.insert({"double " + upper + ";\n",
               "double " + lower + ";\n",
               "double " + upper + "_loc;\n",
               "double " + lower + "_loc;\n",
               "int " + upper + "_idx;\n",
               "int " + lower + "_idx;\n",
               "double " + gradient_name + "_delta;\n"});
  for(int i = 0; i < 3; ++i)
  {
    if(i < topo_code->num_dims)
    {
      // positive (upper) direction
      InsertionOrderedSet<std::string> upper_body;
      field_idx(upper_body, index_name, association, upper + "_idx", false);
      upper_body.insert(
          upper + " = " +
          array_code.index(field_name, upper + "_idx", component) + ";\n");
      if(association == "vertex")
      {
        topo_code->vertex_coord(upper_body,
                                std::string(1, 'x' + i),
                                index_name + "[" + std::to_string(i) + "]",
                                upper + "_loc",
                                false);
      }
      else
      {
        topo_code->element_coord(upper_body,
                                 std::string(1, 'x' + i),
                                 index_name + "[" + std::to_string(i) + "]",
                                 upper + "_loc",
                                 false);
      }
      const std::string upper_body_str = upper_body.accumulate();
      visit_upper(code, index_name, upper_body_str, upper_body_str, i);

      // negative (lower) direction
      InsertionOrderedSet<std::string> lower_body;
      field_idx(lower_body, index_name, association, lower + "_idx", false);
      lower_body.insert(
          lower + " = " +
          array_code.index(field_name, lower + "_idx", component) + ";\n");
      if(association == "vertex")
      {
        topo_code->vertex_coord(lower_body,
                                std::string(1, 'x' + i),
                                index_name + "[" + std::to_string(i) + "]",
                                lower + "_loc",
                                false);
      }
      else
      {
        topo_code->element_coord(lower_body,
                                 std::string(1, 'x' + i),
                                 index_name + "[" + std::to_string(i) + "]",
                                 lower + "_loc",
                                 false);
      }
      const std::string lower_body_str = lower_body.accumulate();
      visit_lower(code, index_name, lower_body_str, lower_body_str, i);

      // calculate delta
      code.insert(gradient_name + "_delta = " + upper + "_loc - " + lower +
                      "_loc;\n",
                  false);

      // calculate gradient
      code.insert(gradient_name + "[" + std::to_string(i) + "] = (" + upper +
                  " - " + lower + ") / " + gradient_name + "_delta;\n");
    }
    else
    {
      code.insert(gradient_name + "[" + std::to_string(i) + "] = 0;\n");
    }
  }
}

void
FieldCode::curl(InsertionOrderedSet<std::string> &code) const
{
  // assumes the gradient for each component is present (generated in
  // JitableFunctions::curl)
  const std::string curl_name = field_name + "_curl";
  code.insert("double " + curl_name + "[3];\n");
  if(num_components == 3)
  {

    code.insert({curl_name + "[0] = " + field_name + "_2_gradient[1] - " +
                     field_name + "_1_gradient[2];\n",
                 curl_name + "[1] = " + field_name + "_0_gradient[2] - " +
                     field_name + "_2_gradient[0];\n"});
  }
  else if(num_components == 2)
  {
    code.insert({curl_name + "[0] = 0;\n", curl_name + "[1] = 0;\n"});
  }
  code.insert(curl_name + "[2] = " + field_name + "_1_gradient[0] - " +
              field_name + "_0_gradient[1];\n");
}

// recursive function to run "body" for all elements surrounding a vertex in a
// structured topology
void
FieldCode::visit_vertex_elements(InsertionOrderedSet<std::string> &code,
                                 const std::string &index_name,
                                 const std::string &if_body,
                                 const std::string &else_body,
                                 const int dim) const
{
  if(topo_code->topo_type != "uniform" &&
     topo_code->topo_type != "rectilinear" &&
     topo_code->topo_type != "structured")
  {
    ASCENT_ERROR("Function visit_vertex_elements only works on uniform, "
                 "rectilinear, and structured topologies.");
  }
  if(dim > 0)
  {
    InsertionOrderedSet<std::string> lower_code;
    InsertionOrderedSet<std::string> upper_code;
    visit_lower(lower_code, index_name, if_body, else_body, dim - 1);
    visit_current(upper_code, index_name, if_body, else_body, dim - 1);
    visit_vertex_elements(
        code, index_name, upper_code.accumulate(), else_body, dim - 1);
    visit_vertex_elements(
        code, index_name, lower_code.accumulate(), else_body, dim - 1);
  }
  else
  {
    code.insert(if_body);
  }
}

void
FieldCode::recenter(InsertionOrderedSet<std::string> &code,
                    const std::string &target_association,
                    const std::string &res_name) const
{

  if(target_association == "element")
  {
    const std::string vertex_values = res_name + "_vertex_values";
    if(component == -1 && num_components > 1)
    {
      code.insert("double " + vertex_values + "[" +
                  std::to_string(num_components) + "][" +
                  std::to_string(topo_code->shape_size) + "];\n");
      for(int i = 0; i < num_components; ++i)
      {
        element_vertex_values(
            code, vertex_values + "[" + std::to_string(i) + "]", i, false);
        math_code.array_avg(code,
                            topo_code->shape_size,
                            vertex_values + "[" + std::to_string(i) + "]",
                            res_name,
                            true);
      }
    }
    else
    {
      element_vertex_values(code, vertex_values, component, true);
      math_code.array_avg(
          code, topo_code->shape_size, vertex_values, res_name, true);
    }
  }
  else
  {
    if(topo_code->topo_type == "uniform" ||
       topo_code->topo_type == "rectilinear" ||
       topo_code->topo_type == "structured")
    {
      topo_code->vertex_idx(code);
      const std::string index_name = topo_code->topo_name + "_vertex_idx";

      InsertionOrderedSet<std::string> if_body;
      InsertionOrderedSet<std::string> avg_code;
      if_body.insert(res_name + "_num_adj += 1;\n");
      field_idx(if_body, index_name, association, "field_idx", true);
      if(component == -1 && num_components > 1)
      {
        // prelude
        code.insert({"int " + res_name + "_num_adj = 0;\n",
                     "double " + res_name + "_sum[" +
                         std::to_string(num_components) + "];\n"});

        // declare res array
        avg_code.insert("double " + res_name + "[" +
                        std::to_string(num_components) + "];\n");
        for(int i = 0; i < num_components; ++i)
        {
          const std::string i_str = std::to_string(i);
          code.insert(res_name + "_sum[" + i_str + "] = 0;\n");

          // if-statement body
          if_body.insert(res_name + "_sum[" + i_str + "] += " +
                         array_code.index(field_name, "field_idx", i) + ";\n");

          // average to get result
          avg_code.insert(res_name + "[" + i_str + "] = " + res_name + "_sum[" +
                          i_str + "] / " + res_name + "_num_adj;\n");
        }
      }
      else
      {
        // prelude
        code.insert({"int " + res_name + "_num_adj = 0;\n",
                     "double " + res_name + "_sum = 0;\n"});

        // if-statement body
        if_body.insert(res_name + "_sum += " +
                       array_code.index(field_name, "field_idx", component) +
                       ";\n");

        // average to get result
        avg_code.insert("const double " + res_name + " = " + res_name +
                        "_sum / " + res_name + "_num_adj;\n");
      }

      visit_vertex_elements(
          code, index_name, if_body.accumulate(), "", topo_code->num_dims);

      code.insert(avg_code);
    }
    else
    {
      ASCENT_ERROR("Element to Vertex recenter is not implemented on "
                   "unstructured meshes.");
    }
  }
}

// }}}

//-----------------------------------------------------------------------------
// -- JitableFunctions
//-----------------------------------------------------------------------------
// {{{
JitableFunctions::JitableFunctions(
    const conduit::Node &params,
    const std::vector<const Jitable *> &input_jitables,
    const std::vector<const Kernel *> &input_kernels,
    const std::string &filter_name,
    const conduit::Node &dataset,
    const int dom_idx,
    const bool not_fused,
    Jitable &out_jitable,
    Kernel &out_kernel)
    : params(params), input_jitables(input_jitables),
      input_kernels(input_kernels), filter_name(filter_name), dataset(dataset),
      dom_idx(dom_idx), not_fused(not_fused), out_jitable(out_jitable),
      out_kernel(out_kernel), inputs(params["inputs"]),
      domain(dataset.child(dom_idx))
{
}

void
JitableFunctions::binary_op()
{
  if(not_fused)
  {
    const int lhs_port = inputs["lhs/port"].to_int32();
    const int rhs_port = inputs["rhs/port"].to_int32();
    const Kernel &lhs_kernel = *input_kernels[lhs_port];
    const Kernel &rhs_kernel = *input_kernels[rhs_port];
    // union the field/mesh vars
    out_kernel.fuse_kernel(lhs_kernel);
    out_kernel.fuse_kernel(rhs_kernel);
    const std::string lhs_expr = lhs_kernel.expr;
    const std::string rhs_expr = rhs_kernel.expr;
    const std::string &op_str = params["op_string"].as_string();
    if(lhs_kernel.num_components == 1 && rhs_kernel.num_components == 1)
    {
      // scalar ops
      if(op_str == "not")
      {
        out_kernel.expr = "!(" + rhs_expr + ")";
      }
      else
      {
        std::string occa_op_str;
        if(op_str == "and")
        {
          occa_op_str = "&&";
        }
        else if(op_str == "or")
        {
          occa_op_str = "||";
        }
        else
        {
          occa_op_str = op_str;
        }
        out_kernel.expr = "(" + lhs_expr + " " + op_str + " " + rhs_expr + ")";
      }
      out_kernel.num_components = 1;
    }
    else
    {
      // vector ops
      bool error = false;
      if(lhs_kernel.num_components == rhs_kernel.num_components)
      {
        if(op_str == "+")
        {
          MathCode().vector_add(out_kernel.for_body,
                                lhs_expr,
                                rhs_expr,
                                filter_name,
                                lhs_kernel.num_components);
        }
        else if(op_str == "-")
        {
          MathCode().vector_subtract(out_kernel.for_body,
                                     lhs_expr,
                                     rhs_expr,
                                     filter_name,
                                     lhs_kernel.num_components);
        }
        else if(op_str == "*")
        {
          MathCode().dot_product(out_kernel.for_body,
                                 lhs_expr,
                                 rhs_expr,
                                 filter_name,
                                 lhs_kernel.num_components);
        }
        else
        {
          error = true;
        }
        out_kernel.expr = filter_name;
        out_kernel.num_components = lhs_kernel.num_components;
      }
      else
      {
        error = true;
      }
      if(error)
      {
        ASCENT_ERROR("Unsupported binary_op: (field with "
                     << lhs_kernel.num_components << " components) " << op_str
                     << " (field with " << rhs_kernel.num_components
                     << " components).");
      }
    }
  }
  else
  {
    // kernel of this type has already been fused, do nothing on a
    // per-domain basis
  }
}

void
JitableFunctions::builtin_functions(const std::string &function_name)
{
  if(not_fused)
  {
    out_kernel.expr = function_name + "(";
    const int num_inputs = inputs.number_of_children();
    for(int i = 0; i < num_inputs; ++i)
    {
      const int port_num = inputs.child(i)["port"].to_int32();
      const Kernel &inp_kernel = *input_kernels[port_num];
      if(inp_kernel.num_components > 1)
      {
        ASCENT_ERROR("Built-in function '"
                     << function_name
                     << "' does not support vector fields with "
                     << inp_kernel.num_components << " components.");
      }
      const std::string &inp_expr = inp_kernel.expr;
      out_kernel.fuse_kernel(inp_kernel);
      if(i != 0)
      {
        out_kernel.expr += ", ";
      }
      out_kernel.expr += inp_expr;
    }
    out_kernel.expr += ")";
    out_kernel.num_components = 1;
  }
}

bool
available_component(const std::string &axis, const int num_axes)
{
  // if a field has only 1 component it doesn't have .x .y .z
  if((axis == "x" && num_axes >= 2) || (axis == "y" && num_axes >= 2) ||
     (axis == "z" && num_axes >= 3))
  {
    return true;
  }
  ASCENT_ERROR("Derived field with "
               << num_axes << " components does not have component '" << axis
               << "'.");
  return false;
}

bool
available_axis(const std::string &axis,
               const int num_axes,
               const std::string &topo_name)
{
  if(((axis == "x" || axis == "dx") && num_axes >= 1) ||
     ((axis == "y" || axis == "dy") && num_axes >= 2) ||
     ((axis == "z" || axis == "dz") && num_axes >= 3))
  {
    return true;
  }
  ASCENT_ERROR("Topology '" << topo_name << "' with " << num_axes
                            << " dimensions does not have axis '" << axis
                            << "'.");
  return false;
}

void
JitableFunctions::topo_attrs(const conduit::Node &obj, const std::string &name)
{
  const std::string &topo_name = obj["value"].as_string();
  std::unique_ptr<Topology> topo = topologyFactory(topo_name, domain);
  if(obj.has_path("attr"))
  {
    if(not_fused)
    {
      const conduit::Node &assoc = obj["attr"].child(0);
      TopologyCode topo_code =
          TopologyCode(topo_name, domain, out_jitable.arrays[dom_idx]);
      if(assoc.name() == "cell")
      {
        // x, y, z
        if(is_xyz(name) && available_axis(name, topo->num_dims, topo_name))
        {
          topo_code.element_coord(
              out_kernel.for_body, name, "", topo_name + "_cell_" + name);
          out_kernel.expr = topo_name + "_cell_" + name;
        }
        // dx, dy, dz
        else if(name[0] == 'd' && is_xyz(std::string(1, name[1])) &&
                available_axis(name, topo->num_dims, topo_name))
        {
          if(topo->topo_type == "uniform")
          {
            out_kernel.expr = topo_name + "_spacing_" + name;
          }
          else if(topo->topo_type == "rectilinear")
          {
            topo_code.dxdydz(out_kernel.for_body);
            out_kernel.expr = topo_name + "_" + name;
          }
          else
          {
            ASCENT_ERROR("Can only get dx, dy, dz for uniform or rectilinear "
                         "topologies, not topologies of type '"
                         << topo->topo_type << "'.");
          }
        }
        else if(name == "volume")
        {
          if(topo->num_dims != 3)
          {
            ASCENT_ERROR("Cell volume is only defined for topologies with 3 "
                         "dimensions. The specified topology '"
                         << topo->topo_name << "' has " << topo->num_dims
                         << " dimensions.");
          }
          topo_code.volume(out_kernel.for_body);
          out_kernel.expr = topo_name + "_volume";
        }
        else if(name == "area")
        {
          if(topo->num_dims < 2)
          {
            ASCENT_ERROR("Cell area is only defined for topologies at most 2 "
                         "dimensions. The specified topology '"
                         << topo->topo_name << "' has " << topo->num_dims
                         << " dimensions.");
          }
          topo_code.area(out_kernel.for_body);
          out_kernel.expr = topo_name + "_area";
        }
        else if(name == "surface_area")
        {
          if(topo->num_dims != 3)
          {
            ASCENT_ERROR(
                "Cell surface area is only defined for topologies with 3 "
                "dimensions. The specified topology '"
                << topo->topo_name << "' has " << topo->num_dims
                << " dimensions.");
          }
          topo_code.surface_area(out_kernel.for_body);
          out_kernel.expr = topo_name + "_surface_area";
        }
        else if(name == "id")
        {
          out_kernel.expr = "item";
        }
        else
        {
          ASCENT_ERROR("Could not find attribute '"
                       << name << "' of topo.cell at runtime.");
        }
      }
      else if(assoc.name() == "vertex")
      {
        if(is_xyz(name) && available_axis(name, topo->num_dims, topo_name))
        {
          topo_code.vertex_coord(
              out_kernel.for_body, name, "", topo_name + "_vertex_" + name);
          out_kernel.expr = topo_name + "_vertex_" + name;
        }
        else if(name == "id")
        {
          out_kernel.expr = "item";
        }
        else
        {
          ASCENT_ERROR("Could not find attribute '"
                       << name << "' of topo.vertex at runtime.");
        }
      }
      else
      {
        ASCENT_ERROR("Could not find attribute '" << assoc.name()
                                                  << "' of topo at runtime.");
      }
    }
  }
  else
  {
    if(name == "cell")
    {
      if(topo->topo_type == "points")
      {
        ASCENT_ERROR("Point topology '" << topo_name
                                        << "' has no cell attributes.");
      }
      out_jitable.dom_info.child(dom_idx)["entries"] = topo->get_num_cells();
      out_jitable.association = "element";
    }
    else
    {
      out_jitable.dom_info.child(dom_idx)["entries"] = topo->get_num_points();
      out_jitable.association = "vertex";
    }
    out_jitable.obj = obj;
    out_jitable.obj["attr/" + name];
  }
}

void
JitableFunctions::expr_dot()
{
  const int obj_port = inputs["obj/port"].as_int32();
  const Kernel &obj_kernel = *input_kernels[obj_port];
  const conduit::Node &obj = input_jitables[obj_port]->obj;
  const std::string &name = params["name"].as_string();
  // derived fields or trivial fields
  if(!obj.has_path("type") || obj["type"].as_string() == "field")
  {
    if(is_xyz(name) && available_component(name, obj_kernel.num_components))
    {
      out_kernel.expr =
          obj_kernel.expr + "[" + std::to_string(name[0] - 'x') + "]";
      out_kernel.num_components = 1;
    }
    else
    {

      ASCENT_ERROR("Could not find attribute '" << name
                                                << "' of field at runtime.");
    }
  }
  else if(obj["type"].as_string() == "topo")
  {
    // needs to run for every domain not just every kernel type to
    // populate entries
    topo_attrs(obj, name);
    // for now all topology attributes have one component :)
    out_kernel.num_components = 1;
  }
  else
  {
    ASCENT_ERROR("JIT: Unknown obj:\n" << obj.to_yaml());
  }
  if(not_fused)
  {
    out_kernel.fuse_kernel(obj_kernel);
  }
}

void
JitableFunctions::expr_if()
{
  if(not_fused)
  {
    const int condition_port = inputs["condition/port"].as_int32();
    const int if_port = inputs["if/port"].as_int32();
    const int else_port = inputs["else/port"].as_int32();
    const Kernel &condition_kernel = *input_kernels[condition_port];
    const Kernel &if_kernel = *input_kernels[if_port];
    const Kernel &else_kernel = *input_kernels[else_port];
    out_kernel.functions.insert(condition_kernel.functions);
    out_kernel.functions.insert(if_kernel.functions);
    out_kernel.functions.insert(else_kernel.functions);
    out_kernel.kernel_body.insert(condition_kernel.kernel_body);
    out_kernel.kernel_body.insert(if_kernel.kernel_body);
    out_kernel.kernel_body.insert(else_kernel.kernel_body);
    const std::string cond_name = filter_name + "_cond";
    const std::string res_name = filter_name + "_res";

    out_kernel.for_body.insert(condition_kernel.for_body);
    out_kernel.for_body.insert(
        condition_kernel.generate_output(cond_name, true));

    InsertionOrderedSet<std::string> if_else;
    if_else.insert("double " + res_name + ";\n");
    if_else.insert("if(" + cond_name + ")\n{\n");
    if_else.insert(if_kernel.for_body.accumulate() +
                   if_kernel.generate_output(res_name, false));
    if_else.insert("}\nelse\n{\n");
    if_else.insert(else_kernel.for_body.accumulate() +
                   else_kernel.generate_output(res_name, false));
    if_else.insert("}\n");

    out_kernel.for_body.insert(if_else.accumulate());
    out_kernel.expr = res_name;
    if(if_kernel.num_components != else_kernel.num_components)
    {
      ASCENT_ERROR("Jitable if-else: The if-branch results in "
                   << if_kernel.num_components
                   << " components and the else-branch results in "
                   << else_kernel.num_components
                   << " but they must have the same number of components.");
    }
    out_kernel.num_components = if_kernel.num_components;
  }
}

void
JitableFunctions::derived_field()
{
  // setting association and topology should run once for Jitable not for
  // each domain, but won't hurt
  if(inputs.has_path("assoc"))
  {
    const conduit::Node &string_obj =
        input_jitables[inputs["assoc/port"].as_int32()]->obj;
    const std::string &new_association = string_obj["value"].as_string();
    if(new_association != "vertex" && new_association != "element")
    {
      ASCENT_ERROR("derived_field: Unknown association '"
                   << new_association
                   << "'. Known associations are 'vertex' and 'element'.");
    }
    out_jitable.association = new_association;
  }
  if(inputs.has_path("topo"))
  {
    const conduit::Node &string_obj =
        input_jitables[inputs["topo/port"].as_int32()]->obj;
    const std::string &new_topology = string_obj["value"].as_string();
    // We repeat this check because we pass the topology name as a
    // string. If we pass a topo object it will get packed which is
    // unnecessary if we only want to output on the topology.
    if(!has_topology(dataset, new_topology))
    {
      std::set<std::string> topo_names = topology_names(dataset);
      std::string res;
      for(auto &name : topo_names)
      {
        res += name + " ";
      }
      ASCENT_ERROR(": dataset does not contain topology '"
                   << new_topology << "'"
                   << " known = " <<res);
    }
    if(!out_jitable.association.empty() && out_jitable.association != "none")
    {
      // update entries
      std::unique_ptr<Topology> topo = topologyFactory(new_topology, domain);
      conduit::Node &cur_dom_info = out_jitable.dom_info.child(dom_idx);
      int new_entries = 0;
      if(out_jitable.association == "vertex")
      {
        new_entries = topo->get_num_points();
      }
      else if(out_jitable.association == "element")
      {
        new_entries = topo->get_num_cells();
      }
      // ensure entries doesn't change if it's already defined
      if(cur_dom_info.has_child("entries"))
      {
        const int cur_entries = cur_dom_info["entries"].to_int32();
        if(new_entries != cur_entries)
        {
          ASCENT_ERROR(
              "derived_field: cannot put a derived field with "
              << cur_entries << " entries as a " << out_jitable.association
              << "-associated derived field on the topology '" << new_topology
              << "' since the resulting field would need to have "
              << new_entries << " entries.");
        }
      }
      else
      {
        cur_dom_info["entries"] = new_entries;
      }
    }
    out_jitable.topology = new_topology;
  }
  if(not_fused)
  {
    const int arg1_port = inputs["arg1/port"].as_int32();
    const Kernel &arg1_kernel = *input_kernels[arg1_port];
    out_kernel.fuse_kernel(arg1_kernel);
    out_kernel.expr = arg1_kernel.expr;
    out_kernel.num_components = arg1_kernel.num_components;
  }
}

// generate a temporary field on the device, used by things like gradient
// which have data dependencies that require the entire field to be present.
// essentially wrap field_kernel in a for loop
void
JitableFunctions::temporary_field(const Kernel &field_kernel,
                                  const std::string &field_name)
{
  // pass the value of entries for the temporary field
  const auto entries =
      out_jitable.dom_info.child(dom_idx)["entries"].to_int64();
  const std::string entries_name = filter_name + "_inp_entries";
  out_jitable.dom_info.child(dom_idx)["args/" + entries_name] = entries;

  // we will need to allocate a temporary array so make a schema for it and
  // put it in the array map
  // TODO for now temporary fields are interleaved
  conduit::Schema s;
  schemaFactory("interleaved",
                conduit::DataType::FLOAT64_ID,
                entries,
                detail::component_names(field_kernel.num_components),
                s);
  // The array will have to be allocated but doesn't point to any data so we
  // won't put it in args but it will still be passed in
  out_jitable.arrays[dom_idx].array_map.insert(
      std::make_pair(field_name, SchemaBool(s, false)));
  if(not_fused)
  {
    // not a regular kernel_fuse because we have to generate a for-loop and add
    // it to kernel_body instead of fusing for_body
    out_kernel.functions.insert(field_kernel.functions);
    out_kernel.kernel_body.insert(field_kernel.kernel_body);
    out_kernel.kernel_body.insert(field_kernel.generate_loop(
        field_name, out_jitable.arrays[dom_idx], entries_name));
  }
}

std::string
JitableFunctions::possible_temporary(const int field_port)
{
  const Jitable &field_jitable = *input_jitables[field_port];
  const Kernel &field_kernel = *input_kernels[field_port];
  const conduit::Node &obj = field_jitable.obj;
  std::string field_name;
  if(obj.has_path("value"))
  {
    field_name = obj["value"].as_string();
    out_kernel.fuse_kernel(field_kernel);
  }
  else
  {
    field_name = filter_name + "_inp";
    temporary_field(field_kernel, field_name);
  }
  return field_name;
}

void
JitableFunctions::gradient(const int field_port, const int component)
{
  const Kernel &field_kernel = *input_kernels[field_port];

  if(component == -1 && field_kernel.num_components > 1)
  {
    ASCENT_ERROR("gradient is only supported on scalar fields but a field with "
                 << field_kernel.num_components << " components was given.");
  }

  // association and topology should be the same for out_jitable and
  // field_jitable because we have already fused jitables at this point
  if(out_jitable.topology.empty() || out_jitable.topology == "none")
  {
    ASCENT_ERROR("Could not take the gradient of the derived field because the "
                 "associated topology could not be determined.");
  }
  if(out_jitable.association.empty() || out_jitable.association == "none")
  {
    ASCENT_ERROR("Could not take the gradient of the derived field "
                 "because the association could not be determined.");
  }
  std::unique_ptr<Topology> topo =
      topologyFactory(out_jitable.topology, domain);
  std::string field_name = possible_temporary(field_port);
  if((topo->topo_type == "structured" || topo->topo_type == "unstructured") &&
     out_jitable.association == "vertex")
  {
    // this does a vertex to cell gradient so update entries
    conduit::Node &n_entries = out_jitable.dom_info.child(dom_idx)["entries"];
    n_entries = topo->get_num_cells();
    out_jitable.association = "element";
  }
  if(not_fused)
  {
    const auto topo_code = std::make_shared<const TopologyCode>(
        topo->topo_name, domain, out_jitable.arrays[dom_idx]);
    FieldCode field_code = FieldCode(field_name,
                                     out_jitable.association,
                                     topo_code,
                                     out_jitable.arrays[dom_idx],
                                     1,
                                     component);
    field_code.gradient(out_kernel.for_body);
    out_kernel.expr = field_name +
                      (component == -1 ? "" : "_" + std::to_string(component)) +
                      "_gradient";
    out_kernel.num_components = 3;
  }
}

void
JitableFunctions::gradient()
{
  const int field_port = inputs["field/port"].as_int32();
  gradient(field_port, -1);
}

void
JitableFunctions::curl()
{
  const int field_port = inputs["field/port"].as_int32();
  const Kernel &field_kernel = *input_kernels[field_port];
  if(field_kernel.num_components < 2)
  {
    ASCENT_ERROR("Vorticity is only implemented for fields with at least 2 "
                 "components. The input field has "
                 << field_kernel.num_components << ".");
  }
  const std::string field_name = possible_temporary(field_port);
  // calling gradient here reuses the logic to update entries and association
  for(int i = 0; i < field_kernel.num_components; ++i)
  {
    gradient(field_port, i);
  }
  // TODO make it easier to construct FieldCode
  if(not_fused)
  {
    const auto topo_code = std::make_shared<const TopologyCode>(
        out_jitable.topology, domain, out_jitable.arrays[dom_idx]);
    FieldCode field_code = FieldCode(field_name,
                                     out_jitable.association,
                                     topo_code,
                                     out_jitable.arrays[dom_idx],
                                     field_kernel.num_components,
                                     -1);
    field_code.curl(out_kernel.for_body);
    out_kernel.expr = field_name + "_curl";
    out_kernel.num_components = field_kernel.num_components;
  }
}

void
JitableFunctions::recenter()
{
  if(not_fused)
  {
    const int field_port = inputs["field/port"].as_int32();
    const Kernel &field_kernel = *input_kernels[field_port];

    std::string mode;
    if(inputs.has_path("mode"))
    {
      const int mode_port = inputs["mode/port"].as_int32();
      const Jitable &mode_jitable = *input_jitables[mode_port];
      mode = mode_jitable.obj["value"].as_string();
      if(mode != "toggle" && mode != "vertex" && mode != "element")
      {
        ASCENT_ERROR("recenter: Unknown mode '"
                     << mode
                     << "'. Known modes are 'toggle', 'vertex', 'element'.");
      }
      if(out_jitable.association == mode)
      {
        ASCENT_ERROR("Recenter: The field is already "
                     << out_jitable.association
                     << " associated, redundant recenter.");
      }
    }
    else
    {
      mode = "toggle";
    }
    std::string target_association;
    if(mode == "toggle")
    {
      if(out_jitable.association == "vertex")
      {
        target_association = "element";
      }
      else
      {
        target_association = "vertex";
      }
    }
    else
    {
      target_association = mode;
    }

    const std::string field_name = possible_temporary(field_port);

    // update entries and association
    conduit::Node &n_entries = out_jitable.dom_info.child(dom_idx)["entries"];
    std::unique_ptr<Topology> topo =
        topologyFactory(out_jitable.topology, domain);
    if(target_association == "vertex")
    {
      n_entries = topo->get_num_points();
    }
    else
    {
      n_entries = topo->get_num_cells();
    }
    out_jitable.association = target_association;

    const auto topo_code = std::make_shared<const TopologyCode>(
        out_jitable.topology, domain, out_jitable.arrays[dom_idx]);
    FieldCode field_code = FieldCode(field_name,
                                     out_jitable.association,
                                     topo_code,
                                     out_jitable.arrays[dom_idx],
                                     field_kernel.num_components,
                                     -1);
    const std::string res_name = field_name + "_recenter_" + target_association;
    field_code.recenter(out_kernel.for_body, target_association, res_name);
    out_kernel.expr = res_name;
    out_kernel.num_components = field_kernel.num_components;
  }
}

void
JitableFunctions::magnitude()
{
  if(not_fused)
  {
    const int vector_port = inputs["vector/port"].as_int32();
    const Kernel &vector_kernel = *input_kernels[vector_port];
    if(vector_kernel.num_components <= 1)
    {
      ASCENT_ERROR("Cannot take the magnitude of a vector with "
                   << vector_kernel.num_components << " components.");
    }
    out_kernel.fuse_kernel(vector_kernel);
    MathCode().magnitude(out_kernel.for_body,
                         vector_kernel.expr,
                         filter_name,
                         vector_kernel.num_components);
    out_kernel.expr = filter_name;
    out_kernel.num_components = 1;
  }
}

void
JitableFunctions::vector()
{
  const int arg1_port = inputs["arg1/port"].to_int32();
  const int arg2_port = inputs["arg2/port"].to_int32();
  const int arg3_port = inputs["arg3/port"].to_int32();
  const Jitable &arg1_jitable = *input_jitables[arg1_port];
  const Jitable &arg2_jitable = *input_jitables[arg2_port];
  const Jitable &arg3_jitable = *input_jitables[arg3_port];
  // if all the inputs to the vector are "trivial" fields then we don't need
  // to regenerate the vector in a separate for-loop
  if(arg1_jitable.obj.has_path("type") && arg2_jitable.obj.has_path("type") &&
     arg3_jitable.obj.has_path("type"))
  {
    // We construct a fake schema with the input arrays as the components
    const std::string &arg1_field = arg1_jitable.obj["value"].as_string();
    const std::string &arg2_field = arg2_jitable.obj["value"].as_string();
    const std::string &arg3_field = arg3_jitable.obj["value"].as_string();
    std::unordered_map<std::string, SchemaBool> &array_map =
        out_jitable.arrays[dom_idx].array_map;
    conduit::Schema s;
    s[arg1_field].set(array_map.at(arg1_field).schema);
    s[arg2_field].set(array_map.at(arg2_field).schema);
    s[arg3_field].set(array_map.at(arg3_field).schema);
    array_map.insert(std::make_pair(filter_name, SchemaBool(s, true)));
    out_jitable.obj["value"] = filter_name;
    out_jitable.obj["type"] = "field";
  }

  if(not_fused)
  {
    const Kernel &arg1_kernel = *input_kernels[arg1_port];
    const Kernel &arg2_kernel = *input_kernels[arg2_port];
    const Kernel &arg3_kernel = *input_kernels[arg3_port];
    if(arg1_kernel.num_components != 1 || arg2_kernel.num_components != 1 ||
       arg2_kernel.num_components != 1)
    {
      ASCENT_ERROR("Vector arguments must all have exactly one component.");
    }
    out_kernel.fuse_kernel(arg1_kernel);
    out_kernel.fuse_kernel(arg2_kernel);
    out_kernel.fuse_kernel(arg3_kernel);
    const std::string arg1_expr = arg1_kernel.expr;
    const std::string arg2_expr = arg2_kernel.expr;
    const std::string arg3_expr = arg3_kernel.expr;
    out_kernel.for_body.insert({"double " + filter_name + "[3];\n",
                                filter_name + "[0] = " + arg1_expr + ";\n",
                                filter_name + "[1] = " + arg2_expr + ";\n",
                                filter_name + "[2] = " + arg3_expr + ";\n"});
    out_kernel.expr = filter_name;
    out_kernel.num_components = 3;
  }
}

void
JitableFunctions::binning_value(const conduit::Node &binning)
{
  // bin lookup functions
  // clang-format off
  const std::string rectilinear_bin =
    "int\n"
    "rectilinear_bin(const double value,\n"
    "                const double *const bins_begin,\n"
    "                const int len,\n"
    "                const bool clamp)\n"
    "{\n"
      // implements std::upper_bound
      "int mid;\n"
      "int low = 0;\n"
      "int high = len;\n"
      "while(low < high)\n"
      "{\n"
        "mid = (low + high) / 2;\n"
        "if(value >= bins_begin[mid])\n"
        "{\n"
          "low = mid + 1;\n"
        "}\n"
        "else\n"
        "{\n"
          "high = mid;\n"
        "}\n"
      "}\n"

      "if(clamp)\n"
      "{\n"
        "if(low <= 0)\n"
        "{\n"
          "return 0;\n"
        "}\n"
        "else if(low >= len)\n"
        "{\n"
          "return len - 2;\n"
        "}\n"
      "}\n"
      "else if(low <= 0 || low >= len)\n"
      "{\n"
        "return -1;\n"
      "}\n"
      "return low - 1;\n"
    "}\n\n";
  const std::string uniform_bin =
    "int\n"
    "uniform_bin(const double value,\n"
    "            const double min_val,\n"
    "            const double max_val,\n"
    "            const int num_bins,\n"
    "            const bool clamp)\n"
    "{\n"
      "const double inv_delta = num_bins / (max_val - min_val);\n"
      "const int bin_index = (int)((value - min_val) * inv_delta);\n"
      "if(clamp)\n"
      "{\n"
        "if(bin_index < 0)\n"
        "{\n"
          "return 0;\n"
        "}\n"
        "else if(bin_index >= num_bins)\n"
        "{\n"
          "return num_bins - 1;\n"
        "}\n"
      "}\n"
      "else if(bin_index < 0 || bin_index >= num_bins)\n"
      "{\n"
        "return -1;\n"
      "}\n"
      "return bin_index;\n"
    "}\n\n";
  // clang-format on
  //---------------------------------------------------------------------------

  // assume the necessary fields have been packed and are present in all
  // domains

  // get the passed association
  std::string assoc_str_;
  if(inputs.has_path("assoc"))
  {
    const conduit::Node &assoc_obj =
        input_jitables[inputs["assoc/port"].as_int32()]->obj;
    assoc_str_ = assoc_obj["value"].as_string();
  }

  const conduit::Node &bin_axes = binning["attrs/bin_axes/value"];
  std::vector<std::string> axis_names = bin_axes.child_names();

  // set/verify out_jitable.topology and out_jitable.association
  const conduit::Node &topo_and_assoc =
      final_topo_and_assoc(dataset, bin_axes, out_jitable.topology, assoc_str_);
  std::string assoc_str = topo_and_assoc["assoc_str"].as_string();
  if(assoc_str.empty())
  {
    // use the association from the binning
    assoc_str = binning["attrs/association/value"].as_string();
  }
  out_jitable.association = assoc_str;

  // set entries based on out_jitable.topology and out_jitable.association
  std::unique_ptr<Topology> topo =
      topologyFactory(out_jitable.topology, domain);
  conduit::Node &n_entries = out_jitable.dom_info.child(dom_idx)["entries"];
  if(out_jitable.association == "vertex")
  {
    n_entries = topo->get_num_points();
  }
  else if(out_jitable.association == "element")
  {
    n_entries = topo->get_num_cells();
  }

  if(not_fused)
  {
    const TopologyCode topo_code =
        TopologyCode(topo->topo_name, domain, out_jitable.arrays[dom_idx]);
    const std::string &binning_name = inputs["binning/filter_name"].as_string();
    InsertionOrderedSet<std::string> &code = out_kernel.for_body;
    const int num_axes = bin_axes.number_of_children();
    bool used_uniform = false;
    bool used_rectilinear = false;

    code.insert("int " + filter_name + "_home = 0;\n");
    code.insert("int " + filter_name + "_stride = 1;\n");
    for(int axis_index = 0; axis_index < num_axes; ++axis_index)
    {
      const conduit::Node &axis = bin_axes.child(axis_index);
      const std::string axis_name = axis.name();
      const std::string axis_prefix = binning_name + "_" + axis_name + "_";
      // find the value associated with the axis for the current item
      std::string axis_value;
      if(domain.has_path("fields/" + axis_name))
      {
        axis_value = axis_name + "_item";
        code.insert("const double " + axis_value + " = " +
                    out_jitable.arrays[dom_idx].index(axis_name, "item") +
                    ";\n");
      }
      else if(is_xyz(axis_name))
      {
        if(out_jitable.association == "vertex")
        {
          axis_value = topo->topo_name + "_vertex_" + axis_name;
          topo_code.vertex_coord(code, axis_name, "", axis_value);
        }
        else if(out_jitable.association == "element")
        {
          axis_value = topo->topo_name + "_cell_" + axis_name;
          topo_code.element_coord(code, axis_name, "", axis_value);
        }
      }

      size_t stride_multiplier;
      if(axis.has_path("num_bins"))
      {
        // uniform axis
        stride_multiplier = bin_axes.child(axis_index)["num_bins"].as_int32();

        // find the value's index in the axis
        if(!used_uniform)
        {
          used_uniform = true;
          out_kernel.functions.insert(uniform_bin);
        }
        code.insert("int " + axis_prefix + "bin_index = uniform_bin(" +
                    axis_value + ", " + axis_prefix + "min_val, " +
                    axis_prefix + "max_val, " + axis_prefix + "num_bins, " +
                    axis_prefix + "clamp);\n");
      }
      else
      {
        // rectilinear axis
        stride_multiplier =
            bin_axes.child(axis_index)["bins"].dtype().number_of_elements() - 1;

        // find the value's index in the axis
        if(!used_rectilinear)
        {
          used_rectilinear = true;
          out_kernel.functions.insert(rectilinear_bin);
        }
        code.insert("int " + axis_prefix + "bin_index = rectilinear_bin(" +
                    axis_value + ", " + axis_prefix + "bins, " + axis_prefix +
                    "bins_len, " + axis_prefix + "clamp);\n");
      }

      // update the current item's home
      code.insert("if(" + axis_prefix + "bin_index != -1 && " + filter_name +
                  "_home != -1)\n{\n" + filter_name + "_home += " +
                  axis_prefix + "bin_index * " + filter_name + "_stride;\n}\n");
      // update stride
      code.insert(filter_name +
                  "_stride *= " + std::to_string(stride_multiplier) + ";\n");
    }

    // get the value at home
    std::string default_value;
    if(inputs.has_path("default_value"))
    {
      default_value = inputs["default_value/filter_name"].as_string();
    }
    else
    {
      default_value = "0";
    }
    code.insert({"double " + filter_name + ";\n",
                 "if(" + filter_name + "_home != -1)\n{\n" + filter_name +
                     " = " + binning_name + "_value[" + filter_name +
                     "_home];\n}\nelse\n{\n" + filter_name + " = " +
                     default_value + ";\n}\n"});
    out_kernel.expr = filter_name;
    out_kernel.num_components = 1;
  }
}

void
JitableFunctions::rand()
{
  out_jitable.dom_info.child(dom_idx)["args/" + filter_name + "_seed"] =
      time(nullptr);
  if(not_fused)
  {
    // clang-format off
    const std::string halton =
      "double rand(int i)\n"
			"{\n"
				"const int b = 2;\n"
				"double f = 1;\n"
				"double r = 0;\n"
				"while(i > 0)\n"
				"{\n"
					"f = f / b;\n"
					"r = r + f * (i \% b);\n"
					"i = i / b;\n"
				"}\n"
				"return r;\n"
			"}\n\n";
    // clang-format on
    out_kernel.functions.insert(halton);
    out_kernel.expr = "rand(item + " + filter_name + "_seed)";
    out_kernel.num_components = 1;
  }
}
// }}}

//-----------------------------------------------------------------------------
// -- JitExecutionPolicy
//-----------------------------------------------------------------------------
//{{{

JitExecutionPolicy::JitExecutionPolicy()
{
}

// fuse policy
bool
FusePolicy::should_execute(const Jitable &jitable) const
{
  return false;
}

// unique name
std::string
FusePolicy::get_name() const
{
  return "fuse";
}

// roundtrip policy
bool
RoundtripPolicy::should_execute(const Jitable &jitable) const
{
  return jitable.can_execute();
}

std::string
RoundtripPolicy::get_name() const
{
  return "roundtrip";
}

// Used when we need to execute (e.g. for a top-level JitFilter or a JitFilter
// feeding into a reduction)
bool
AlwaysExecutePolicy::should_execute(const Jitable &jitable) const
{
  return true;
}

std::string
AlwaysExecutePolicy::get_name() const
{
  return "always_execute";
}

//}}}

//-----------------------------------------------------------------------------
// -- Kernel
//-----------------------------------------------------------------------------
// {{{
void
Kernel::fuse_kernel(const Kernel &from)
{
  functions.insert(from.functions);
  kernel_body.insert(from.kernel_body);
  for_body.insert(from.for_body);
}

// copy expr into a variable (scalar or vector) "output"
std::string
Kernel::generate_output(const std::string &output, bool declare) const
{
  std::string res;
  if(declare)
  {
    res += "double " + output;
    if(num_components > 1)
    {
      res += "[" + std::to_string(num_components) + "]";
    }
    res += ";\n";
  }
  if(num_components > 1)
  {
    for(int i = 0; i < num_components; ++i)
    {
      res += output + "[" + std::to_string(i) + "] = " + expr + "[" +
             std::to_string(i) + "];\n";
    }
  }
  else
  {
    res += output + " = " + expr + ";\n";
  }
  return res;
}

// generate a loop to set expr into the array "output"
std::string
Kernel::generate_loop(const std::string &output,
                      const ArrayCode &array_code,
                      const std::string &entries_name) const
{
  // clang-format off
  std::string res =
    "for (int group = 0; group < " + entries_name + "; group += 128; @outer)\n"
       "{\n"
         "for (int item = group; item < (group + 128); ++item; @inner)\n"
         "{\n"
           "if (item < " + entries_name + ")\n"
           "{\n" +
              for_body.accumulate();
              if(num_components > 1)
              {
                for(int i = 0; i < num_components; ++i)
                {
                  res += array_code.index(output, "item", i) + " = " + expr +
                    "[" + std::to_string(i) + "];\n";
                }
              }
              else
              {
                res += array_code.index(output, "item") + " = " + expr + ";\n";
              }
  res +=
           "}\n"
         "}\n"
       "}\n";
  return res;
  // clang-format on
}
// }}}

//-----------------------------------------------------------------------------
// -- Jitable
//-----------------------------------------------------------------------------
// {{{
// I pass in args because I want execute to generate new args and also be
// const so it can't just update the object's args
std::string
Jitable::generate_kernel(const int dom_idx, const conduit::Node &args) const
{
  const conduit::Node &cur_dom_info = dom_info.child(dom_idx);
  const Kernel &kernel = kernels.at(cur_dom_info["kernel_type"].as_string());
  std::string kernel_string;
  kernel_string += kernel.functions.accumulate();
  kernel_string += "@kernel void map(";
  const int num_args = args.number_of_children();
  bool first = true;
  for(int i = 0; i < num_args; ++i)
  {
    const conduit::Node &arg = args.child(i);
    std::string type;
    // array type was already determined in device_alloc_array
    if(!arg.has_path("index"))
    {
      type = "const " + detail::type_string(arg.dtype()) + " ";
    }
    if(!first)
    {
      kernel_string += "                 ";
    }
    kernel_string += type + arg.name() + (i == num_args - 1 ? ")\n{\n" : ",\n");
    first = false;
  }
  kernel_string += kernel.kernel_body.accumulate();
  kernel_string += kernel.generate_loop("output", arrays[dom_idx], "entries");
  kernel_string += "}";
  return detail::indent_code(kernel_string, 0);
}

void
Jitable::fuse_vars(const Jitable &from)
{
  // none is set when we try to fuse kernels with different topologies or
  // associations. This allows the expression to have multiple topologies but
  // we'll need a way of figuring out where to output things can't infer it
  // anymore. The derived_field() function can be used to explicitly specify
  // the topology and association.
  if(!from.topology.empty())
  {
    if(topology.empty())
    {
      topology = from.topology;
    }
    else if(topology != from.topology)
    {
      topology = "none";
    }
  }

  if(!from.association.empty())
  {
    if(association.empty())
    {
      association = from.association;
    }
    else if(association != from.association)
    {
      association = "none";
    }
  }

  int num_domains = from.dom_info.number_of_children();
  for(int dom_idx = 0; dom_idx < num_domains; ++dom_idx)
  {
    // fuse entries, ensure they are the same
    const conduit::Node &src_dom_info = from.dom_info.child(dom_idx);
    conduit::Node &dest_dom_info = dom_info.child(dom_idx);
    if(src_dom_info.has_path("entries"))
    {
      if(dest_dom_info.has_path("entries"))
      {
        if(dest_dom_info["entries"].to_int64() !=
           src_dom_info["entries"].to_int64())
        {
          ASCENT_ERROR("JIT: Failed to fuse kernels due to an incompatible "
                       "number of entries: "
                       << dest_dom_info["entries"].to_int64() << " versus "
                       << src_dom_info["entries"].to_int64());
        }
      }
      else
      {
        dest_dom_info["entries"] = src_dom_info["entries"];
      }
    }

    // copy kernel_type
    // This will be overwritten in all cases except when func="execute" in
    // JitFilter::execute
    dest_dom_info["kernel_type"] = src_dom_info["kernel_type"];

    // fuse args, set_external arrays and copy other arguments
    if(src_dom_info.has_path("args"))
    {
      conduit::NodeConstIterator arg_itr = src_dom_info["args"].children();
      while(arg_itr.has_next())
      {
        const conduit::Node &arg = arg_itr.next();
        conduit::Node &dest_args = dest_dom_info["args"];
        if(!dest_args.has_path(arg.name()))
        {
          if(arg.number_of_children() != 0 ||
             arg.dtype().number_of_elements() > 1)
          {
            dest_args[arg.name()].set_external(arg);
          }
          else
          {
            dest_args[arg.name()].set(arg);
          }
        }
      }
    }

    // fuse array_code
    for(size_t i = 0; i < from.arrays.size(); ++i)
    {
      arrays[i].array_map.insert(from.arrays[i].array_map.begin(),
                                 from.arrays[i].array_map.end());
    }
  }
}

bool
Jitable::can_execute() const
{
  return !(topology.empty() || topology == "none") &&
         !(association.empty() || association == "none") &&
         !kernels.begin()->second.expr.empty();
}

//-----------------------------------------------------------------------------
// How to Debug OCCA Kernels with LLDB
//-----------------------------------------------------------------------------
// 1. occa::setDevice("mode: 'Serial'");
// 2. export CXXFLAGS="-g" OCCA_VERBOSE=1
//    OCCA_CUDA_COMPILER_FLAGS
//    OCCA_CXXFLAGS
// 3. Run ascent (e.g. ./tests/ascent/t_ascent_derived)
// 4. Occa will print the path to the kernel binaries
//    (e.g. ~/.occa/cache/e1da5a95477a48db/build)
// 5. Run lldb on the kernel binary
//    (e.g. lldb ~/.occa/cache/e1da5a95477a48db/build)
// 6. In lldb: 'image lookup -r -F map'
//    assuming the occa kernel is named 'map'
// 7. Copy that function name and quit lldb
//    (e.g. "::map(const int &, const double *, const double &, double *)")
// 8  lldb ./tests/ascent/t_ascent_derived
// 9. break at the function name found above and run
//    (e.g. "b ::map(const int &, const double *, const double &, double
//    *)")

// we put the field on the mesh when calling execute and delete it later if
// it's an intermediate field

void
Jitable::execute(conduit::Node &dataset, const std::string &field_name)
{
  ASCENT_DATA_OPEN("jitable_execute");
  // TODO set this during initialization not here
  static bool device_set = false;
  if(!device_set)
  {
    // running this in a loop segfaults...
#ifdef ASCENT_CUDA_ENABLED
    // TODO get the right device_id
    occa::setDevice("mode: 'CUDA', device_id: 0");
#elif defined(ASCENT_USE_OPENMP)
    occa::setDevice("mode: 'OpenMP'");
#else
    occa::setDevice("mode: 'Serial'");
#endif
    device_set = true;
  }
  occa::device &device = occa::getDevice();
  ASCENT_DATA_ADD("occa device", device.mode());
  occa::kernel occa_kernel;

  // we need an association and topo so we can put the field back on the mesh
  if(topology.empty() || topology == "none")
  {
    ASCENT_ERROR("Error while executing derived field: Could not infer the "
                 "topology. Try using the derived_field function to set it "
                 "explicitely.");
  }
  if(association.empty() || association == "none")
  {
    ASCENT_ERROR("Error while executing derived field: Could not determine the "
                 "association. Try using the derived_field function to set it "
                 "explicitely.");
  }

  const int num_domains = dataset.number_of_children();
  for(int dom_idx = 0; dom_idx < num_domains; ++dom_idx)
  {
    ASCENT_DATA_OPEN("domain execute");
    conduit::Node &dom = dataset.child(dom_idx);

    conduit::Node &cur_dom_info = dom_info.child(dom_idx);

    const Kernel &kernel = kernels.at(cur_dom_info["kernel_type"].as_string());

    if(kernel.expr.empty())
    {
      ASCENT_ERROR("Cannot compile a kernel with an empty expr field. This "
                   "shouldn't happen, call someone.");
    }

    // the final number of entries
    const int entries = cur_dom_info["entries"].to_int64();

    // pass entries into args just before we need to execute
    cur_dom_info["args/entries"] = entries;

    // create output array schema and put it in array_map
    conduit::Schema output_schema;

    // TODO output to the host is always interleaved
    schemaFactory("interleaved",
                  conduit::DataType::FLOAT64_ID,
                  entries,
                  detail::component_names(kernel.num_components),
                  output_schema);

    arrays[dom_idx].array_map.insert(
        std::make_pair("output", SchemaBool(output_schema, false)));

    // allocate the output array in conduit
    conduit::Node &n_output = dom["fields/" + field_name];
    n_output["association"] = association;
    n_output["topology"] = topology;

    ASCENT_DATA_OPEN("host output alloc");
    n_output["values"].set(output_schema);
    unsigned char *output_ptr =
        static_cast<unsigned char *>(n_output["values"].data_ptr());
    // output to the host will always be compact
    ASCENT_DATA_ADD("bytes", output_schema.total_bytes_compact());
    ASCENT_DATA_CLOSE();

    // these are reference counted
    // need to keep the mem in scope or bad things happen
    std::vector<Array<unsigned char>> array_buffers;
    // slice is {index in array_buffers, offset, size}
    std::vector<detail::slice_t> slices;
    ASCENT_DATA_OPEN("host array alloc");
    // allocate arrays
    size_t output_index;
    conduit::Node new_args;
    for(const auto &array : arrays[dom_idx].array_map)
    {
      if(array.second.codegen_array)
      {
        // codegen_arrays are false arrays used by the codegen
        continue;
      }
      if(cur_dom_info["args"].has_path(array.first))
      {
        detail::device_alloc_array(cur_dom_info["args/" + array.first],
                                   array.second.schema,
                                   new_args,
                                   array_buffers,
                                   slices);
      }
      else
      {
        // not in args so doesn't point to any data, allocate a temporary
        if(array.first == "output")
        {
          output_index = array_buffers.size();
        }
        if(array.first == "output" &&
           (device.mode() == "Serial" || device.mode() == "OpenMP"))
        {
          // in Serial and OpenMP we don't need a separate output array for
          // the device, so just pass it conduit's array
          detail::device_alloc_temporary(array.first,
                                         array.second.schema,
                                         new_args,
                                         array_buffers,
                                         slices,
                                         output_ptr);
        }
        else
        {
          detail::device_alloc_temporary(array.first,
                                         array.second.schema,
                                         new_args,
                                         array_buffers,
                                         slices,
                                         nullptr);
        }
      }
    }
    // copy the non-array types to new_args
    const int original_num_args = cur_dom_info["args"].number_of_children();
    for(int i = 0; i < original_num_args; ++i)
    {
      const conduit::Node &arg = cur_dom_info["args"].child(i);
      if(arg.dtype().number_of_elements() == 1 &&
         arg.number_of_children() == 0 && !arg.dtype().is_string())
      {
        new_args[arg.name()] = arg;
      }
    }
    ASCENT_DATA_CLOSE();

    // generate and compile the kernel
    const std::string kernel_string = generate_kernel(dom_idx, new_args);

    // std::cout << kernel_string << std::endl;

    // store kernels so that we don't have to recompile, even loading a cached
    // kernel from disk is slow
    static std::unordered_map<std::string, occa::kernel> kernel_map;
    try
    {
      flow::Timer kernel_compile_timer;
      auto kernel_it = kernel_map.find(kernel_string);
      if(kernel_it == kernel_map.end())
      {
        occa_kernel = device.buildKernelFromString(kernel_string, "map");
        kernel_map[kernel_string] = occa_kernel;
      }
      else
      {
        occa_kernel = kernel_it->second;
      }
      ASCENT_DATA_ADD("kernel compile", kernel_compile_timer.elapsed());
    }
    catch(const occa::exception &e)
    {
      ASCENT_ERROR("Jitable: Expression compilation failed:\n"
                   << e.what() << "\n\n"
                   << kernel_string);
    }
    catch(...)
    {
      ASCENT_ERROR("Jitable: Expression compilation failed with an unknown "
                   "error.\n\n"
                   << kernel_string);
    }

    // pass input arguments
    occa_kernel.clearArgs();
    // get occa mem for devices
    std::vector<occa::memory> array_memories;
    detail::get_occa_mem(array_buffers, slices, array_memories);

    flow::Timer push_args_timer;
    const int num_new_args = new_args.number_of_children();
    for(int i = 0; i < num_new_args; ++i)
    {
      const conduit::Node &arg = new_args.child(i);
      if(arg.dtype().is_integer())
      {
        occa_kernel.pushArg(arg.to_int64());
      }
      else if(arg.dtype().is_float64())
      {
        occa_kernel.pushArg(arg.to_float64());
      }
      else if(arg.dtype().is_float32())
      {
        occa_kernel.pushArg(arg.to_float32());
      }
      else if(arg.has_path("index"))
      {
        occa_kernel.pushArg(array_memories[arg["index"].to_int32()]);
      }
      else
      {
        ASCENT_ERROR("JIT: Unknown argument type of argument: " << arg.name());
      }
    }
    ASCENT_DATA_ADD("push_input_args", push_args_timer.elapsed());

    flow::Timer kernel_run_timer;
    //std::cout<<"Running kernel "<<kernel_string<<"\n";
    occa_kernel.run();
    ASCENT_DATA_ADD("kernel runtime", kernel_run_timer.elapsed());

    // copy back
    flow::Timer copy_back_timer;
    if(device.mode() != "Serial" && device.mode() != "OpenMP")
    {
      array_memories[output_index].copyTo(output_ptr);
    }
    ASCENT_DATA_ADD("copy to host", copy_back_timer.elapsed());

    // dom["fields/" + field_name].print();
    ASCENT_DATA_CLOSE();
  }
  ASCENT_DATA_CLOSE();
}
// }}}
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
