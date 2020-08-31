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

#include <flow.hpp>

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

void
get_occa_mem(std::vector<Array<unsigned char>> &buffers,
             std::vector<occa::memory> &occa)
{
  flow::Timer device_array_timer;
  occa::device &device = occa::getDevice();
  ASCENT_DATA_ADD("occa device ", device.mode());
  const std::string mode = device.mode();

  // I think the valid modes are: "Serial" "OpenMP", "CUDA:
  const size_t size = buffers.size();
  occa.resize(size);

  for(size_t i = 0; i < size; ++i)
  {
    size_t buff_size = buffers[i].size();
    if(mode == "Serial" || mode == "OpenMP")
    {
      unsigned char *ptr = buffers[i].get_host_ptr();
      occa[i] =
          occa::cpu::wrapMemory(device, ptr, buff_size * sizeof(unsigned char));
    }
#ifdef ASCENT_CUDA_ENABLED
    else if(mode == "CUDA")
    {
      unsigned char *ptr = buffers[i].get_device_ptr();
      occa[i] = occa::cuda::wrapMemory(
          device, ptr, buff_size * sizeof(unsigned char));
    }
#endif
    else
    {
      ASCENT_ERROR("Unknow occa mode " << mode);
    }
  }

  ASCENT_DATA_ADD("copy to device", device_array_timer.elapsed());
}

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
// -- Array Allocation Functions
//-----------------------------------------------------------------------------
//{{{
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

    src_array.info().print();
    dest_array.info().print();
  }
}

// temporaries will always be one chunk of memory
void
device_alloc_temporary(const std::string &array_name,
                       const conduit::Schema &dest_schema,
                       conduit::Node &args,
                       std::vector<Array<unsigned char>> &array_memories)
{
  flow::Timer device_array_timer;
  Array<unsigned char> mem;
  mem.resize(dest_schema.total_bytes_compact());
  // occa::memory mem = device.malloc(dest_schema.total_bytes_compact());
  array_memories.push_back(mem);
  if(dest_schema.number_of_children() == 0)
  {
    const std::string param =
        detail::type_string(dest_schema.dtype()) + " *" + array_name;
    args[param + "/index"] = array_memories.size() - 1;
  }
  else
  {
    for(const std::string &component : dest_schema.child_names())
    {
      const std::string param =
          detail::type_string(dest_schema[component].dtype()) + " *" +
          array_name + "_" + component;
      args[param + "/index"] = array_memories.size() - 1;
    }
  }
  ASCENT_DATA_ADD("temp array bytes", dest_schema.total_bytes_compact());
  ASCENT_DATA_ADD("temp allocation time", device_array_timer.elapsed());
}

void
device_alloc_array(const conduit::Node &array,
                   const conduit::Schema &dest_schema,
                   conduit::Node &args,
                   std::vector<Array<unsigned char>> &array_memories)
{
  flow::Timer array_timer;
  flow::Timer host_array_timer;
  conduit::Node res_array;
  host_realloc_array(array, dest_schema, res_array);
  ASCENT_DATA_ADD("host array reallocation time", host_array_timer.elapsed());
  flow::Timer device_array_timer;
  if(array.number_of_children() == 0)
  {
    unsigned char *start_ptr =
        static_cast<unsigned char *>(const_cast<void *>(res_array.data_ptr()));

    Array<unsigned char> mem;
    mem.set(start_ptr, res_array.total_bytes_compact());
    // occa::memory mem =
    //    device.malloc(res_array.total_bytes_compact(), start_ptr);

    const std::string param =
        detail::type_string(array.dtype()) + " *" + array.name();
    args[param + "/index"] = array_memories.size();
    array_memories.push_back(mem);
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
        // occa::memory mem = device.malloc(
        //    full_region_it->end - full_region_it->start,
        //    full_region_it->start);
        full_region_it->index = array_memories.size();
        array_memories.push_back(mem);
      }
      const std::string param = detail::type_string(n_component.dtype()) +
                                " *" + array.name() + "_" + component;
      // TODO should make a slice, push it and use that to support cases where
      // we have multiple pointers inside one allocation
      args[param + "/index"] = full_region_it->index;
    }
  }
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
    // array is not in the map meaning it was created in the kernel and is
    // interleaved
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
    const int num_components = array_it->second.number_of_children();
    if(component == -1)
    {
      // check that the array only has one component
      if(num_components != 0)
      {
        ASCENT_ERROR("ArrayCode could not get the index of array '"
                     << array_name << "' because it has " << num_components
                     << " components and no component was specified.");
      }
      offset = array_it->second.dtype().offset();
      stride = array_it->second.dtype().stride();
      pointer_size = array_it->second.dtype().element_bytes();

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
          array_it->second.child(component);
      offset = component_schema.dtype().offset();
      stride = component_schema.dtype().stride();
      pointer_size = component_schema.dtype().element_bytes();

      pointer_name = array_name + "_" + component_schema.name();
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
  return index(array_name, idx, array_it->second.child_index(component));
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
        this->shape_size = detail::get_num_vertices(shape);
      }
      else
      {
        this->shape_size = -1;
      }
    }
    else
    {
      // single shape
      this->shape_size = detail::get_num_vertices(shape);
    }
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
  if(topo_type != "structured")
  {
    ASCENT_ERROR("The function structured_vertices only supports structured "
                 "topologies.");
  }
  element_idx(code);

  // vertex indices
  code.insert("int " + topo_name + "_vertices[" +
              std::to_string(static_cast<int>(std::pow(2, num_dims))) + "];\n");
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

  // locations
  code.insert("double " + topo_name + "_vertex_locs[" +
              std::to_string(static_cast<int>(std::pow(2, num_dims))) + "][" +
              std::to_string(num_dims) + "];\n");
  vertex_xyz(code,
             topo_name + "_vertices[0]",
             false,
             topo_name + "_vertex_locs[0]",
             false);
  vertex_xyz(code,
             topo_name + "_vertices[1]",
             false,
             topo_name + "_vertex_locs[1]",
             false);
  if(num_dims >= 2)
  {
    vertex_xyz(code,
               topo_name + "_vertices[2]",
               false,
               topo_name + "_vertex_locs[2]",
               false);
    vertex_xyz(code,
               topo_name + "_vertices[3]",
               false,
               topo_name + "_vertex_locs[3]",
               false);
  }
  if(num_dims == 3)
  {
    vertex_xyz(code,
               topo_name + "_vertices[4]",
               false,
               topo_name + "_vertex_locs[4]",
               false);
    vertex_xyz(code,
               topo_name + "_vertices[5]",
               false,
               topo_name + "_vertex_locs[5]",
               false);
    vertex_xyz(code,
               topo_name + "_vertices[6]",
               false,
               topo_name + "_vertex_locs[6]",
               false);
    vertex_xyz(code,
               topo_name + "_vertices[7]",
               false,
               topo_name + "_vertex_locs[7]",
               false);
  }
}

// TODO generate vertices array
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
               topo_name + "_vertex_locs[i]",
               false);
    for_loop.insert("}\n");
    code.insert(for_loop.accumulate());
  }
  else
  {
    // single shape
    // inline the for-loop
    code.insert("int " + topo_name + "_vertices[" +
                std::to_string(shape_size) + "];\n");
    for(int i = 0; i < shape_size; ++i)
    {
      code.insert(topo_name + "_vertices[" + std::to_string(i) +
                  "] = " + topo_name + "_connectivity[" + index_name + " * " +
                  std::to_string(shape_size) + " + " + std::to_string(i) +
                  "];\n");
    }
    code.insert("double " + topo_name + "_vertex_locs[" +
                std::to_string(shape_size) + "][" + std::to_string(num_dims) +
                "];\n");
    for(int i = 0; i < shape_size; ++i)
    {
      vertex_xyz(code,
                 array_code.index(topo_name + "_vertices", std::to_string(i)),
                 false,
                 topo_name + "_vertex_locs[" + std::to_string(i) + "]",
                 false);
    }
  }
}

// average value of a component given an array of vectors
void
component_avg(InsertionOrderedSet<std::string> &code,
              const int length,
              const std::string &array_name,
              const std::string &coord,
              const std::string &res_name,
              const bool declare)
{
  const int component = coord[0] - 'x';
  std::stringstream vert_avg;
  vert_avg << "(";
  for(int j = 0; j < length; ++j)
  {
    if(j != 0)
    {
      vert_avg << " + ";
    }
    vert_avg << array_name + "[" << j << "][" << component << "]";
  }
  vert_avg << ") / " << length;
  code.insert((declare ? "const double " : "") + res_name + " = " +
              vert_avg.str() + ";\n");
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
        topo_name + "element_locs[" + std::to_string(coord[0] - 'x') + "]";
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
    structured_vertices(code);
    component_avg(code,
                  std::pow(2, num_dims),
                  topo_name + "_vertex_locs",
                  coord,
                  res_name,
                  declare);
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertices(code);
    if(shape_size == -1)
    {
      // multiple shapes
      // This will generate 3 for loops if we want to calculate element_xyz
      // If this is an issue we can make a special case for it in
      // element_xyz
      InsertionOrderedSet<std::string> for_loop;
      for_loop.insert(
          {"for(int i = 0; i < " + topo_name + "_shape_size; ++i)\n", "{\n"});
      component_avg(for_loop,
                    std::pow(2, num_dims),
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
        component_avg(code,
                      std::pow(2, num_dims),
                      topo_name + "_vertex_locs",
                      coord,
                      res_name,
                      declare);
        code.insert("}\n");
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
  unstructured_vertices(for_loop,
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
    structured_vertices(code);
    hexahedral_volume(code, topo_name + "_vertex_locs", topo_name + "_volume");
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertices(code);
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
    structured_vertices(code);
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
    unstructured_vertices(code);
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
    structured_vertices(code);
    hexahedral_surface_area(
        code, topo_name + "_vertex_locs", topo_name + "_area");
  }
  else if(topo_type == "unstructured")
  {
    unstructured_vertices(code);
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
// can copy from it later.
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
    array_code.array_map[name] = array.schema();
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
    // TODO for now it's always contiguous
    schemaFactory("contiguous", type_id, size, array.child_names(), s);
    array_code.array_map[name] = s;
  }
}

// TODO a lot of duplicated code from TopologyCode and Topology
void
pack_topology(const std::string &topo_name,
              const conduit::Node &domain,
              conduit::Node &args,
              ArrayCode &array)
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
    pack_array(coords["values"], topo_name + "_coords", args, array);
  }
  else if(topo_type == "structured")
  {
    for(size_t i = 0; i < num_dims; ++i)
    {
      const std::string dim = std::string(1, 'i' + i);
      args[topo_name + "_dims_" + dim] =
          topo["elements/dims"].child(i).to_int64() + 1;
    }
    pack_array(coords["values"], topo_name + "_coords", args, array);
  }
  else if(topo_type == "unstructured")
  {
    pack_array(coords["values"], topo_name + "_coords", args, array);
    pack_array(topo["elements/connectivity"],
               topo_name + "_connectivity",
               args,
               array);

    // TODO polygonal and polyhedral need to pack additional things
    // if(shape == "polygonal")
    // {
    //   args[topo_name + "_sizes"].set_external(sizes);
    //   args[topo_name + "_offsets"].set_external(offsets);
    // }
    // else if(shape == "polyhedral")
    // {
    //   args[topo_name + "_polyhedral_sizes"].set_external(polyhedral_sizes);
    //   args[topo_name +
    //   "_polyhedral_offsets"].set_external(polyhedral_offsets);
    //   args[topo_name
    //   + "_polyhedral_connectivity"].set_external(
    //       polyhedral_connectivity);

    //   args[topo_name + "_sizes"].set_external(sizes);
    //   args[topo_name + "_offsets"].set_external(offsets);
    //   if(polyhedral_shape != "polygonal")
    //   {
    //     args[topo_name + "_shape_size"] = polyhedral_shape_size;
    //   }
    // }
    // else
    // {
    // single shape
    // }
  }
}
// }}}

//-----------------------------------------------------------------------------
// -- FieldCode
//-----------------------------------------------------------------------------
// {{{
FieldCode::FieldCode(const std::string &field_name,
                     const std::string &association,
                     TopologyCode &&topo_code,
                     const ArrayCode &arrays,
                     const int num_components,
                     const int component)
    : field_name(field_name), association(association),
      num_components(num_components), component(component), array_code(arrays),
      topo_code(topo_code), math_code()
{
}

// get the flat index from index_name[3]
void
FieldCode::field_idx(InsertionOrderedSet<std::string> &code,
                     const std::string &index_name,
                     const std::string &res_name,
                     const std::string &association,
                     const bool declare)
{
  std::string res;
  if(declare)
  {
    res += "const double ";
  }
  res += res_name + " = " + index_name + "[0]";
  if(topo_code.num_dims >= 2)
  {
    res += " + " + index_name + "[1] * (" + topo_code.topo_name + "_dims_i";
    if(association == "element")
    {
      res += " - 1";
    }
    res += ")";
  }
  if(topo_code.num_dims == 3)
  {
    res += " + " + index_name + "[2] * (" + topo_code.topo_name + "_dims_i";
    if(association == "element")
    {
      res += " - 1";
    }
    res += ") * (" + topo_code.topo_name + "_dims_j";
    if(association == "element")
    {
      res += " - 1";
    }
    res += ")";
  }
  res += ";\n";
  code.insert(res);
}

// Calculate the element associated gradient of a vertex associated field on
// a quadrilateral mesh
// https://github.com/visit-dav/visit/blob/f835d5132bdf7c6c8da09157ff86541290675a6f/src/avt/Expressions/General/avtGradientExpression.C#L1417
// gradient mapping : vtk mapping
// 1 : 0
// 2 : 1
// 3 : 2
// 0 : 3
void
FieldCode::quad_gradient(InsertionOrderedSet<std::string> &code,
                         const std::string &res_name)
{
  // xi = .5 * (x[3] + x[0] - x[1] - x[2]);
  // xj = .5 * (x[0] + x[1] - x[2] - x[3]);

  // yi = .5 * (y[3] + y[0] - y[1] - y[2]);
  // yj = .5 * (y[0] + y[1] - y[2] - y[3]);

  // vi = .5 * (v[3] + v[0] - v[1] - v[2]);
  // vj = .5 * (v[0] + v[1] - v[2] - v[3]);
  const std::string vertex_locs = topo_code.topo_name + "_vertex_locs";
  const std::string vertices = topo_code.topo_name + "_vertices";
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
       res_name + "_v[0] = .5 * (" +
           array_code.index(field_name, vertices + "[3]", component) + " + " +
           array_code.index(field_name, vertices + "[0]", component) + " - " +
           array_code.index(field_name, vertices + "[1]", component) + " - " +
           array_code.index(field_name, vertices + "[2]", component) + ");\n",
       res_name + "_v[1] = .5 * (" +
           array_code.index(field_name, vertices + "[0]", component) + " + " +
           array_code.index(field_name, vertices + "[1]", component) + " - " +
           array_code.index(field_name, vertices + "[2]", component) + " - " +
           array_code.index(field_name, vertices + "[3]", component) + ");\n"});
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

// Calculate the element associated gradient of a vertex associated field on
// a hexahedral mesh
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
                        const std::string &res_name)
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
  const std::string vertex_locs = topo_code.topo_name + "_vertex_locs";
  const std::string vertices = topo_code.topo_name + "_vertices";
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
      res_name + "_v[0] = .25 * ( (" +
          array_code.index(field_name, vertices + "[3]", component) + " + " +
          array_code.index(field_name, vertices + "[0]", component) + " + " +
          array_code.index(field_name, vertices + "[7]", component) + " + " +
          array_code.index(field_name, vertices + "[4]", component) + ") - (" +
          array_code.index(field_name, vertices + "[2]", component) + " + " +
          array_code.index(field_name, vertices + "[1]", component) + " + " +
          array_code.index(field_name, vertices + "[5]", component) + " + " +
          array_code.index(field_name, vertices + "[6]", component) + ") );\n",
      res_name + "_v[1] = .25 * ( (" +
          array_code.index(field_name, vertices + "[0]", component) + " + " +
          array_code.index(field_name, vertices + "[1]", component) + " + " +
          array_code.index(field_name, vertices + "[5]", component) + " + " +
          array_code.index(field_name, vertices + "[4]", component) + ") - (" +
          array_code.index(field_name, vertices + "[3]", component) + " + " +
          array_code.index(field_name, vertices + "[2]", component) + " + " +
          array_code.index(field_name, vertices + "[6]", component) + " + " +
          array_code.index(field_name, vertices + "[7]", component) + ") );\n",
      res_name + "_v[2] = .25 * ( (" +
          array_code.index(field_name, vertices + "[7]", component) + " + " +
          array_code.index(field_name, vertices + "[4]", component) + " + " +
          array_code.index(field_name, vertices + "[5]", component) + " + " +
          array_code.index(field_name, vertices + "[6]", component) + ") - (" +
          array_code.index(field_name, vertices + "[3]", component) + " + " +
          array_code.index(field_name, vertices + "[0]", component) + " + " +
          array_code.index(field_name, vertices + "[1]", component) + " + " +
          array_code.index(field_name, vertices + "[2]", component) + ") );\n",
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

void
FieldCode::gradient(InsertionOrderedSet<std::string> &code)
{
  const std::string gradient_name =
      field_name + (component == -1 ? "" : "_" + std::to_string(component)) +
      "_gradient";
  code.insert("double " + gradient_name + "[3];\n");

  // handle hex and quad gradients elsewhere
  if(association == "vertex" && (topo_code.topo_type == "structured" ||
                                 topo_code.topo_type == "unstructured"))
  {
    code.insert("double " + gradient_name + "[3];\n");
    code.insert("const double tiny = 1.e-37;\n");
    if(topo_code.topo_type == "structured")
    {
      topo_code.structured_vertices(code);
      if(topo_code.num_dims == 3)
      {
        hex_gradient(code, gradient_name);
      }
      else if(topo_code.num_dims == 2)
      {
        quad_gradient(code, gradient_name);
      }
      else
      {
        ASCENT_ERROR("Gradient is not implemented for 1D structured meshes.");
      }
    }
    else if(topo_code.topo_type == "unstructured")
    {
      topo_code.unstructured_vertices(code);
      if(topo_code.shape == "hex")
      {
        hex_gradient(code, gradient_name);
      }
      else if(topo_code.shape == "quad")
      {
        quad_gradient(code, gradient_name);
      }
      else
      {
        ASCENT_ERROR("Gradient of unstructured vertex associated fields only "
                     "works on hex and quad shapes. The given shape was '"
                     << topo_code.shape << "'.");
      }
    }
    return;
  }

  // handle uniforma and rectilinear gradients
  if(topo_code.topo_type != "uniform" && topo_code.topo_type != "rectilinear")
  {
    ASCENT_ERROR("Unsupported topo_type: '"
                 << topo_code.topo_type
                 << "'. Gradient is not implemented for unstructured "
                    "topologies nor structured element associated fields.");
  }

  if(association == "element")
  {
    topo_code.element_idx(code);
  }
  else if(association == "vertex")
  {
    topo_code.vertex_idx(code);
  }
  else
  {
    ASCENT_ERROR("Gradient: unknown association: '" << association << "'.");
  }

  const std::string index_name =
      topo_code.topo_name + "_" + association + "_idx";
  const std::string upper = gradient_name + "_upper";
  const std::string lower = gradient_name + "_lower";
  code.insert({"double " + upper + ";\n",
               "double " + lower + ";\n",
               "double " + upper + "_loc;\n",
               "double " + lower + "_loc;\n",
               "int " + upper + "_idx;\n",
               "int " + lower + "_idx;\n",
               "double " + gradient_name + "_delta;\n"});
  if(topo_code.topo_type == "rectilinear")
  {
    code.insert({"double " + upper + "_loc;\n", "double " + lower + "_loc;\n"});
  }
  for(int i = 0; i < 3; ++i)
  {
    if(i < topo_code.num_dims)
    {
      // positive (upper) direction
      InsertionOrderedSet<std::string> u_if_code;
      u_if_code.insert({"if(" + index_name + "[" + std::to_string(i) + "] < " +
                            topo_code.topo_name + "_dims_" +
                            std::string(1, 'i' + i) + " - " +
                            (association == "element" ? "2" : "1") + ")\n",
                        "{\n"});
      u_if_code.insert(index_name + "[" + std::to_string(i) + "] += 1;\n");

      InsertionOrderedSet<std::string> upper_body;
      field_idx(upper_body, index_name, upper + "_idx", association, false);
      upper_body.insert(
          upper + " = " +
          array_code.index(field_name, upper + "_idx", component) + ";\n");
      if(association == "vertex")
      {
        topo_code.vertex_coord(upper_body,
                               std::string(1, 'x' + i),
                               index_name + "[" + std::to_string(i) + "]",
                               upper + "_loc",
                               false);
      }
      else
      {
        topo_code.element_coord(upper_body,
                                std::string(1, 'x' + i),
                                index_name + "[" + std::to_string(i) + "]",
                                upper + "_loc",
                                false);
      }
      const std::string upper_body_str = upper_body.accumulate();

      u_if_code.insert(upper_body_str);

      u_if_code.insert(index_name + "[" + std::to_string(i) + "] -= 1;\n");
      u_if_code.insert("}\n");
      InsertionOrderedSet<std::string> p_else_code;
      p_else_code.insert({"else\n", "{\n"});

      p_else_code.insert(upper_body_str);

      p_else_code.insert("}\n");
      code.insert(u_if_code.accumulate() + p_else_code.accumulate());

      // negative (lower) direction
      InsertionOrderedSet<std::string> l_if_code;
      l_if_code.insert(
          {"if(" + index_name + "[" + std::to_string(i) + "] > 0)\n", "{\n"});
      l_if_code.insert(index_name + "[" + std::to_string(i) + "] -= 1;\n");

      InsertionOrderedSet<std::string> lower_body;
      field_idx(lower_body, index_name, lower + "_idx", association, false);
      lower_body.insert(
          lower + " = " +
          array_code.index(field_name, lower + "_idx", component) + ";\n");
      if(association == "vertex")
      {
        topo_code.vertex_coord(lower_body,
                               std::string(1, 'x' + i),
                               index_name + "[" + std::to_string(i) + "]",
                               lower + "_loc",
                               false);
      }
      else
      {
        topo_code.element_coord(lower_body,
                                std::string(1, 'x' + i),
                                index_name + "[" + std::to_string(i) + "]",
                                lower + "_loc",
                                false);
      }
      const std::string lower_body_str = lower_body.accumulate();
      l_if_code.insert(lower_body_str);

      l_if_code.insert(index_name + "[" + std::to_string(i) + "] += 1;\n");
      l_if_code.insert("}\n");
      InsertionOrderedSet<std::string> n_else_code;
      n_else_code.insert({"else\n", "{\n"});

      n_else_code.insert(lower_body_str);

      n_else_code.insert("}\n");
      code.insert(l_if_code.accumulate() + n_else_code.accumulate());

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
FieldCode::vorticity(InsertionOrderedSet<std::string> &code)
{
  // assumes the gradient for each component is present (generated in
  // JitableFunctions::vorticity)
  const std::string vorticity_name = field_name + "_vorticity";
  code.insert({
      "double " + vorticity_name + "[" + std::to_string(num_components) +
          "];\n",
      vorticity_name + "[0] = " + field_name + "_2_gradient[1] - " +
          field_name + "_1_gradient[2];\n",
      vorticity_name + "[1] = " + field_name + "_0_gradient[2] - " +
          field_name + "_2_gradient[0];\n",
      vorticity_name + "[2] = " + field_name + "_1_gradient[0] - " +
          field_name + "_0_gradient[1];\n",
  });
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
      // generate the new expression string (main line of code)
      out_kernel.expr = "(" + lhs_expr + " " + op_str + " " + rhs_expr + ")";
      out_kernel.num_components = 1;
    }
    else
    {
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
        ASCENT_ERROR("Unsupported binary_op: field["
                     << lhs_kernel.num_components << "] " << op_str << " field["
                     << rhs_kernel.num_components << "].");
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
            ASCENT_ERROR("Can only get dx, dy, dz for uniform or rectiilnear "
                         "topologies not topologies of type '"
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
  if(!obj.has_path("type"))
  {
    // field objects
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
  else
  {
    // all other objects
    if(obj["type"].as_string() == "topo")
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
  if(inputs.has_path("association"))
  {
    const conduit::Node &string_obj =
        input_jitables[inputs["association/port"].as_int32()]->obj;
    const std::string &new_association = string_obj["value"].as_string();
    if(new_association != "vertex" && new_association != "element")
    {
      ASCENT_ERROR("derived_field: Unknown association '"
                   << new_association
                   << "'. Known associations are 'vertex' and 'element'.");
    }
    out_jitable.association = new_association;
  }
  if(inputs.has_path("topology"))
  {
    const conduit::Node &string_obj =
        input_jitables[inputs["topology/port"].as_int32()]->obj;
    const std::string &new_topology = string_obj["value"].as_string();
    // We repeat this check because we pass the topology name as a
    // string. If we pass a topo object it will get packed which is
    // unnecessary if we only want to output on the topology.
    if(!has_topology(dataset, new_topology))
    {
      ASCENT_ERROR(": dataset does not contain topology '"
                   << new_topology << "'"
                   << " known = " << known_topos(dataset));
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
  }
}

// generate a temporary field on the device, used by things like gradient which
// have data dependencies that require the entire field to be present
void
JitableFunctions::temporary_field(const Kernel &field_kernel)
{
  // pass the value of entries for the temporary field
  const auto entries =
      out_jitable.dom_info.child(dom_idx)["entries"].to_int64();
  const std::string entries_name = filter_name + "_entries";
  out_jitable.dom_info.child(dom_idx)["args/" + entries_name] = entries;

  // we will need to allocate a temporary array so make a schema for it and put
  // it in the array map
  const std::string tmp_field = filter_name + "_inp";
  conduit::Schema s;
  schemaFactory("interleaved",
                conduit::DataType::FLOAT64_ID,
                entries,
                {"x", "y", "z"},
                s);
  // The array will have to be allocated but doesn't point to any data so we
  // won't put it in args
  out_jitable.arrays[dom_idx].array_map[tmp_field] = s;
  if(not_fused)
  {
    out_kernel.kernel_body.insert(field_kernel.generate_loop(
        tmp_field, out_jitable.arrays[dom_idx], filter_name + "_entries"));
  }
}

void
JitableFunctions::gradient(const Jitable &field_jitable,
                           const Kernel &field_kernel,
                           const std::string &input_field,
                           const int component)
{
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
  conduit::Node &args = out_jitable.dom_info.child(dom_idx)["args"];
  // gradient needs to pack the topology
  std::unique_ptr<Topology> topo =
      topologyFactory(field_jitable.topology, domain);
  pack_topology(
      field_jitable.topology, domain, args, out_jitable.arrays[dom_idx]);
  // we need to change entries for each domain if we're doing a vertex to
  // element gradient
  std::string my_input_field;
  if(input_field.empty())
  {
    // need to generate a temporary field
    temporary_field(field_kernel);
    my_input_field = filter_name + "_inp";
  }
  else
  {
    my_input_field = input_field;
  }
  if((topo->topo_type == "structured" || topo->topo_type == "unstructured") && field_jitable.association == "vertex")
  {
    // this does a vertex to cell gradient so update entries
    conduit::Node &n_entries = out_jitable.dom_info.child(dom_idx)["entries"];
    n_entries = topo->get_num_cells();
    out_jitable.association = "element";
  }
  if(not_fused)
  {
    // vectors use the intereleaved memory layout because it is easier to
    // pass "gradient[i]" to a function that takes in a vector and
    // have it concatentate "[0]" to access the first element resulting in
    // "gradient[i][0]" rather than "gradient[0][i]" where the "[0]" would
    // have to be inserted between "gradient" and "[i]"

    // generate a new derived field if the field we're taking the gradient of
    // wasn't originally on the dataset
    TopologyCode topo_code =
        TopologyCode(topo->topo_name, domain, out_jitable.arrays[dom_idx]);
    FieldCode field_code = FieldCode(my_input_field,
                                     field_jitable.association,
                                     std::move(topo_code),
                                     out_jitable.arrays[dom_idx],
                                     1,
                                     component);
    field_code.gradient(out_kernel.for_body);
    out_kernel.expr = my_input_field +
                      (component == -1 ? "" : "_" + std::to_string(component)) +
                      "_gradient";
  }
}

void
JitableFunctions::gradient()
{
  const int field_port = inputs["field/port"].as_int32();
  const Jitable &field_jitable = *input_jitables[field_port];
  const Kernel &field_kernel = *input_kernels[field_port];
  const conduit::Node &obj = field_jitable.obj;
  std::string field_name;
  if(obj.has_path("value"))
  {
    field_name = obj["value"].as_string();
  }
  if(field_kernel.num_components > 1)
  {
    ASCENT_ERROR("gradient is only supported on scalar fields but a field with "
                 << field_kernel.num_components << " components was given.");
  }
  gradient(field_jitable, field_kernel, field_name, -1);
  out_kernel.num_components = 3;
}

void
JitableFunctions::vorticity()
{
  const int field_port = inputs["field/port"].as_int32();
  const Jitable &field_jitable = *input_jitables[field_port];
  const Kernel &field_kernel = *input_kernels[field_port];
  const conduit::Node &obj = field_jitable.obj;
  if(field_kernel.num_components < 2)
  {
    ASCENT_ERROR("Vorticity is only implemented for fields with at least 2 "
                 "components. The input field has "
                 << field_kernel.num_components << ".");
  }
  std::string field_name;
  if(obj.has_path("value"))
  {
    field_name = obj["value"].as_string();
  }
  else
  {
    field_name = filter_name + "_inp";
    temporary_field(field_kernel);
  }
  // calling gradient here reuses the logic to update entries and association
  for(int i = 0; i < field_kernel.num_components; ++i)
  {
    gradient(field_jitable, field_kernel, field_name, i);
  }
  // TODO make it easier to construct FieldCode
  if(not_fused)
  {
    TopologyCode topo_code =
        TopologyCode(out_jitable.topology, domain, out_jitable.arrays[dom_idx]);
    FieldCode field_code = FieldCode(field_name,
                                     field_jitable.association,
                                     std::move(topo_code),
                                     out_jitable.arrays[dom_idx],
                                     field_kernel.num_components,
                                     -1);
    field_code.vorticity(out_kernel.for_body);
    out_kernel.expr = field_name + "_vorticity";
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
  if(not_fused)
  {
    const int arg1_port = inputs["arg1/port"].to_int32();
    const int arg2_port = inputs["arg2/port"].to_int32();
    const int arg3_port = inputs["arg3/port"].to_int32();
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
// }}}

//-----------------------------------------------------------------------------
// -- Kernel
//-----------------------------------------------------------------------------
// {{{
void
Kernel::fuse_kernel(const Kernel &from)
{
  kernel_body.insert(from.kernel_body);
  for_body.insert(from.for_body);
}

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
  std::string kernel_string = "@kernel void map(";
  const int num_args = args.number_of_children();
  bool first = true;
  for(int i = 0; i < num_args; ++i)
  {
    const conduit::Node &arg = args.child(i);
    std::string type;
    // array type was already determined in alloc_device_array
    if(!arg.has_path("index"))
    {
      type = detail::type_string(arg.dtype()) + " ";
    }
    if(!first)
    {
      kernel_string += "                 ";
    }
    // TODO i need a better way of figuring out what to make const
    // i can't just check this because temporaries can't be const either
    // if(arg.name().find("output") == args.name().npos)
    // {
    //   kernel_string += "const ";
    // }
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

    // fuse array_code
    for(size_t i = 0; i < from.arrays.size(); ++i)
    {
      arrays[i].array_map.insert(from.arrays[i].array_map.begin(),
                                 from.arrays[i].array_map.end());
    }
  }
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

// TODO for now we just put the field on the mesh when calling execute
// should probably delete it later if it's an intermediate field
void
Jitable::execute(conduit::Node &dataset, const std::string &field_name)
{
  ASCENT_DATA_OPEN("jitable_execute");
  // TODO set this automatically?
  // occa::setDevice("mode: 'OpenCL', platform_id: 0, device_id: 1");
  static bool device_set = false;
  if(!device_set)
  {
    // running this in a loop segfaults...
    occa::setDevice("mode: 'CUDA', device_id: 0");
    // occa::setDevice("mode: 'Serial'");
    device_set = true;
  }
  occa::device &device = occa::getDevice();
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
    flow::Timer jitable_execute_timer;
    conduit::Node &dom = dataset.child(dom_idx);

    const conduit::Node &cur_dom_info = dom_info.child(dom_idx);

    const Kernel &kernel = kernels.at(cur_dom_info["kernel_type"].as_string());

    // the final number of entries
    const int entries = cur_dom_info["entries"].to_int64();

    // create output array and put it in array_map
    conduit::Schema output_schema;
    std::vector<std::string> component_names{};
    if(kernel.num_components > 1)
    {
      for(int i = 0; i < kernel.num_components; ++i)
      {
        component_names.push_back(std::string(1, 'x' + i));
      }
    }

    schemaFactory("interleaved",
                  conduit::DataType::FLOAT64_ID,
                  entries,
                  component_names,
                  output_schema);

    arrays[dom_idx].array_map["output"] = output_schema;

    // these are reference counted
    // need to keep the mem in scope or bad things happen
    std::vector<Array<unsigned char>> array_buffers;

    flow::Timer array_allocation_timer;
    // allocate arrays
    size_t output_index;
    conduit::Node new_args;
    for(const auto &array : arrays[dom_idx].array_map)
    {
      if(cur_dom_info["args"].has_path(array.first))
      {
        device_alloc_array(cur_dom_info["args/" + array.first],
                           array.second,
                           new_args,
                           array_buffers);
      }
      else
      {
        if(array.first == "output")
        {
          output_index = array_buffers.size();
        }
        // if it's not in args it doesn't point to any data so it's a temporary
        device_alloc_temporary(
            array.first, array.second, new_args, array_buffers);
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
    ASCENT_DATA_ADD("total input allocation time",
                    array_allocation_timer.elapsed());

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
      ASCENT_DATA_ADD("kernal compile time", kernel_compile_timer.elapsed());
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
    detail::get_occa_mem(array_buffers, array_memories);

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

    conduit::Node &n_output = dom["fields/" + field_name];
    n_output["association"] = association;
    n_output["topology"] = topology;

    flow::Timer alloc_output_timer;
    conduit::float64 *output_ptr;
    n_output["values"].set(output_schema);
    output_ptr = (conduit::float64 *)n_output["values"].data_ptr();
    // output to the host will always be contiguous
    ASCENT_DATA_ADD("cpu output alloc", alloc_output_timer.elapsed());
    ASCENT_DATA_ADD("cpu output bytes", output_schema.total_bytes_compact());

    flow::Timer kernel_run_timer;
    occa_kernel.run();
    ASCENT_DATA_ADD("kernel runtime", kernel_run_timer.elapsed());

    // copy back
    flow::Timer copy_back_timer;
    array_memories[output_index].copyTo(output_ptr);
    ASCENT_DATA_ADD("copy to host", copy_back_timer.elapsed());

    // dom["fields/" + field_name].print();
    ASCENT_DATA_ADD("domain execute time: ", jitable_execute_timer.elapsed());
  }
  ASCENT_DATA_CLOSE();
}
// }}}
//-----------------------------------------------------------------------------

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
