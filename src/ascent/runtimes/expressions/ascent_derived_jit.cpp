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

#include <ascent_config.h>
#include "ascent_derived_jit.hpp"
#include "ascent_array.hpp"
#include "ascent_blueprint_architect.hpp"
#include "ascent_blueprint_topologies.hpp"
#include "ascent_expressions_ast.hpp"
#include <runtimes/flow_filters/ascent_runtime_utils.hpp>

#include <ascent_data_logger.hpp>
#include <ascent_logging.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

#ifdef ASCENT_JIT_ENABLED
#include <occa.hpp>
#include <occa/utils/env.hpp>
#include <stdlib.h>
  #ifdef ASCENT_CUDA_ENABLED
    #include <occa/modes/cuda/utils.hpp>
  #endif
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

#ifdef ASCENT_JIT_ENABLED
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
    unsigned char * ptr;
    if(mode == "Serial" || mode == "OpenMP")
    {
      ptr = buf.get_host_ptr();
    }
#ifdef ASCENT_CUDA_ENABLED
    else if(mode == "CUDA")
    {
      ptr = buf.get_device_ptr();
    }
#endif
    else
    {
      ASCENT_ERROR("Unknow occa mode " << mode);
    }

    void * v_ptr = (void *)(ptr + buf_offset);
    occa[i] = device.wrapMemory(v_ptr, buf_size * sizeof(unsigned char));
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
#endif

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
#ifdef ASCENT_JIT_ENABLED
  ASCENT_DATA_OPEN("jitable_execute");
  // TODO set this during initialization not here
  static bool init = false;
  if(!init)
  {
    // running this in a loop segfaults...
#ifdef ASCENT_CUDA_ENABLED
    // TODO get the right device_id
    occa::setDevice({{"mode", "CUDA"}, {"device_id", 0}});
#elif defined(ASCENT_USE_OPENMP)
    occa::setDevice({{"mode", "OpenMP"}});
#else
    occa::setDevice({{"mode", "Serial"}});
#endif
    occa::env::setOccaCacheDir(::ascent::runtime::filters::default_dir());
    init = true;
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
                  kernel.num_components,
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
#else
  ASCENT_ERROR("JIT compilation for derived fields requires OCCA support"<<
               " but Ascent was not compiled with OCCA.");
#endif
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
