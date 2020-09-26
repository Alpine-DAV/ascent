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
/// file: ascent_derived_jit.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_DERVIVED_JIT_HPP
#define ASCENT_DERVIVED_JIT_HPP

#include <ascent.hpp>
#include <conduit.hpp>
#include <flow.hpp>
#include <unordered_map>
#include <unordered_set>
// Matt: there is a lot of code that needs its own file

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
template <typename T>
class InsertionOrderedSet
{
public:
  void
  insert(const T &item, const bool unique = true)
  {
    if(!unique)
    {
      insertion_ordered_data.push_back(item);
    }
    else if(data_set.find(item) == data_set.end())
    {
      data_set.insert(item);
      insertion_ordered_data.push_back(item);
    }
  }

  void
  insert(std::initializer_list<T> ilist, const bool unique = true)
  {
    for(const auto &item : ilist)
    {
      insert(item, unique);
    }
  }

  void
  insert(const InsertionOrderedSet<T> &ios, const bool unique = true)
  {
    for(const auto &item : ios.data())
    {
      insert(item, unique);
    }
  }

  T
  accumulate() const
  {
    T res;
    for(const auto &item : insertion_ordered_data)
    {
      res += item;
    }
    return res;
  }

  const std::vector<T> &
  data() const
  {
    return insertion_ordered_data;
  }

private:
  std::unordered_set<T> data_set;
  std::vector<T> insertion_ordered_data;
};

// codegen_arrays are a way to trick the codegen into bundling multiple fields
// into a single vector field by constructing a false schema. Their components
// are accessed using just the "component_name" rather than
// "<array_name>_<component_name>". They are not packed into args.
class SchemaBool
{
public:
  SchemaBool(const conduit::Schema &schema, bool codegen_array)
      : schema(schema), codegen_array(codegen_array){};

  conduit::Schema schema;
  bool codegen_array;
};

class ArrayCode
{
public:
  std::string index(const std::string &array_name,
                    const std::string &idx,
                    const int component = -1) const;
  std::string index(const std::string &idx,
                    const std::string name,
                    const std::ptrdiff_t offset,
                    const std::ptrdiff_t stride,
                    const size_t pointer_size) const;
  std::string index(const std::string &array_name,
                    const std::string &idx,
                    const std::string &component) const;

  std::unordered_map<std::string, SchemaBool> array_map;
};

class MathCode
{
public:
  void determinant_2x2(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &res_name,
                       const bool declare = true) const;
  void determinant_3x3(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &c,
                       const std::string &res_name,
                       const bool declare = true) const;
  void vector_subtract(InsertionOrderedSet<std::string> &code,
                       const std::string &a,
                       const std::string &b,
                       const std::string &res_name,
                       const int num_components,
                       const bool declare = true) const;
  void vector_add(InsertionOrderedSet<std::string> &code,
                  const std::string &a,
                  const std::string &b,
                  const std::string &res_name,
                  const int num_components,
                  const bool declare = true) const;
  void cross_product(InsertionOrderedSet<std::string> &code,
                     const std::string &a,
                     const std::string &b,
                     const std::string &res_name,
                     const int num_components,
                     const bool declare = true) const;
  void dot_product(InsertionOrderedSet<std::string> &code,
                   const std::string &a,
                   const std::string &b,
                   const std::string &res_name,
                   const int num_components,
                   const bool declare = true) const;
  void magnitude(InsertionOrderedSet<std::string> &code,
                 const std::string &a,
                 const std::string &res_name,
                 const int num_components,
                 const bool declare = true) const;
  void array_avg(InsertionOrderedSet<std::string> &code,
                 const int length,
                 const std::string &array_name,
                 const std::string &res_name,
                 const bool declare) const;
  void component_avg(InsertionOrderedSet<std::string> &code,
                     const int length,
                     const std::string &array_name,
                     const std::string &coord,
                     const std::string &res_name,
                     const bool declare) const;
};

class TopologyCode
{
public:
  TopologyCode(const std::string &topo_name,
               const conduit::Node &domain,
               const ArrayCode &array_code);

  void pack(conduit::Node &args) const;

  void vertex_xyz(InsertionOrderedSet<std::string> &code) const;
  void element_xyz(InsertionOrderedSet<std::string> &code) const;
  void volume(InsertionOrderedSet<std::string> &code) const;
  void area(InsertionOrderedSet<std::string> &code) const;

  // helper functions
  void vertex_idx(InsertionOrderedSet<std::string> &code) const;
  void element_idx(InsertionOrderedSet<std::string> &code) const;
  void dxdydz(InsertionOrderedSet<std::string> &code) const;
  void structured_vertices(InsertionOrderedSet<std::string> &code) const;
  void structured_vertex_locs(InsertionOrderedSet<std::string> &code) const;
  void unstructured_vertices(InsertionOrderedSet<std::string> &code,
                             const std::string &index_name = "item") const;
  void unstructured_vertex_locs(InsertionOrderedSet<std::string> &code,
                                const std::string &index_name = "item") const;
  void vertex_coord(InsertionOrderedSet<std::string> &code,
                    const std::string &coord,
                    const std::string &index_name,
                    const std::string &res_name,
                    const bool declare = true) const;
  void element_coord(InsertionOrderedSet<std::string> &code,
                     const std::string &coord,
                     const std::string &index_name,
                     const std::string &res_name,
                     const bool declare = true) const;
  void vertex_xyz(InsertionOrderedSet<std::string> &code,
                  const std::string &index_name,
                  const bool index_array,
                  const std::string &res_name,
                  const bool declare = true) const;
  void hexahedral_volume(InsertionOrderedSet<std::string> &code,
                         const std::string &vertex_locs,
                         const std::string &res_name) const;
  void tetrahedral_volume(InsertionOrderedSet<std::string> &code,
                          const std::string &vertex_locs,
                          const std::string &res_name) const;
  void quadrilateral_area(InsertionOrderedSet<std::string> &code,
                          const std::string &p0,
                          const std::string &p1,
                          const std::string &p2,
                          const std::string &p3,
                          const std::string &res_name) const;
  void quadrilateral_area(InsertionOrderedSet<std::string> &code,
                          const std::string &vertex_locs,
                          const std::string &res_name) const;
  void triangle_area(InsertionOrderedSet<std::string> &code,
                     const std::string &p0,
                     const std::string &p1,
                     const std::string &p2,
                     const std::string &res_name) const;
  void triangle_area(InsertionOrderedSet<std::string> &code,
                     const std::string &vertex_locs,
                     const std::string &res_name) const;
  void polygon_area_vec(InsertionOrderedSet<std::string> &code,
                        const std::string &vertex_locs,
                        const std::string &res_name) const;
  void polygon_area(InsertionOrderedSet<std::string> &code,
                    const std::string &vertex_locs,
                    const std::string &res_name) const;
  void polyhedron_volume(InsertionOrderedSet<std::string> &code,
                         const std::string &vertex_locs,
                         const std::string &res_name) const;
  void hexahedral_surface_area(InsertionOrderedSet<std::string> &code,
                               const std::string &vertex_locs,
                               const std::string &res_name) const;
  void tetrahedral_surface_area(InsertionOrderedSet<std::string> &code,
                                const std::string &vertex_locs,
                                const std::string &res_name) const;
  void surface_area(InsertionOrderedSet<std::string> &code) const;

  const std::string topo_name;
  const conduit::Node &domain;
  std::string topo_type;
  int num_dims;
  std::string shape;
  int shape_size;

  const ArrayCode &array_code;

  const MathCode math_code;
};

class FieldCode
{
public:
  // if component is -1 use all the field's components
  FieldCode(const std::string &field_name,
            const std::string &association,
            const std::shared_ptr<const TopologyCode> topo_code,
            const ArrayCode &array_code,
            const int num_components,
            const int component);
  void gradient(InsertionOrderedSet<std::string> &code) const;
  void curl(InsertionOrderedSet<std::string> &code) const;
  void recenter(InsertionOrderedSet<std::string> &code,
                const std::string &target_association,
                const std::string &res_name) const;

private:
  // Calculate the element associated gradient of a vertex associated field on
  // a hexahedral mesh
  void hex_gradient(InsertionOrderedSet<std::string> &code,
                    const std::string &res_name) const;
  // Calculate the element associated gradient of a vertex associated field on
  // a quadrilateral mesh
  void quad_gradient(InsertionOrderedSet<std::string> &code,
                     const std::string &res_name) const;
  void element_vertex_values(InsertionOrderedSet<std::string> &code,
                             const std::string &res_name,
                             const int component,
                             const bool declare) const;
  void field_idx(InsertionOrderedSet<std::string> &code,
                 const std::string &index_name,
                 const std::string &association,
                 const std::string &res_name,
                 const bool declare) const;
  void visit_upper(InsertionOrderedSet<std::string> &code,
                   const std::string &index_name,
                   const std::string &if_body,
                   const std::string &else_body,
                   const int dim) const;
  void visit_lower(InsertionOrderedSet<std::string> &code,
                   const std::string &index_name,
                   const std::string &if_body,
                   const std::string &else_body,
                   const int dim) const;
  void visit_current(InsertionOrderedSet<std::string> &code,
                     const std::string &index_name,
                     const std::string &if_body,
                     const std::string &else_body,
                     const int dim) const;
  void visit_vertex_elements(InsertionOrderedSet<std::string> &code,
                             const std::string &index_name,
                             const std::string &if_body,
                             const std::string &else_body,
                             const int dim) const;

  const std::string field_name;
  const std::string association;
  const int num_components;
  const int component;

  const ArrayCode &array_code;

  const std::shared_ptr<const TopologyCode> topo_code;
  const MathCode math_code;
};

class Kernel
{
public:
  void fuse_kernel(const Kernel &from);
  std::string generate_output(const std::string &output,
                              bool output_exists) const;
  std::string generate_loop(const std::string &output,
                            const ArrayCode &array_code,
                            const std::string &entries_name) const;

  InsertionOrderedSet<std::string> functions;
  InsertionOrderedSet<std::string> kernel_body;
  InsertionOrderedSet<std::string> for_body;
  std::string expr;
  // number of components associated with the expression in expr
  // if the expression is a vector expr will just be the name of a single vector
  int num_components;
};

class Jitable
{
public:
  Jitable(const int num_domains)
  {
    for(int i = 0; i < num_domains; ++i)
    {
      dom_info.append();
    }
    arrays.resize(num_domains);
  }

  void fuse_vars(const Jitable &from);
  bool can_execute() const;
  void execute(conduit::Node &dataset, const std::string &field_name);
  std::string generate_kernel(const int dom_idx,
                              const conduit::Node &args) const;

  // map of kernel types (e.g. for different topologies)
  std::unordered_map<std::string, Kernel> kernels;
  // stores entries and argument values for each domain
  conduit::Node dom_info;
  // Store the array schemas. Used by code generation. We will copy to these
  // schemas when we execute
  std::vector<ArrayCode> arrays;
  std::string topology;
  std::string association;
  // metadata used to make the . operator work and store various jitable state
  conduit::Node obj;
};

// handles kernel fusion for various functions
// calls TopologyCode, FieldCode, etc.
class JitableFunctions
{
public:
  JitableFunctions(const conduit::Node &params,
                   const std::vector<const Jitable *> &input_jitables,
                   const std::vector<const Kernel *> &input_kernels,
                   const std::string &filter_name,
                   const conduit::Node &dataset,
                   const int dom_idx,
                   const bool not_fused,
                   Jitable &out_jitable,
                   Kernel &out_kernel);

  void binary_op();
  void builtin_functions(const std::string &function_name);
  void expr_dot();
  void expr_if();
  void derived_field();
  void gradient();
  void curl();
  void recenter();
  void magnitude();
  void vector();
  void binning_value(const conduit::Node &binning);
  void rand();

private:
  void topo_attrs(const conduit::Node &obj, const std::string &name);
  void gradient(const int field_port, const int component);
  void temporary_field(const Kernel &field_kernel,
                       const std::string &field_name);
  std::string possible_temporary(const int field_port);

  const conduit::Node &params;
  const std::vector<const Jitable *> &input_jitables;
  const std::vector<const Kernel *> &input_kernels;
  const std::string &filter_name;
  const conduit::Node &dataset;
  const int dom_idx;
  const bool not_fused;
  Jitable &out_jitable;
  Kernel &out_kernel;
  const conduit::Node &inputs;
  const conduit::Node &domain;
};

class MemoryRegion
{
public:
  MemoryRegion(const void *start, const void *end);
  MemoryRegion(const void *start, const size_t size);
  bool operator<(const MemoryRegion &other) const;

  const unsigned char *start;
  const unsigned char *end;
  mutable bool allocated;
  mutable size_t index;
};

class JitExecutionPolicy
{
public:
  JitExecutionPolicy();
  virtual bool should_execute(const Jitable &jitable) const = 0;
  virtual std::string get_name() const = 0;
};

class FusePolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

class AlwaysExecutePolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

class RoundtripPolicy final : public JitExecutionPolicy
{
public:
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;
};

// fuse until the number of bytes in args exceeds a threshold
class InputBytesPolicy final : public JitExecutionPolicy
{
public:
  InputBytesPolicy(const size_t num_bytes);
  bool should_execute(const Jitable &jitable) const override;
  std::string get_name() const override;

private:
  const size_t num_bytes;
};

void pack_topology(const std::string &topo_name,
                   const conduit::Node &domain,
                   conduit::Node &args,
                   ArrayCode &array);
void pack_array(const conduit::Node &array,
                const std::string &name,
                conduit::Node &args,
                ArrayCode &array_code);
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
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
