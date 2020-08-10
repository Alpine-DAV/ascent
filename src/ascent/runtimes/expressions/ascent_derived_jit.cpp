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
#include "ascent_blueprint_architect.hpp"
#include "ascent_expressions_ast.hpp"

#include <ascent_logging.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <occa.hpp>

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
      indent = indent.substr(2);
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

std::string
Kernel::generate_inner_scope() const
{
  std::string inner_scope_string;
  for(const auto &line : inner_scope.data())
  {
    inner_scope_string += line;
  }
  return inner_scope_string;
}

std::string
Kernel::generate_for_body(const std::string &output, bool output_exists) const
{
  std::string new_for_body = for_body + generate_inner_scope();
  if(output_exists)
  {
    new_for_body += output + " = " + expr + ";\n";
  }
  else
  {
    new_for_body += "const double " + output + " = " + expr + ";\n";
  }
  return new_for_body;
}

// clang-format off
std::string
Kernel::generate_loop(const std::string& output) const
{
  return "for (int group = 0; group < entries; group += 128; @outer)\n"
         "{\n"
           "for (int item = group; item < (group + 128); ++item; @inner)\n"
           "{\n"
             "if (item < entries)\n"
             "{\n" +
                generate_for_body(output, false) +
                output+"_ptr[item] = "+output+";\n"
             "}\n"
           "}\n"
         "}\n";
}
// clang-format on

std::string
Jitable::generate_kernel(const int dom_idx) const
{
  const conduit::Node &cur_dom_info = dom_info.child(dom_idx);
  const Kernel &kernel = kernels.at(cur_dom_info["kernel_type"].as_string());
  std::string kernel_string = "@kernel void map(const int entries,\n";
  for(const auto &param : cur_dom_info["args"].child_names())
  {
    kernel_string += "                 " + param;
  }
  kernel_string += "                 double *output_ptr)\n{\n";
  kernel_string += kernel.kernel_body;
  kernel_string += kernel.generate_loop("output");
  kernel_string += "}";
  return detail::indent_code(kernel_string, 0);
}

//-----------------------------------------------------------------------------
// -- TopologyCode
//-----------------------------------------------------------------------------
TopologyCode::TopologyCode(const std::string &topo_name,
                           const conduit::Node &dom)
{
  const conduit::Node &n_topo = dom["topologies/" + topo_name];
  const std::string coords_name = n_topo["coordset"].as_string();
  this->topo_type = n_topo["type"].as_string();
  this->topo_name = topo_name;
  this->num_dims = topo_dim(topo_name, dom);
}

void
TopologyCode::cell_idx(InsertionOrderedSet<std::string> &code)
{
  if(topo_type == "uniform" || topo_type == "rectilinear" ||
     topo_type == "structured")
  {
    code.insert(
        {"int " + topo_name + "_cell_idx[" + std::to_string(num_dims) + "];\n",
         topo_name + "_cell_idx[0] = item % (" + topo_name +
             "_dims_i - 1);\n"});

    if(num_dims >= 2)
    {
      code.insert(topo_name + "_cell_idx[1] = (item / (" + topo_name +
                  "_dims_i - 1)) % (" + topo_name + "_dims_j - 1);\n");
    }
    if(num_dims == 3)
    {
      code.insert(topo_name + "_cell_idx[2] = item / ((" + topo_name +
                  "_dims_i - 1) * (" + topo_name + "_dims_j - 1));\n");
    }
  }
  else
  {
    ASCENT_ERROR("cell_idx for unstructured is not implemented.");
  }
}

// vertices are ordered in the VTK format
// https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
void
TopologyCode::structured_vertices(InsertionOrderedSet<std::string> &code)
{
  if(topo_type != "structured")
  {
    ASCENT_ERROR("The function structured_vertices only supports structured "
                 "topologies.");
  }
  cell_idx(code);

  // vertex indices
  code.insert("int " + topo_name + "_vertices[" +
              std::to_string(static_cast<int>(std::pow(2, num_dims))) + "];\n");
  if(num_dims == 1)
  {
    code.insert(
        {topo_name + "_vertices[0] = " + topo_name + "_cell_idx[0];\n",
         topo_name + "_vertices[1] = " + topo_name + "_vertices[0] + 1;\n"});
  }
  else if(num_dims == 2)
  {
    code.insert(
        {topo_name + "_vertices[0] = " + topo_name + "_cell_idx[1] * " +
             topo_name + "_dims_i + " + topo_name + "_cell_idx[0];\n",
         topo_name + "_vertices[1] = " + topo_name + "_vertices[0] + 1;\n",
         topo_name + "_vertices[2] = " + topo_name + "_vertices[1] + " +
             topo_name + "_dims_i;\n",
         topo_name + "_vertices[3] = " + topo_name + "_vertices[2] - 1;\n"});
  }
  else if(num_dims == 3)
  {
    code.insert({
        topo_name + "_vertices[0] = (" + topo_name + "_cell_idx[2] * " +
            topo_name + "_dims_j + " + topo_name + "_cell_idx[1]) * " +
            topo_name + "_dims_i + " + topo_name + "_cell_idx[0];\n",
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
  vertex_xyz(
      code, topo_name + "_vertices[0]", false, topo_name + "_vertices_0");
  vertex_xyz(
      code, topo_name + "_vertices[1]", false, topo_name + "_vertices_1");
  if(num_dims >= 2)
  {
    vertex_xyz(
        code, topo_name + "_vertices[2]", false, topo_name + "_vertices_2");
    vertex_xyz(
        code, topo_name + "_vertices[3]", false, topo_name + "_vertices_3");
  }
  if(num_dims == 3)
  {
    vertex_xyz(
        code, topo_name + "_vertices[4]", false, topo_name + "_vertices_4");
    vertex_xyz(
        code, topo_name + "_vertices[5]", false, topo_name + "_vertices_5");
    vertex_xyz(
        code, topo_name + "_vertices[6]", false, topo_name + "_vertices_6");
    vertex_xyz(
        code, topo_name + "_vertices[7]", false, topo_name + "_vertices_7");
  }
}

void
TopologyCode::unstructured_vertices(InsertionOrderedSet<std::string> &code)
{
  if(topo_type != "unstructured")
  {
    ASCENT_ERROR(
        "The function unstructured_vertices only supports unstructured "
        "topologies.");
  }
  cell_idx(code);
}

void
TopologyCode::cell_xyz(InsertionOrderedSet<std::string> &code)
{
  code.insert("double " + topo_name + "_cell_loc[" + std::to_string(num_dims) +
              "];\n");
  if(topo_type == "uniform")
  {
    cell_idx(code);
    code.insert(topo_name + "_cell_loc[0] = " + topo_name + "_origin_x + " +
                topo_name + "_cell_idx[0] * " + topo_name + "_spacing_dx;\n");
    if(num_dims >= 2)
    {
      code.insert(topo_name + "_cell_loc[0] = " + topo_name + "_origin_y + " +
                  topo_name + "_cell_idx[1] * " + topo_name + "_spacing_dy;\n");
    }
    if(num_dims == 3)
    {
      code.insert(topo_name + "_cell_loc[0] = " + topo_name + "_origin_z + " +
                  topo_name + "_cell_idx[2] * " + topo_name + "_spacing_dz;\n");
    }
  }
  else if(topo_type == "rectilinear")
  {
    cell_idx(code);
    code.insert(topo_name + "_cell_loc[0] = (" + topo_name + "_coords_x[" +
                topo_name + "_cell_idx[0]] + " + topo_name + "_coords_x[" +
                topo_name + "_cell_idx[0] + 1]) / 2.0;\n");

    if(num_dims >= 2)
    {
      code.insert(topo_name + "_cell_loc[1] = (" + topo_name + "_coords_y[" +
                  topo_name + "_cell_idx[1]] + " + topo_name + "_coords_y[" +
                  topo_name + "_cell_idx[1] + 1]) / 2.0;\n");
    }
    if(num_dims == 3)
    {
      code.insert(topo_name + "_cell_loc[2] = (" + topo_name + "_coords_z[" +
                  topo_name + "_cell_idx[2]] + " + topo_name + "_coords_z[" +
                  topo_name + "_cell_idx[2] + 1]) / 2.0;\n");
    }
  }
  else if(topo_type == "structured")
  {
    structured_vertices(code);

    // sum components and divide by num_vertices
    const double num_vertices = std::pow(2, num_dims);
    for(int i = 0; i < num_dims; ++i)
    {
      std::stringstream vert_avg;
      vert_avg << "(";
      for(int j = 0; j < num_vertices; ++j)
      {
        if(j != 0)
        {
          vert_avg << " + ";
        }
        vert_avg << topo_name + "_vertices_" << j << "[" << i << "]";
      }
      vert_avg << ") / " << num_vertices;
      code.insert(topo_name + "_cell_loc[" + std::to_string(i) +
                  "] = " + vert_avg.str() + ";\n");
    }
  }
  else
  {
    ASCENT_ERROR("Cannot get cell location for unstructured topology with "
                 << num_dims << " dimensions.");
  }
}

void
TopologyCode::vertex_idx(InsertionOrderedSet<std::string> &code)
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
    // vertex_idx is just item for explicit (unstructured) coords
    // vertex_idx[0] = item
    // vertex_idx[1] = item
    // vertex_idx[2] = item
    ASCENT_ERROR("vertex_idx does not need to be calculated for unstructured "
                 "topologies.");
  }
}

void
TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code,
                         const std::string &index_name,
                         const bool index_array,
                         const std::string &res_name)
{
  code.insert({"double " + res_name + "[" + std::to_string(num_dims) + "];\n",
               res_name + "[0] = " + topo_name + "_coords_x[" + index_name +
                   (index_array ? "[0]" : "") + "];\n"});
  if(num_dims >= 2)
  {
    code.insert(res_name + "[1] = " + topo_name + "_coords_y[" + index_name +
                (index_array ? "[1]" : "") + "];\n");
  }
  if(num_dims == 3)
  {
    code.insert(res_name + "[2] = " + topo_name + "_coords_z[" + index_name +
                (index_array ? "[2]" : "") + "];\n");
  }
}

void
TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code)
{
  if(topo_type == "uniform" || topo_type == "rectilinear" ||
     topo_type == "structured")
  {
    vertex_idx(code);
    vertex_xyz(
        code, topo_name + "_vertex_idx", true, topo_name + "_vertex_loc");
  }
  else if(topo_type == "unstructured")
  {
    vertex_xyz(code, "item", false, topo_name + "_vertex_loc");
  }
}

void
TopologyCode::dxdydz(InsertionOrderedSet<std::string> &code)
{
  if(topo_type == "rectilinear")
  {
    cell_idx(code);
    code.insert("double " + topo_name + "_dx = " + topo_name + "_coords_x[" +
                topo_name + "_cell_idx[0] + 1] - " + topo_name + "_coords_x[" +
                topo_name + "_cell_idx[0]];\n");
    if(num_dims >= 2)
    {
      code.insert("double " + topo_name + "_dy = " + topo_name + "_coords_y[" +
                  topo_name + "_cell_idx[1] + 1] - " + topo_name +
                  "_coords_y[" + topo_name + "_cell_idx[1]];\n");
    }
    if(num_dims == 3)
    {
      code.insert({"double " + topo_name + "_dz = " + topo_name + "_coords_z[" +
                   topo_name + "_cell_idx[2] + 1] - " + topo_name +
                   "_coords_z[" + topo_name + "_cell_idx[2]];\n"});
    }
  }
}

void
determinant_3x3(InsertionOrderedSet<std::string> &code,
                const std::string &a,
                const std::string &b,
                const std::string &c,
                const std::string &res_name)
{
  code.insert("double " + res_name + " = " + a + "[0] * (" + b + "[1] * " + c +
              "[2] - " + c + "[1] * " + b + "[2]) - " + a + "[1] * (" + b +
              "[0] * " + c + "[2] - " + c + "[0] * " + b + "[2]) + " + a +
              "[2] * (" + b + "[0] * " + c + "[1] - " + c + "[0] * " + b +
              "[1]);\n");
}

void
vector_subtract(InsertionOrderedSet<std::string> &code,
                const std::string &a,
                const std::string &b,
                const std::string &res_name,
                const int num_dims)
{
  code.insert("double " + res_name + "[" + std::to_string(num_dims) + "];\n");
  for(int i = 0; i < num_dims; ++i)
  {
    code.insert(res_name + "[" + std::to_string(i) + "] = " + a + "[" +
                std::to_string(i) + "] - " + b + "[" + std::to_string(i) +
                "];\n");
  }
}

// https://www.osti.gov/servlets/purl/632793 (14)
// switch vertices 2 and 3 and vertices 6 and 7 to match vtk order
void
hexahedral_volume(InsertionOrderedSet<std::string> &code,
                  const std::string &vertices_name,
                  const std::string &res_name)
{
  vector_subtract(
      code, vertices_name + "_6", vertices_name + "_0", res_name + "_6m0", 3);
  vector_subtract(
      code, vertices_name + "_1", vertices_name + "_0", res_name + "_1m0", 3);
  vector_subtract(
      code, vertices_name + "_2", vertices_name + "_5", res_name + "_2m5", 3);
  vector_subtract(
      code, vertices_name + "_4", vertices_name + "_0", res_name + "_4m0", 3);
  vector_subtract(
      code, vertices_name + "_5", vertices_name + "_7", res_name + "_5m7", 3);
  vector_subtract(
      code, vertices_name + "_3", vertices_name + "_0", res_name + "_3m0", 3);
  vector_subtract(
      code, vertices_name + "_7", vertices_name + "_2", res_name + "_7m2", 3);
  // can save 4 flops if we use the fact that 6m0 is always the first column of
  // the determinant
  determinant_3x3(code,
                  res_name + "_6m0",
                  res_name + "_1m0",
                  res_name + "_2m5",
                  res_name + "_det0");
  determinant_3x3(code,
                  res_name + "_6m0",
                  res_name + "_4m0",
                  res_name + "_5m7",
                  res_name + "_det1");
  determinant_3x3(code,
                  res_name + "_6m0",
                  res_name + "_3m0",
                  res_name + "_7m2",
                  res_name + "_det2");
  code.insert("double " + res_name + " = (" + res_name + "_det0 + " + res_name +
              "_det1 + " + res_name + "_det2) / 6.0;\n");
}

void
TopologyCode::volume(InsertionOrderedSet<std::string> &code)
{
  if(topo_type == "uniform")
  {
    if(num_dims == 3)
    {
      code.insert("double " + topo_name + "_volume = " + topo_name +
                  "_spacing_dx * " + topo_name + "_spacing_dy * " + topo_name +
                  "_spacing_dz;\n");
    }
    else if(num_dims == 2)
    {
      code.insert("double " + topo_name + "_volume = " + topo_name +
                  "_spacing_dx * " + topo_name + "_spacing_dy;\n");
    }
    else
    {
      code.insert("double " + topo_name + "_volume = " + topo_name +
                  "_spacing_dx;\n");
    }
  }
  else if(topo_type == "rectilinear")
  {
    dxdydz(code);
    if(num_dims == 3)
    {
      code.insert("double " + topo_name + "_volume = " + topo_name + "_dx * " +
                  topo_name + "_dy * " + topo_name + "_dz;\n");
    }
    else if(num_dims == 2)
    {
      code.insert("double " + topo_name + "_volume = " + topo_name + "_dx * " +
                  topo_name + "_dy;\n");
    }
    else
    {
      code.insert("double " + topo_name + "_volume = " + topo_name + "_dx;\n");
    }
  }
  else if(topo_type == "structured")
  {
    structured_vertices(code);
    if(num_dims == 3)
    {
      hexahedral_volume(code, topo_name + "_vertices", topo_name + "_volume");
    }
    else if(num_dims == 1)
    {
      vector_subtract(code,
                      topo_name + "vertices_1",
                      topo_name + "vertices_0",
                      topo_name + "_volume",
                      1);
    }
    else
    {
      ASCENT_ERROR("volume is not implemented for structured topologies with "
                   << num_dims << " dimensions.");
    }
  }
  else if(topo_type == "unstructured")
  {
  }
}

//-----------------------------------------------------------------------------
// -- Kernel
//-----------------------------------------------------------------------------
void
Kernel::fuse_kernel(const Kernel &from)
{
  kernel_body = kernel_body + from.kernel_body;
  for_body = for_body + from.for_body;
  inner_scope.insert(from.inner_scope);
}

//-----------------------------------------------------------------------------
// -- Jitable
//-----------------------------------------------------------------------------
void
Jitable::fuse_vars(const Jitable &from)
{
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
      // TODO should this throw an error?
      association = "none";
    }
  }

  int num_domains = from.dom_info.number_of_children();
  for(int dom_idx = 0; dom_idx < num_domains; ++dom_idx)
  {
    // fuse entries
    const conduit::Node &from_dom_info = from.dom_info.child(dom_idx);
    conduit::Node &to_dom_info = dom_info.child(dom_idx);
    if(from_dom_info.has_path("entries"))
    {
      if(to_dom_info.has_path("entries"))
      {
        if(to_dom_info["entries"].to_int64() !=
           from_dom_info["entries"].to_int64())
        {
          ASCENT_ERROR("JIT: Failed to fuse kernels due to an incompatible "
                       "number of entries: "
                       << to_dom_info["entries"].to_int64() << " versus "
                       << from_dom_info["entries"].to_int64());
        }
      }
      else
      {
        to_dom_info["entries"] = from_dom_info["entries"];
      }
    }

    // copy kernel_type
    dom_info.child(dom_idx)["kernel_type"] = from_dom_info["kernel_type"];

    // fuse args
    conduit::NodeConstIterator arg_itr = from_dom_info["args"].children();
    while(arg_itr.has_next())
    {
      const conduit::Node &arg = arg_itr.next();
      conduit::Node &to_args = dom_info.child(dom_idx)["args"];
      if(!to_args.has_path(arg.name()))
      {
        if(arg.dtype().number_of_elements() > 1)
        {
          // don't copy arrays
          to_args[arg.name()].set_external(arg);
        }
        else
        {
          to_args[arg.name()].set(arg);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
// How to Debug OCCA Kernels with LLDB
//-----------------------------------------------------------------------------
// 1. occa::setDevice("mode: 'Serial'");
// 2. export CXXFLAGS="-g" OCCA_VERBOSE=1
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
//    (e.g. "b ::map(const int &, const double *, const double &, double *)")

// TODO for now we just put the field on the mesh when calling execute
// should probably delete it later if it's an intermediate field
void
Jitable::execute(conduit::Node &dataset, const std::string &field_name)
{
  // TODO set this automatically?
  // occa::setDevice("mode: 'OpenCL', platform_id: 0, device_id: 1");
  occa::setDevice("mode: 'Serial'");
  occa::device &device = occa::getDevice();
  occa::kernel occa_kernel;

  // we need an association and topo so we can put the field back on the mesh
  // TODO create a new topo with vertex assoc for temporary fields
  if(topology.empty() || topology == "none")
  {
    ASCENT_ERROR("Error while executing derived field: Could not determine the "
                 "topology.");
  }
  if(association.empty() || association == "none")
  {
    ASCENT_ERROR("Error while executing derived field: Could not determine the "
                 "association.");
  }

  const int num_domains = dataset.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = dataset.child(i);

    const conduit::Node &cur_dom_info = dom_info.child(i);

    const std::string kernel_string = generate_kernel(i);

    const int entries = cur_dom_info["entries"].to_int64();

    std::cout << kernel_string << std::endl;

    try
    {
      occa_kernel = device.buildKernelFromString(kernel_string, "map");
    }
    catch(const occa::exception &e)
    {
      ASCENT_ERROR("Jitable: Expression compilation failed:\n"
                   << e.what() << "\n\n"
                   << cur_dom_info.to_yaml() << kernel_string);
    }
    catch(...)
    {
      ASCENT_ERROR(
          "Jitable: Expression compilation failed with an unknown error.\n"
          << cur_dom_info.to_yaml() << kernel_string);
    }

    occa_kernel.clearArgs();

    // pass invocation size
    occa_kernel.pushArg(entries);

    // these are reference counted
    // need to keep the mem in scope or bad things happen
    std::vector<occa::memory> array_memories;
    const int num_args = cur_dom_info["args"].number_of_children();
    for(int i = 0; i < num_args; ++i)
    {
      const conduit::Node &arg = cur_dom_info["args"].child(i);
      const int size = arg.dtype().number_of_elements();
      if(size > 1)
      {
        array_memories.emplace_back();
        if(arg.dtype().is_float64())
        {
          const conduit::float64 *vals = arg.as_float64_ptr();
          array_memories.back() =
              device.malloc(size * sizeof(conduit::float64), vals);
        }
        else if(arg.dtype().is_float32())
        {
          const conduit::float32 *vals = arg.as_float32_ptr();
          array_memories.back() =
              device.malloc(size * sizeof(conduit::float32), vals);
        }
        else if(arg.dtype().is_int64())
        {
          const conduit::int64 *vals = arg.as_int64_ptr();
          array_memories.back() =
              device.malloc(size * sizeof(conduit::int64), vals);
        }
        else if(arg.dtype().is_int32())
        {
          const conduit::int32 *vals = arg.as_int32_ptr();
          array_memories.back() =
              device.malloc(size * sizeof(conduit::int32), vals);
        }
        else
        {
          ASCENT_ERROR(
              "JIT: Unknown array argument type. Array: " << arg.to_yaml());
        }
        occa_kernel.pushArg(array_memories.back());
      }
      else if(arg.dtype().is_integer())
      {
        occa_kernel.pushArg(arg.to_int32());
      }
      else if(arg.dtype().is_floating_point())
      {
        occa_kernel.pushArg(arg.to_float64());
      }
      else
      {
        ASCENT_ERROR("JIT: Unknown argument type of argument: " << arg.name());
      }
    }

    std::cout << "INVOKE SIZE " << entries << "\n";
    conduit::Node &n_output = dom["fields/" + field_name];
    n_output["association"] = association;
    n_output["topology"] = topology;

    n_output["values"] = conduit::DataType::float64(entries);
    double *output_ptr = n_output["values"].as_float64_ptr();
    occa::array<double> o_output(entries);

    occa_kernel.pushArg(o_output);
    occa_kernel.run();

    o_output.memory().copyTo(output_ptr);

    dom["fields/" + field_name].print();
  }
}
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
