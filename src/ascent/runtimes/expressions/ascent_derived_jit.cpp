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
// -- MathCode
//-----------------------------------------------------------------------------
// {{{
void
MathCode::determinant_3x3(InsertionOrderedSet<std::string> &code,
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
MathCode::vector_subtract(InsertionOrderedSet<std::string> &code,
                          const std::string &a,
                          const std::string &b,
                          const std::string &res_name,
                          const int num_components,
                          const bool declare)
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
                     const bool declare)
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
                        const int num_components)
{
  code.insert("double " + res_name + "[3];\n");
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
                      const int num_components)
{
  code.insert("double " + res_name + "[" + std::to_string(num_components) +
              "];\n");
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
                    const int num_components)
{
  if(num_components == 3)
  {
    code.insert("double " + res_name + " = sqrt(" + a + "[0] * " + a +
                "[0] + " + a + "[1] * " + a + "[1] + " + a + "[2] * " + a +
                "[2]);\n");
  }
  else if(num_components == 2)
  {
    code.insert("double " + res_name + " = sqrt(" + a + "[0] * " + a +
                "[0] + " + a + "[1] * " + a + "[1]);\n");
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
                           const conduit::Node &domain)
    : topo_name(topo_name)
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
TopologyCode::element_idx(InsertionOrderedSet<std::string> &code)
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
void
TopologyCode::structured_vertices(InsertionOrderedSet<std::string> &code)
{
  if(topo_type != "structured")
  {
    ASCENT_ERROR("The function structured_vertices only supports structured "
                 "topologies.");
  }

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

void
TopologyCode::unstructured_vertices(InsertionOrderedSet<std::string> &code,
                                    const std::string &index_name)
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
               topo_name + "_connectivity[" + topo_name + "_offset + i]",
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
    code.insert("double " + topo_name + "_vertex_locs[" + topo_name +
                "_shape_size][" + std::to_string(num_dims) + "];\n");
    for(int i = 0; i < shape_size; ++i)
    {
      vertex_xyz(code,
                 topo_name + "_connectivity[" + index_name + " * " + topo_name +
                     "_shape_size + " + std::to_string(i) + "]",
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
  code.insert((declare ? "double " : "") + res_name + " = " + vert_avg.str() +
              ";\n");
}

void
TopologyCode::element_coord(InsertionOrderedSet<std::string> &code,
                            const std::string &coord,
                            const std::string &index_name,
                            const std::string &res_name,
                            const bool declare)
{
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
    code.insert((declare ? "double " : "") + res_name + " = " + topo_name +
                "_origin_" + coord + +" + (" + my_index_name + " + 0.5) * " +
                topo_name + "_spacing_d" + coord + ";\n");
  }
  else if(topo_type == "rectilinear")
  {
    element_idx(code);
    code.insert((declare ? "double " : "") + res_name + " = (" + topo_name +
                "_coords_" + coord + "[" + my_index_name + "] + " + topo_name +
                "_coords_x[" + my_index_name + " + 1]) / 2.0;\n");
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
      // If this is an issue we can make a special case for it in element_xyz
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
TopologyCode::element_xyz(InsertionOrderedSet<std::string> &code)
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
TopologyCode::vertex_coord(InsertionOrderedSet<std::string> &code,
                           const std::string &coord,
                           const std::string &index_name,
                           const std::string &res_name,
                           const bool declare)
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
    code.insert((declare ? "double " : "") + res_name + " = " + topo_name +
                "_origin_" + coord + " + " + my_index_name + " * " + topo_name +
                "_spacing_d" + coord + ";\n");
  }
  else
  {
    code.insert((declare ? "double " : "") + res_name + " = " + topo_name +
                "_coords_" + coord + "[" + my_index_name + "];\n");
  }
}

void
TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code,
                         const std::string &index_name,
                         const bool index_array,
                         const std::string &res_name,
                         const bool declare)
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
TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code)
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

void
TopologyCode::dxdydz(InsertionOrderedSet<std::string> &code)
{
  if(topo_type == "rectilinear")
  {
    element_idx(code);
    code.insert("double " + topo_name + "_dx = " + topo_name + "_coords_x[" +
                topo_name + "_element_idx[0] + 1] - " + topo_name +
                "_coords_x[" + topo_name + "_element_idx[0]];\n");
    if(num_dims >= 2)
    {
      code.insert("double " + topo_name + "_dy = " + topo_name + "_coords_y[" +
                  topo_name + "_element_idx[1] + 1] - " + topo_name +
                  "_coords_y[" + topo_name + "_element_idx[1]];\n");
    }
    if(num_dims == 3)
    {
      code.insert({"double " + topo_name + "_dz = " + topo_name + "_coords_z[" +
                   topo_name + "_element_idx[2] + 1] - " + topo_name +
                   "_coords_z[" + topo_name + "_element_idx[2]];\n"});
    }
  }
}

// https://www.osti.gov/servlets/purl/632793 (14)
// switch vertices 2 and 3 and vertices 6 and 7 to match vtk order
void
TopologyCode::hexahedral_volume(InsertionOrderedSet<std::string> &code,
                                const std::string &vertices_name,
                                const std::string &res_name)
{
  math_code.vector_subtract(
      code, vertices_name + "[6]", vertices_name + "[0]", res_name + "_6m0", 3);
  math_code.vector_subtract(
      code, vertices_name + "[1]", vertices_name + "[0]", res_name + "_1m0", 3);
  math_code.vector_subtract(
      code, vertices_name + "[2]", vertices_name + "[5]", res_name + "_2m5", 3);
  math_code.vector_subtract(
      code, vertices_name + "[4]", vertices_name + "[0]", res_name + "_4m0", 3);
  math_code.vector_subtract(
      code, vertices_name + "[5]", vertices_name + "[7]", res_name + "_5m7", 3);
  math_code.vector_subtract(
      code, vertices_name + "[3]", vertices_name + "[0]", res_name + "_3m0", 3);
  math_code.vector_subtract(
      code, vertices_name + "[7]", vertices_name + "[2]", res_name + "_7m2", 3);
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
  code.insert("double " + res_name + " = (" + res_name + "_det0 + " + res_name +
              "_det1 + " + res_name + "_det2) / 6.0;\n");
}

// ||(p3-p0) (p2-p0) (p1-p0)|| / 6
void
TopologyCode::tetrahedral_volume(InsertionOrderedSet<std::string> &code,
                                 const std::string &vertices_name,
                                 const std::string &res_name)
{
  math_code.vector_subtract(
      code, vertices_name + "[1]", vertices_name + "[0]", res_name + "_1m0", 3);
  math_code.vector_subtract(
      code, vertices_name + "[2]", vertices_name + "[0]", res_name + "_2m0", 3);
  math_code.vector_subtract(
      code, vertices_name + "[3]", vertices_name + "[0]", res_name + "_3m0", 3);
  math_code.determinant_3x3(code,
                            res_name + "_3m0",
                            res_name + "_2m0",
                            res_name + "_1m0",
                            res_name + "_det");
  code.insert("double " + res_name + " = " + res_name + "_det / 6.0;\n");
}

// 1/2 * |(p2 - p0) X (p3 - p1)|
void
TopologyCode::quadrilateral_volume(InsertionOrderedSet<std::string> &code,
                                   const std::string &vertices_name,
                                   const std::string &res_name)
{
  math_code.vector_subtract(code,
                            vertices_name + "[2]",
                            vertices_name + "[0]",
                            res_name + "_2m0",
                            num_dims);
  math_code.vector_subtract(code,
                            vertices_name + "[3]",
                            vertices_name + "[1]",
                            res_name + "_3m1",
                            num_dims);
  math_code.cross_product(code,
                          res_name + "_2m0",
                          res_name + "_3m1",
                          res_name + "_cross",
                          num_dims);
  math_code.magnitude(code, res_name + "_cross", res_name + "_cross_mag", 3);
  code.insert("double " + res_name + " = " + res_name + "_cross_mag / 2.0;\n");
}

// 1/2 * |(p1 - p0) X (p2 - p0)|
void
TopologyCode::triangle_volume(InsertionOrderedSet<std::string> &code,
                              const std::string &vertices_name,
                              const std::string &res_name)
{
  math_code.vector_subtract(code,
                            vertices_name + "[1]",
                            vertices_name + "[0]",
                            res_name + "_1m0",
                            num_dims);
  math_code.vector_subtract(code,
                            vertices_name + "[2]",
                            vertices_name + "[0]",
                            res_name + "_2m0",
                            num_dims);
  math_code.cross_product(code,
                          res_name + "_1m0",
                          res_name + "_2m0",
                          res_name + "_cross",
                          num_dims);
  math_code.magnitude(code, res_name + "_cross", res_name + "_cross_mag", 3);
  code.insert("double " + res_name + " = " + res_name + "_cross_mag / 2.0;\n");
}

// http://index-of.co.uk/Game-Development/Programming/Graphics%20Gems%205.pdf
// k is # vertices, h = (k-1)//2, l = 0 if k is odd, l = k-1 if k is even
// 2A = sum_{i=1}^{h - 1}((P_2i - P_0) X (P_2i+1 - P_2i-1)) +
// (P_2h - P_0) X (P_l - P_2h-1)
void
TopologyCode::polygon_volume_vec(InsertionOrderedSet<std::string> &code,
                                 const std::string &vertices_name,
                                 const std::string &res_name)
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
                            vertices_name + "[2 * i]",
                            vertices_name + "[0]",
                            res_name + "_2im0",
                            num_dims);
  math_code.vector_subtract(for_loop,
                            vertices_name + "[2 * i + 1]",
                            vertices_name + "[2 * i - 1]",
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
                            vertices_name + "[2 * " + res_name + "_h]",
                            vertices_name + "[0]",
                            res_name + "_2hm0",
                            num_dims);
  math_code.vector_subtract(code,
                            vertices_name + "[" + res_name + "_last]",
                            vertices_name + "[2 * " + res_name + "_h - 1]",
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
TopologyCode::polygon_volume(InsertionOrderedSet<std::string> &code,
                             const std::string &vertices_name,
                             const std::string &res_name)
{
  polygon_volume_vec(code, vertices_name, res_name);
  math_code.magnitude(code, res_name + "_vec", res_name + "_vec_mag", 3);
  code.insert("double " + res_name + " = " + res_name + "_vec_mag / 2.0;\n");
}

// TODO this doesn't work because A_j needs to point outside the polyhedron
// m is number of faces
// 1/6 * sum_{j=0}^{m-1}(P_j . 2*A_j)
// P_j is some point on face j
// A_j is the area vector of face j
void
TopologyCode::polyhedron_volume(InsertionOrderedSet<std::string> &code,
                                const std::string &vertices_name,
                                const std::string &res_name)
{
  code.insert({"double " + res_name + "_vec[3];\n",
               res_name + "_vec[0] = 0;\n",
               res_name + "_vec[1] = 0;\n",
               res_name + "_vec[2] = 0;\n",
               "int " + topo_name + "_polyhedral_shape_size = " + topo_name +
                   "_polyhedral_sizes[item];\n",
               "int " + topo_name + "_polyhedral_offset = " + topo_name +
                   "_polyhedral_offsets[item];\n"});

  InsertionOrderedSet<std::string> for_loop;
  for_loop.insert(
      {"for(int j = 0; j < " + topo_name + "_polyhedral_shape_size; ++j)\n",
       "{\n"});
  unstructured_vertices(for_loop,
                        topo_name + "_polyhedral_connectivity[" + topo_name +
                            "_polyhedral_offset + j]");
  polygon_volume_vec(for_loop, topo_name + "_vertex_locs", res_name + "_face");
  math_code.dot_product(for_loop,
                        topo_name + "_vertex_locs[4]",
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
  math_code.magnitude(code, res_name + "_vec", res_name + "_vec_mag", num_dims);
  code.insert("double " + res_name + " = " + res_name + "_vec_mag / 6.0;\n");
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
    element_idx(code);
    structured_vertices(code);
    if(num_dims == 3)
    {
      hexahedral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(num_dims == 2)
    {
      quadrilateral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(num_dims == 1)
    {
      math_code.vector_subtract(code,
                                topo_name + "vertex_locs[1]",
                                topo_name + "vertex_locs[0]",
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
    if(shape == "hex")
    {
      unstructured_vertices(code);
      hexahedral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(shape == "tet")
    {
      unstructured_vertices(code);
      tetrahedral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(shape == "quad")
    {
      unstructured_vertices(code);
      quadrilateral_volume(
          code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(shape == "tri")
    {
      unstructured_vertices(code);
      triangle_volume(code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    else if(shape == "polygonal")
    {
      unstructured_vertices(code);
      polygon_volume(code, topo_name + "_vertex_locs", topo_name + "_volume");
    }
    // else if(shape == "polyhedral")
    // {
    //   polyhedron_volume(
    //       code, topo_name + "_vertex_locs", topo_name + "_volume");
    // }
    else
    {
      ASCENT_ERROR("volume for unstructured topology with shape '"
                   << shape << "' is not implemented.");
    }
  }
}
// }}}

//-----------------------------------------------------------------------------
// -- FieldCode
//-----------------------------------------------------------------------------
// {{{
FieldCode::FieldCode(const std::string &field_name,
                     const std::string &topo_name,
                     const std::string &association,
                     const conduit::Node &domain)
    : field_name(field_name), association(association),
      topo_code(topo_name, domain)
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
    res += "double ";
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

void
FieldCode::gradient(InsertionOrderedSet<std::string> &code)
{
  if(topo_code.topo_type != "uniform" && topo_code.topo_type != "rectilinear" &&
     topo_code.topo_type != "structured")
  {
    ASCENT_ERROR(
        "Unsupported topo_type: '"
        << topo_code.topo_type
        << "'. Gradient is not implemented for unstructured topologies.");
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
  const std::string gradient_name = field_name + "_gradient";
  const std::string upper = gradient_name + "_upper";
  const std::string lower = gradient_name + "_lower";
  code.insert({"double " + gradient_name + "[3];\n",
               "double " + upper + ";\n",
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
      upper_body.insert(upper + " = " + field_name + "[" + upper + "_idx];\n");
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
      lower_body.insert(lower + " = " + field_name + "[" + lower + "_idx];\n");
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
                  "_loc;\n");

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
// }}}

//-----------------------------------------------------------------------------
// -- Kernel
//-----------------------------------------------------------------------------
// {{{
void
Kernel::fuse_kernel(const Kernel &from)
{
  kernel_body = kernel_body + from.kernel_body;
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

// clang-format off
std::string
Kernel::generate_loop(const std::string& output) const
{
  std::string res =
    "for (int group = 0; group < entries; group += 128; @outer)\n"
       "{\n"
         "for (int item = group; item < (group + 128); ++item; @inner)\n"
         "{\n"
           "if (item < entries)\n"
           "{\n" +
              for_body.accumulate();
              if(num_components > 1)
              {
                for(int i = 0; i < num_components; ++i)
                {
                  res += output + "[" + std::to_string(num_components) +
                    " * item + " + std::to_string(i) + "] = " + expr +
                    "[" + std::to_string(i) + "];\n";
                }
              }
              else
              {
                res += output + "[item] = " + expr + ";\n";
              }
  res +=
           "}\n"
         "}\n"
       "}\n";
  return res;
}
// clang-format on
// }}}

//-----------------------------------------------------------------------------
// -- Jitable
//-----------------------------------------------------------------------------
// {{{
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
  kernel_string += kernel.generate_loop("output_ptr");
  kernel_string += "}";
  return detail::indent_code(kernel_string, 0);
}

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
      // allow the expression to have multiple topologies but we'll need a way
      // of figuring out where to output things
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
  static bool device_set = false;
  if(!device_set)
  {
    // running this in a loop segfaults...
    occa::setDevice("mode: 'Serial'");
    device_set = true;
  }
  occa::device &device = occa::getDevice();
  occa::kernel occa_kernel;
  // occa::array<double> l_output(27000);
  // return;

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

    const Kernel &kernel = kernels.at(cur_dom_info["kernel_type"].as_string());

    const std::string kernel_string = generate_kernel(i);

    const int entries = cur_dom_info["entries"].to_int64();

    // std::cout << kernel_string << std::endl;

    try
    {
      occa_kernel = device.buildKernelFromString(kernel_string, "map");
    }
    catch(const occa::exception &e)
    {
      ASCENT_ERROR("Jitable: Expression compilation failed:\n"
                   << e.what() << "\n\n"
                   << kernel_string);
      // ASCENT_ERROR("Jitable: Expression compilation failed:\n"
      //              << e.what() << "\n\n"
      //              << cur_dom_info.to_yaml() << kernel_string);
    }
    catch(...)
    {
      ASCENT_ERROR(
          "Jitable: Expression compilation failed with an unknown error.\n\n"
          << kernel_string);
      // ASCENT_ERROR(
      //     "Jitable: Expression compilation failed with an unknown error.\n"
      //     << cur_dom_info.to_yaml() << kernel_string);
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
    conduit::float64 *output_ptr;
    if(kernel.num_components > 1)
    {
      conduit::Schema s;
      for(int i = 0; i < kernel.num_components; ++i)
      {
        s[std::string(1, 'x' + i)].set(conduit::DataType::float64(
            entries,
            sizeof(conduit::float64) * i,
            sizeof(conduit::float64) * kernel.num_components));
      }
      n_output["values"].set(s);
    }
    else
    {
      n_output["values"] =
          conduit::DataType::float64(entries * kernel.num_components);
    }
    output_ptr = (conduit::float64 *)n_output["values"].data_ptr();
    occa::array<double> o_output(entries * kernel.num_components);

    occa_kernel.pushArg(o_output);
    occa_kernel.run();

    o_output.memory().copyTo(output_ptr);

    // dom["fields/" + field_name].print();
  }
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
