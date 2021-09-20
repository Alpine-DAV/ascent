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
/// file: ascent_jit_topology.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_jit_topology.hpp"
#include <ascent_logging.hpp>
#include "ascent_blueprint_topologies.hpp"
#include <math.h>

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

TopologyCode::TopologyCode(const std::string &topo_name,
                           const conduit::Node &domain,
                           const ArrayCode &array_code)
    : topo_name(topo_name),
      domain(domain),
      array_code(array_code),
      math_code()
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

void
print_vector(InsertionOrderedSet<std::string> &code,
             const std::string v,
             const int num_dims)
{
  std::stringstream ss;
  ss<<"printf(\""+v<<" ";
  for(int i = 0; i < num_dims; ++i)
  {
    ss<<"%f ";
  }
  ss<<"\\n\", ";
  for(int i = 0; i < num_dims; ++i)
  {
    ss<<v<<"["<<i<<"]";
    if(i != num_dims-1)
    {
      ss<<", ";
    }
  }
  ss<<");\n";
  code.insert(ss.str());
}

void
print_int_vector(InsertionOrderedSet<std::string> &code,
             const std::string v,
             const int num_dims)
{
  std::stringstream ss;
  ss<<"printf(\""+v<<" ";
  for(int i = 0; i < num_dims; ++i)
  {
    ss<<"%d ";
  }
  ss<<"\\n\", ";
  for(int i = 0; i < num_dims; ++i)
  {
    ss<<v<<"["<<i<<"]";
    if(i != num_dims-1)
    {
      ss<<", ";
    }
  }
  ss<<");\n";
  code.insert(ss.str());
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
  if(num_dims == 3)
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
  else if(num_dims ==2)
  {
    // 2d cross product is weird so just do it
    // also, this is the signed area, so take the absolute value
    // since I am not currently sure how to ensure the winding order
    code.insert("const double " + res_name + " = abs((" +
                 res_name + "_2m0[0] * " + res_name + "_3m1[1] - " +
                 res_name + "_2m0[1] * " + res_name + "_3m1[0] ) / 2.0); \n");
  }
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
  if(num_dims == 3)
  {
    math_code.cross_product(code,
                            res_name + "_1m0",
                            res_name + "_2m0",
                            res_name + "_cross",
                            num_dims);
    math_code.magnitude(code, res_name + "_cross", res_name + "_cross_mag", 3);
    code.insert("const double " + res_name + " = " + res_name +
                "_cross_mag / 2.0;\n");
  }
  else if(num_dims == 2)
  {
    // 2d cross product is weird so just do it
    // also, this is the signed area, so take the absolute value
    // since I am not currently sure how to ensure the winding order
    code.insert("const double " + res_name + " = abs((" +
                 res_name + "_1m0[0] * " + res_name + "_2m0[1] - " +
                 res_name + "_1m0[1] * " + res_name + "_2m0[0] ) / 2.0); \n");
  }
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
    else
    {
      ASCENT_ERROR("Unsupported unstructured topo_type '"
                   <<topo_type<<"' with shape '"<<shape
                   <<"' for volume calculation");
    }
  }
  else
  {
    ASCENT_ERROR("Unsupported topo_type '"<<topo_type<<"' for volume calculation");
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
  if(num_dims != 2)
  {
    ASCENT_ERROR("'.area' is only defined for 2 dimensional meshes, but "
                 <<" the mesh has topological dims "<<num_dims);
  }

  if(topo_type == "uniform")
  {
    code.insert("const double " + topo_name + "_area = " + topo_name +
                  "_spacing_dx * " + topo_name + "_spacing_dy;\n");
    //else
    //{
    //  // this was originally returning length which I don't think area
    //  // should be defined for anything but 3d
    //  //code.insert("const double " + topo_name + "_area = " + topo_name +
    //  //            "_spacing_dx;\n");
    //}
  }
  else if(topo_type == "rectilinear")
  {
    dxdydz(code);
    code.insert("const double " + topo_name + "_area = " + topo_name +
                "_dx * " + topo_name + "_dy;\n");
    //else
    //{
    //  code.insert("const double " + topo_name + "_area = " + topo_name +
    //              "_dx;\n");
    //}
  }
  else if(topo_type == "structured")
  {
    structured_vertex_locs(code);
    quadrilateral_area(code, topo_name + "_vertex_locs", topo_name + "_area");
    //else if(num_dims == 1)
    //{
    //  math_code.vector_subtract(code,
    //                            topo_name + "vertex_locs[1]",
    //                            topo_name + "vertex_locs[0]",
    //                            topo_name + "_area",
    //                            1);
    //}
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
  else
  {
    ASCENT_ERROR("area for topology type '"<<topo_type
                 <<"' is not implemented.");
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
  else
  {
    ASCENT_ERROR("surface_area for topology type '"<<topo_type
                 <<"' is not implemented.");
  }
}
// }}}

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
