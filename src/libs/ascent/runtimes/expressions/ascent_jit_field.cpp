//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_jit_field.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_jit_field.hpp"
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
