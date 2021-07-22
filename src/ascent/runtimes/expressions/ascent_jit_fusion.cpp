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
/// file: ascent_jit_fusion.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_jit_fusion.hpp"
#include "ascent_blueprint_topologies.hpp"
#include "ascent_blueprint_architect.hpp"
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



JitableFusion::JitableFusion(
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
JitableFusion::binary_op()
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

          out_kernel.num_components = lhs_kernel.num_components;
        }
        else if(op_str == "-")
        {
          MathCode().vector_subtract(out_kernel.for_body,
                                     lhs_expr,
                                     rhs_expr,
                                     filter_name,
                                     lhs_kernel.num_components);

          out_kernel.num_components = lhs_kernel.num_components;
        }
        else if(op_str == "*")
        {
          MathCode().dot_product(out_kernel.for_body,
                                 lhs_expr,
                                 rhs_expr,
                                 filter_name,
                                 lhs_kernel.num_components);

          out_kernel.num_components = 1;
        }
        else
        {
          error = true;
        }
        out_kernel.expr = filter_name;
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
JitableFusion::builtin_functions(const std::string &function_name)
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
JitableFusion::topo_attrs(const conduit::Node &obj, const std::string &name)
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
JitableFusion::expr_dot()
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
JitableFusion::expr_if()
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
JitableFusion::derived_field()
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
JitableFusion::temporary_field(const Kernel &field_kernel,
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
                field_kernel.num_components,
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
JitableFusion::possible_temporary(const int field_port)
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
JitableFusion::gradient(const int field_port, const int component)
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
JitableFusion::gradient()
{
  const int field_port = inputs["field/port"].as_int32();
  gradient(field_port, -1);
}

void
JitableFusion::curl()
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
JitableFusion::recenter()
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
JitableFusion::magnitude()
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
JitableFusion::vector()
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
JitableFusion::binning_value(const conduit::Node &binning)
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
JitableFusion::rand()
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
