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

void
pack_topo(const std::string &topo_name,
          const conduit::Node &dom,
          conduit::Node &args)

{
  const conduit::Node &n_topo = dom["topologies/" + topo_name];
  const std::string coords_name = n_topo["coordset"].as_string();
  const std::string topo_type = n_topo["type"].as_string();
  const conduit::Node &n_coords = dom["coordsets/" + coords_name];
  std::stringstream ss;

  if(topo_type == "uniform")
  {
    const conduit::Node &dims = n_coords["dims"];
    args["const int " + topo_name + "_dims_i,\n"] = dims["i"].to_int32();
    args["const int " + topo_name + "_dims_j,\n"] = dims["j"].to_int32();
    if(n_coords.has_path("dims/k"))
    {
      args["const int " + topo_name + "_dims_k,\n"] = dims["k"].to_int32();
    }

    const conduit::Node &spacing = n_coords["spacing"];
    args["const double " + topo_name + "_spacing_dx,\n"] =
        spacing["dx"].to_float64();
    args["const double " + topo_name + "_spacing_dy,\n"] =
        spacing["dy"].to_float64();
    if(spacing.has_path("dz"))
    {
      args["const double " + topo_name + "_spacing_dz,\n"] =
          spacing["dz"].to_float64();
    }

    const conduit::Node &origin = n_coords["origin"];
    args["const double " + topo_name + "_origin_x,\n"] =
        origin["x"].to_float64();
    args["const double " + topo_name + "_origin_y,\n"] =
        origin["y"].to_float64();
    if(origin.has_path("z"))
    {
      args["const double " + topo_name + "_origin_z,\n"] =
          origin["z"].to_float64();
    }
  }
  else if(topo_type == "rectilinear" || topo_type == "structured")
  {
    const conduit::Node &x_vals = n_coords["values/x"];
    const conduit::Node &y_vals = n_coords["values/y"];
    args["const int " + topo_name + "_dims_i,\n"] =
        x_vals.dtype().number_of_elements();
    args["const int " + topo_name + "_dims_j,\n"] =
        y_vals.dtype().number_of_elements();
    if(n_coords.has_path("values/z"))
    {
      const conduit::Node &z_vals = n_coords["values/z"];
      args["const int " + topo_name + "_dims_k,\n"] =
          z_vals.dtype().number_of_elements();
    }

    args["const double *" + topo_name + "_coords_x,\n"].set_external(x_vals);
    args["const double *" + topo_name + "_coords_y,\n"].set_external(y_vals);
    if(n_coords.has_path("values/z"))
    {
      const conduit::Node &z_vals = n_coords["values/z"];
      args["const double *" + topo_name + "_coords_z,\n"].set_external(z_vals);
    }
  }
  else if(topo_type == "unstructured")
  {
    // TODO pack unstructured mesh
    // "const int dims_x,\n"
    // "const int dims_y,\n"
    // "const int dims_z,\n"
    // "const double * coords_x,\n"
    // "const double * coords_y,\n"
    // "const double * coords_z,\n"
    // "const int * cell_conn,\n"
    // "const int cell_shape,\n";
  }
}

//-----------------------------------------------------------------------------
// clang-format off
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
Kernel::generate_for_body(const std::string& output, bool output_exists) const
{
  std::string new_for_body = for_body + generate_inner_scope();
  if(output_exists)
  {
    new_for_body += output + " = " + expr + ";\n";
  }
  else
  {
    new_for_body += "const double "+output+" = " + expr + ";\n";
  }
  return new_for_body;
}

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

std::string
Jitable::generate_kernel(const int dom_idx) const
{
  const conduit::Node &cur_dom_info = dom_info.child(dom_idx);
  const Kernel &kernel = kernels.at(cur_dom_info["kernel_type"].as_string());
  std::string kernel_string = "@kernel void map(const int entries,\n";
  for(const auto &param : cur_dom_info["args"].child_names())
  {
    kernel_string += param;
  }
  kernel_string += "double *output_ptr)\n{\n";
  kernel_string += kernel.kernel_body;
  kernel_string += kernel.generate_loop("output");
  kernel_string += "}";
  return kernel_string;
}

//-----------------------------------------------------------------------------
TopologyCode::TopologyCode(const std::string &topo_name, const conduit::Node &dom)
{
  const conduit::Node &n_topo = dom["topologies/" + topo_name];
  const std::string coords_name = n_topo["coordset"].as_string();
  this->topo_type = n_topo["type"].as_string();
  this->topo_name = topo_name;
  this->num_dims = topo_dim(topo_name, dom);
}
void TopologyCode::cell_idx(InsertionOrderedSet<std::string> &code)
{
  code.insert({
      "int "+topo_name+"_cell_idx["+std::to_string(num_dims)+"];\n",
      topo_name+"_cell_idx[0] = item % ("+topo_name+"_dims_i - 1);\n",
      topo_name+"_cell_idx[1] = (item / ("+topo_name+"_dims_i - 1)) % ("+topo_name+"_dims_j - 1);\n"});
  if(num_dims == 3)
  {
    code.insert(topo_name+"_cell_idx[2] = item / (("+topo_name+"_dims_i - 1) * ("+topo_name+"_dims_j - 1));\n");
  }
}

void TopologyCode::cell_xyz(InsertionOrderedSet<std::string> &code)
{
  cell_idx(code);
  if(topo_type == "uniform")
  {
    code.insert({
        "double "+topo_name+"_cell_x = "
        +topo_name+"_origin_x + "+topo_name+"_cell_idx[0] * "+topo_name+"_spacing_dx;\n"
    });
    code.insert({
        "double "+topo_name+"_cell_y = "
        +topo_name+"_origin_y + "+topo_name+"_cell_idx[1] * "+topo_name+"_spacing_dy;\n"
    });
    if(num_dims == 3)
    {
    code.insert({
        "double "+topo_name+"_cell_z = "
        +topo_name+"_origin_z + "+topo_name+"_cell_idx[2] * "+topo_name+"_spacing_dz;\n"
    });
    }
  }
  else if (topo_type == "rectilinear")
  {
    code.insert({
        "double "+topo_name+"_cell_x = "
        "("+topo_name+"_coords_x["+topo_name+"_cell_idx[0]] "
        "+ "+topo_name+"_coords_x["+topo_name+"_cell_idx[0] + 1]) / 2;\n",

        "double "+topo_name+"_cell_y = "
        "("+topo_name+"_coords_y["+topo_name+"_cell_idx[1]] "
        "+ "+topo_name+"_coords_y["+topo_name+"_cell_idx[1] + 1]) / 2;\n"
        });
    if(num_dims == 3)
    {
      code.insert(
          "double "+topo_name+"_cell_z = "
          "("+topo_name+"_coords_z["+topo_name+"_cell_idx[2]] "
          "+ "+topo_name+"_coords_z["+topo_name+"_cell_idx[2] + 1]) / 2;\n"
          );
    }
  }
}

void TopologyCode::vertex_idx(InsertionOrderedSet<std::string> &code)
{
  code.insert({
      "dx["+std::to_string(num_dims)+"];\n",
      "dx[0] = item % ("+topo_name+"_dims_i);\n",
      "dx[1] = (item / ("+topo_name+"_dims_i)) % ("+topo_name+"_dims_j);\n"});
  if(num_dims == 3)
  {
    code.insert(topo_name+"_vertex_idx[2] = item / (("+topo_name+"_dims_i) * ("+topo_name+"_dims_j));\n");
  }
}

void TopologyCode::vertex_xyz(InsertionOrderedSet<std::string> &code)
{
  vertex_idx(code);
  code.insert({
      "double "+topo_name+"_vertex_x = "+topo_name+"_coords_x["+topo_name+"_vertex_idx[0]];\n",
      "double "+topo_name+"_vertex_y = "+topo_name+"_coords_y["+topo_name+"_vertex_idx[1]];\n"});
  if(num_dims == 3)
  {
    code.insert("double "+topo_name+"_vertex_z = "+topo_name+"_coords_z["+topo_name+"_vertex_idx[2]];\n");
  }
}

void TopologyCode::dxdydz(InsertionOrderedSet<std::string> &code)
{
  cell_idx(code);
  code.insert({
      "double "+topo_name+"_dx = "+topo_name+"_coords_x["+topo_name+"_cell_idx[0]+1] - "+topo_name+"_coords_x["+topo_name+"_cell_idx[0]];\n",
      "double "+topo_name+"_dy = "+topo_name+"_coords_y["+topo_name+"_cell_idx[1]+1] - "+topo_name+"_coords_y["+topo_name+"_cell_idx[1]];\n"});
  if(num_dims == 3)
  {
    code.insert({"double "+topo_name+"_dz = "+topo_name+"_coords_z["+topo_name+"_cell_idx[2]+1] - "+topo_name+"_coords_z["+topo_name+"_cell_idx[2]];\n"});
  }
}

void TopologyCode::volume(InsertionOrderedSet<std::string> &code)
{
  if(topo_type == "uniform")
  {
    if(num_dims == 3)
    {
      code.insert("double "+topo_name+"_volume = "+topo_name+"_spacing_dx * "+topo_name+"_spacing_dy * "+topo_name+"_spacing_dz;\n");
    }
    else
    {
      code.insert("double "+topo_name+"_volume = "+topo_name+"_spacing_dx * "+topo_name+"_spacing_dy;\n");
    }
  }
  else if(topo_type == "rectilinear")
  {
    dxdydz(code);
    if(num_dims == 3)
    {
      code.insert("double "+topo_name+"_volume = "+topo_name+"_dx * "+topo_name+"_dy * "+topo_name+"_dz;\n");
    }
    else
    {
      code.insert("double "+topo_name+"_volume = "+topo_name+"_dx * "+topo_name+"_dy;\n");
    }
  }
}
//-----------------------------------------------------------------------------

// clang-format on

//-----------------------------------------------------------------------------
// Kernel Class
//-----------------------------------------------------------------------------
void
Kernel::fuse_kernel(const Kernel &from)
{
  kernel_body = kernel_body + from.kernel_body;
  for_body = for_body + from.for_body;
  inner_scope.insert(from.inner_scope);
}

//-----------------------------------------------------------------------------
// Jitable Class
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
        if(to_dom_info["entries"].as_int32() !=
           from_dom_info["entries"].as_int32())
        {
          ASCENT_ERROR("JIT: Failed to fuse kernels due to an incompatible "
                       "number of entries: "
                       << to_dom_info["entries"].as_int32() << " versus "
                       << from_dom_info["entries"].as_int32());
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

    const int entries = cur_dom_info["entries"].as_int32();

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
          const double *vals = arg.as_float64_ptr();
          array_memories.back() = device.malloc(size * sizeof(double), vals);
        }
        else if(arg.dtype().is_float32())
        {
          const float *vals = arg.as_float32_ptr();
          array_memories.back() = device.malloc(size * sizeof(float), vals);
        }
        else if(arg.dtype().is_int32())
        {
          const int *vals = arg.as_int32_ptr();
          array_memories.back() = device.malloc(size * sizeof(int), vals);
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
        ASCENT_ERROR("JIT: Unknown argument type. Argument: " << arg.to_yaml());
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
/*
std::string
remove_duplicate_lines(const std::string &input_str)
{
  std::stringstream ss(input_str);
  std::string line;
  std::unordered_set<std::string> lines;
  std::string output_str;
  while(std::getline(ss, line))
  {
    if(lines.find(line) == lines.end())
    {
      output_str += line + "\n";
      lines.insert(line);
    }
  }
  return output_str;
}

void
remove_duplicate_params(conduit::Node &jitable)
{
  const int num_doms = jitable["dom_info"].number_of_children();
  std::unordered_map<std::string, std::vector<int>> ktype_doms_map;
  for(int dom_idx = 0; dom_idx < num_doms; ++dom_idx)
  {
    ktype_doms_map
        [jitable["dom_info"].child(dom_idx)["kernel_type"].as_string()]
            .push_back(dom_idx);
  }

  for(const auto &ktype_doms : ktype_doms_map)
  {
    conduit::Node &kernel = jitable["kernels/" + ktype_doms.first];
    std::stringstream ss(kernel["params"].as_string());
    std::unordered_set<std::string> lines;
    std::string line;
    std::string params;
    for(int i = 0; std::getline(ss, line); ++i)
    {
      if(lines.find(line) == lines.end())
      {
        params += line + "\n";
        lines.insert(line);
      }
      else
      {
        for(const auto dom_idx : ktype_doms.second)
        {
          jitable["dom_info"].child(dom_idx)["args"].remove(i);
        }
        --i;
      }
    }
    kernel["params"] = params;
  }
}
*/

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
