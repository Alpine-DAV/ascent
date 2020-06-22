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

#include <ascent_logging.hpp>

#include <occa.hpp>
#include <cstring>
#include <cmath>
#include <limits>

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

//void override_print(const char *str)
//{
//  std::cout<<"HERE "<<str<<"\n";
//  ASCENT_ERROR("OCCA error" <<str);
//}

void modes()
{
  auto modes = occa::modeMap();
  for(auto mode : modes)
  {
    std::cout<<"mode "<<mode.first<<"\n";
  }
  //occa::io::stderr.setOverride(&override_print);
}

void check_fields(const conduit::Node &dataset, const conduit::Node &vars)
{
  bool valid = true;
  std::vector<std::string> bad_names;
  for(int i = 0; i < vars.number_of_children(); ++i)
  {
    std::string field = vars.child(i).as_string();
    if(!has_field(dataset, field))
    {
      bad_names.push_back(field);
      valid = false;
    }
  }

  if(!valid)
  {
    std::stringstream bad_list;
    bad_list<<"[";
    for(auto bad : bad_names)
    {
      bad_list<<bad<<" ";
    }
    bad_list<<"]";

    std::vector<std::string> names = dataset.child(0)["fields"].child_names();
    std::stringstream ss;
    ss<<"[";
    for(int i = 0; i < names.size(); ++i)
    {
      ss<<" "<<names[i];
    }
    ss<<"]";
    ASCENT_ERROR("Field: dataset does not contain fields '"<<bad_list.str()<<"'"
                 <<" known = "<<ss.str());
  }

  // we have valid fields. Now check for valid assocs
  std::set<std::string> assocs_set;
  std::map<std::string,std::string> assocs_map;
  // we have the same number of vars on each rank so
  // mpi comm is safe (ie. same expression everywhere)
  for(int i = 0; i < vars.number_of_children(); ++i)
  {
    std::string field = vars.child(i).as_string();
    std::string assoc = field_assoc(dataset, field);
    assocs_set.insert(assoc);
    assocs_map[field] = assoc;
  }

  if(assocs_set.size() > 1)
  {
    std::stringstream ss;
    for(auto assoc : assocs_map)
    {
      ss<<assoc.first<<" : "<<assoc.second<<"\n";
    }
    ASCENT_ERROR("Error: expression has fields of mixed assocation."
                 <<" They all must be either 'element' or 'vertex'\n"
                 << ss.str());
  }
}


std::string create_map_kernel(std::map<std::string,std::string> &in_vars,
                              std::string out_type,
                              std::string expr)
{
  std::stringstream ss;
  ss << "@kernel void map(const int entries,\n";
  for(auto var : in_vars)
  {
    ss<<"                 const " << var.second << " *"<< var.first <<"_ptr,\n";
  }
  ss << "                       " << out_type << " *output_ptr)\n"
     << "{\n"
     << "  for (int group = 0; group < entries; group += 128; @outer)\n"
     << "  {\n"
     << "    for (int item = group; item < (group + 128); ++item; @inner)\n"
     << "    {\n"
     << "      const int n = item;\n\n"
     << "      if (n < entries)\n"
     << "      {\n";
  for(auto var : in_vars)
  {
    ss<<"        const " << var.second << " "<< var.first
      <<" = "<<var.first <<"_ptr[n];\n";
  }
  ss << "        " << out_type << " output;\n"
     << "        output = " << expr << ";\n"
     << "        output_ptr[n] = output;\n"
     << "      }\n"
     << "    }\n"
     << "  }\n"
     << '}';
  return ss.str();
}

}; // namespace detail


void do_it(conduit::Node &dataset, std::string expr, const conduit::Node &info)
{
  std::cout<<"doint it\n";
  info.print();
  const int num_domains = dataset.number_of_children();
  std::cout<<"Domains "<<num_domains<<"\n";

  if(info.has_path("vars"))
  {
    detail::check_fields(dataset,info["vars"]);
  }

  // build the kernel
  std::set<std::string> var_names;

  // var_name : type
  std::map<std::string, std::string> var_types;

  for(int i = 0; i < info["vars"].number_of_children(); ++i)
  {
    var_names.insert(info["vars"].child(i).as_string());
  }

  for(auto name : var_names)
  {
    std::string type = field_type(dataset, name);
    var_types[name] = type;
  }

  // indentify constants per domain
  std::set<std::string> constants;

  std::string kernel_str = detail::create_map_kernel(var_types,
                                                     "double", //output type
                                                     expr);

  detail::modes();
  occa::setDevice("mode: 'OpenCL', platform_id: 0, device_id: 2");
  occa::device &device = occa::getDevice();
  occa::kernel kernel;

  try
  {
    kernel = device.buildKernelFromString(kernel_str, "map");
    std::cout<<kernel_str<<"\n";
  }
  catch(const occa::exception &e)
  {
    ASCENT_ERROR("Expression compilation failed:\n"<<e.what());
  }
  catch(...)
  {
    ASCENT_ERROR("Expression compilation failed with an unknown error");
  }

  for(int i = 0; i < num_domains; ++i)
  {
    // TODO: we need to skip domains that don't have what we need
    conduit::Node &dom = dataset.child(i);

    std::string var = info["vars"].child(0).as_string();
    const conduit::Node &field = dom["fields/"+var];
    const int size = field["values"].dtype().number_of_elements();

    kernel.clearArgs();
    int invoke_size = -1;
    std::string assoc;
    std::string topo;

    // these are reference counted
    std::vector<occa::memory> field_memory;

    for(auto vt : var_types)
    {
      const std::string &var_name = vt.first;
      const std::string &type = vt.second;
      const conduit::Node &field = dom["fields/"+var];
      const int size = field["values"].dtype().number_of_elements();
      if(invoke_size < 0)
      {
        // input / output size and field assocs
        invoke_size = size;
        assoc = field["association"].as_string();
        topo = field["topology"].as_string();
      }
      else if(invoke_size != size)
      {
        ASCENT_ERROR("field sizes do not match "<<invoke_size<<" "<<size);
      }
      occa::memory o_vals;
      // we only have two types currently
      // zero copy occa::wrapMemory(???
      if(type == "double")
      {
        const double *vals = field["values"].as_float64_ptr();
        o_vals = device.malloc(size * sizeof(double), vals);
      }
      else
      {
        const float *vals = field["values"].as_float32_ptr();
        o_vals = device.malloc(size * sizeof(float), vals);
      }
      std::cout<<"Mode "<<o_vals.mode()<<"\n";

      field_memory.push_back(o_vals);
    }

    // pass agrs to the kernel
    kernel.pushArg(invoke_size);
    for(auto mem : field_memory)
    {
      kernel.pushArg(mem);
    }

    conduit::Node &n_output = dom["fields/output"];
    n_output["association"] = assoc;
    n_output["topology"] = topo;

    n_output["values"] = conduit::DataType::float64(size);
    double *output_ptr = n_output["values"].as_float64_ptr();
    occa::array<double> o_output(size);

    kernel.pushArg(o_output);
    kernel.run();

    o_output.memory().copyTo(output_ptr);

    dom["fields/output"].print();
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





