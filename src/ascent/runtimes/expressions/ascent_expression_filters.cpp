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
/// file: ascent_expression_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_expression_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

using namespace conduit;
using namespace std;

using namespace flow;

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

bool is_math(const std::string &op)
{
  if(op == "*" || op == "+" || op == "/" || op == "-")
  {
    return true;
  }
  else
  {
    return false;
  }
}

template<typename T>
T math_op(const T& lhs, const T& rhs, const std::string &op)
{
  T res;
  if(op == "+")
  {
    res = lhs + rhs;
  }
  else if(op == "-")
  {
    res = lhs - rhs;
  }
  else if(op == "*")
  {
    res = lhs * rhs;
  }
  else if(op == "/")
  {
    res = lhs / rhs;
  }
  else
  {
    ASCENT_ERROR("unknow math op "<<op);
  }
  return res;
}

template<typename T>
int comp_op(const T& lhs, const T& rhs, const std::string &op)
{
  int res;
  if(op == "<")
  {
    res = lhs < rhs;
  }
  else if(op == "<=")
  {
    res = lhs <= rhs;
  }
  else if(op == ">")
  {
    res = lhs > rhs;
  }
  else if(op == ">=")
  {
    res = lhs >= rhs;
  }
  else
  {
    ASCENT_ERROR("unknow comparison op "<<op);
  }
  return res;
}

} // namespace detail

//-----------------------------------------------------------------------------
Integer::Integer()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Integer::~Integer()
{
// empty
}

//-----------------------------------------------------------------------------
void
Integer::declare_interface(Node &i)
{
    i["type_name"]   = "expr_integer";
    i["port_names"] = DataType::empty();
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Integer::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;
    if(!params.has_path("value"))
    {
       info["errors"].append() = "Missing required numeric parameter 'value'";
       res = false;
    }
    return res;
}


//-----------------------------------------------------------------------------
void
Integer::execute()
{

   conduit::Node *output = new conduit::Node();
   *output = params()["value"].to_int32();
   set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Double::Double()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Double::~Double()
{
// empty
}

//-----------------------------------------------------------------------------
void
Double::declare_interface(Node &i)
{
    i["type_name"]   = "expr_double";
    i["port_names"] = DataType::empty();
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Double::verify_params(const conduit::Node &params,
                      conduit::Node &info)
{
    info.reset();
    bool res = true;
    if(!params.has_path("value"))
    {
       info["errors"].append() = "Missing required numeric parameter 'value'";
       res = false;
    }
    return res;
}


//-----------------------------------------------------------------------------
void
Double::execute()
{

   conduit::Node *output = new conduit::Node();
   *output = params()["value"].to_float64();
   set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
MeshVar::MeshVar()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
MeshVar::~MeshVar()
{
// empty
}

//-----------------------------------------------------------------------------
void
MeshVar::declare_interface(Node &i)
{
    i["type_name"]   = "expr_meshvar";
    i["port_names"] = DataType::empty();
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
MeshVar::verify_params(const conduit::Node &params,
                       conduit::Node &info)
{
    info.reset();
    bool res = true;
    if(!params.has_path("value"))
    {
       info["errors"].append() = "Missing required string parameter 'value'";
       res = false;
    }
    return res;
}

//-----------------------------------------------------------------------------
void
MeshVar::execute()
{
   conduit::Node *output = new conduit::Node();
   *output = params()["value"].as_string();
   set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
BinaryOp::BinaryOp()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BinaryOp::~BinaryOp()
{
// empty
}

//-----------------------------------------------------------------------------
void
BinaryOp::declare_interface(Node &i)
{
    i["type_name"]   = "expr_binary_op";
    i["port_names"].append() = "lhs";
    i["port_names"].append() = "rhs";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BinaryOp::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();
    bool res = true;
    if(!params.has_path("op_string"))
    {
       info["errors"].append() = "Missing required string parameter 'op_string'";
       res = false;
    }
    return res;
}


//-----------------------------------------------------------------------------
void
BinaryOp::execute()
{

  Node *lhs = input<Node>("lhs");
  Node *rhs = input<Node>("rhs");



  lhs->print();
  rhs->print();

  bool has_float = false;

  if(lhs->dtype().is_floating_point() ||
     rhs->dtype().is_floating_point())
  {
    has_float = true;
  }

  conduit::Node *output = new conduit::Node();

  std::string op = params()["op_string"].as_string();
  // promote to double if at one is a double
  bool is_math = detail::is_math(op);

  if(has_float)
  {
    double d_rhs = rhs->to_float64();
    double d_lhs = lhs->to_float64();
    if(is_math)
    {
      *output = detail::math_op(d_lhs, d_rhs, op);
    }
    else
    {
      *output = detail::comp_op(d_lhs, d_rhs, op);
    }
  }
  else
  {
    int i_rhs = rhs->to_int32();
    int i_lhs = lhs->to_int32();
    if(is_math)
    {
      *output = detail::math_op(i_lhs, i_rhs, op);
    }
    else
    {
      *output = detail::comp_op(i_lhs, i_rhs, op);
    }
  }

  std::cout<<" operation "<<op<<"\n";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ScalarMax::ScalarMax()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
ScalarMax::~ScalarMax()
{
// empty
}

//-----------------------------------------------------------------------------
void
ScalarMax::declare_interface(Node &i)
{
    i["type_name"]   = "scalar_max";
    i["port_names"].append() = "arg1";
    i["port_names"].append() = "arg2";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ScalarMax::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();
    bool res = true;
    return res;
}


//-----------------------------------------------------------------------------
void
ScalarMax::execute()

{

  Node *arg1 = input<Node>("arg1");
  Node *arg2 = input<Node>("arg2");


  arg1->print();
  arg2->print();

  bool has_float = false;

  if(arg1->dtype().is_floating_point() ||
     arg2->dtype().is_floating_point())
  {
    has_float = true;
  }

  conduit::Node *output = new conduit::Node();

  if(has_float)
  {
    double d_rhs = arg1->to_float64();
    double d_lhs = arg2->to_float64();
    *output = std::max(d_lhs, d_rhs);
  }
  else
  {
    int i_rhs = arg1->to_int32();
    int i_lhs = arg2->to_int32();
    *output = std::max(i_lhs, i_rhs);
  }

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldMax::FieldMax()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
FieldMax::~FieldMax()
{
// empty
}

//-----------------------------------------------------------------------------
void
FieldMax::declare_interface(Node &i)
{
    i["type_name"]   = "field_max";
    i["port_names"].append() = "arg1";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldMax::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();
    bool res = true;
    return res;
}


//-----------------------------------------------------------------------------
void
FieldMax::execute()

{

  Node *arg1 = input<Node>("arg1");


  arg1->print();

  const std::string field = arg1->as_string();

  conduit::Node *output = new conduit::Node();

  if(!graph().workspace().registry().has_entry("dataset"))
  {
    ASCENT_ERROR("FieldMax: Missing dataset");
  }

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  dataset->print();

  *output = 1.0;

  set_output<conduit::Node>(output);
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





