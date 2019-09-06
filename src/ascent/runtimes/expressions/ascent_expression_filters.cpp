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
#include "ascent_conduit_reductions.hpp"
#include "ascent_blueprint_architect.hpp"
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

#include <limits>
#include <math.h>
#include <typeinfo>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif
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
// -- begin ascent::runtime::expressions --
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{

bool is_math(const std::string &op)
{
  return op == "*"
         || op == "+"
         || op == "/"
         || op == "-"
         || op == "%";
}

bool is_logic(const std::string &op)
{
  return op == "or"
         || op == "and"
         || op == "not";
}

bool is_scalar(const std::string &type)
{
  return type == "int"
         || type == "double"
         || type == "scalar";
}

void
vector_op(const double lhs[3],
          const double rhs[3],
          const std::string &op,
          double res[3])
{

  if(op == "+")
  {
    res[0] = lhs[0] + rhs[0];
    res[1] = lhs[1] + rhs[1];
    res[2] = lhs[2] + rhs[2];
  }
  else if(op == "-")
  {
    res[0] = lhs[0] - rhs[0];
    res[1] = lhs[1] - rhs[1];
    res[2] = lhs[2] - rhs[2];
  }
  else
  {
    ASCENT_ERROR("Unsupported vector op "<<op);
  }
}


template<typename T>
T math_op(const T lhs, const T rhs, const std::string &op)
{
  ASCENT_ERROR("unknown type: "<<typeid(T).name());
}

template<>
double math_op(const double lhs, const double rhs, const std::string &op)
{
  double res;
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
    ASCENT_ERROR("unknown math op "<<op<<" for type double");
  }
  return res;
}

template<>
int math_op(const int lhs, const int rhs, const std::string &op)
{
  int res;
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
  else if(op == "%")
  {
    res = lhs % rhs;
  }
  else
  {
    ASCENT_ERROR("unknown math op "<<op<<" for type int");
  }
  return res;
}

bool comp_op(const double lhs, const double rhs, const std::string &op)
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
  else if(op == "==")
  {
    res = lhs == rhs;
  }
  else if(op == "!=")
  {
    res = lhs != rhs;
  }
  else
  {
    ASCENT_ERROR("unknown comparison op "<<op);
  }
  return res;
}

bool logic_op(const bool lhs, const bool rhs, const std::string &op)
{
  bool res;
  if(op == "or")
  {
    res = lhs || rhs;
  }
  else if(op == "and")
  {
    res = lhs && rhs;
  }
  else if(op == "not")
  {
    res = !rhs;
  }
  else
  {
    ASCENT_ERROR("unknown boolean op "<<op);
  }
  return res;
}

} // namespace detail

//-----------------------------------------------------------------------------
NullArg::NullArg()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
NullArg::~NullArg()
{
// empty
}

//-----------------------------------------------------------------------------
void
NullArg::declare_interface(Node &i)
{
  i["type_name"]   = "null_arg";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
NullArg::verify_params(const conduit::Node &params,
                       conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
NullArg::execute()
{
  conduit::Node *output = new conduit::Node();
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Identifier::Identifier()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Identifier::~Identifier()
{
// empty
}

//-----------------------------------------------------------------------------
void
Identifier::declare_interface(Node &i)
{
  i["type_name"]   = "expr_identifier";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Identifier::verify_params(const conduit::Node &params,
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
Identifier::execute()
{
  conduit::Node *output = new conduit::Node();
  std::string i_name = params()["value"].as_string();

  conduit::Node *cache = graph().workspace().registry().fetch<Node>("cache");
  if(!cache->has_path(i_name))
  {
    ASCENT_ERROR("Unknown expression identifier: '"<<i_name<<"'");
  }

  const int entries = (*cache)[i_name].number_of_children();
  if(entries < 1)
  {
    ASCENT_ERROR("Expression identifier: needs a non-zero number of entires: "<<entries);
  }
  // grab the last one calculated
  (*output) = (*cache)[i_name].child(entries - 1);
  (*output)["attrs/history"] = (*cache)[i_name];
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Boolean::Boolean()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Boolean::~Boolean()
{
// empty
}

//-----------------------------------------------------------------------------
void
Boolean::declare_interface(Node &i)
{
  i["type_name"]   = "expr_bool";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Boolean::verify_params(const conduit::Node &params,
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
Boolean::execute()
{
  conduit::Node *output = new conduit::Node();
  (*output)["value"] = params()["value"].to_uint8();
  (*output)["type"] = "bool";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
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
  (*output)["value"] = params()["value"].to_int32();
  (*output)["type"] = "int";
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
  (*output)["value"] = params()["value"].to_float64();
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
String::String()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
String::~String()
{
// empty
}

//-----------------------------------------------------------------------------
void
String::declare_interface(Node &i)
{
  i["type_name"]   = "expr_string";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
String::verify_params(const conduit::Node &params,
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
String::execute()
{
  conduit::Node *output = new conduit::Node();

  (*output)["value"] = params()["value"].as_string();
  (*output)["type"] = "string";
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
  Node *n_lhs = input<Node>("lhs");
  Node *n_rhs = input<Node>("rhs");

  const Node &lhs = (*n_lhs)["value"];
  const Node &rhs = (*n_rhs)["value"];

  std::string op_str = params()["op_string"].as_string();
  const std::string l_type = (*n_lhs)["type"].as_string();
  const std::string r_type = (*n_rhs)["type"].as_string();

  conduit::Node *output = new conduit::Node();
  std::stringstream msg;

  if(detail::is_math(op_str))
  {
    if(detail::is_scalar(l_type) && detail::is_scalar(r_type))
    {
      // promote to double if at least one is a double
      if(l_type == "double" || r_type == "double")
      {
        double d_rhs = rhs.to_float64();
        double d_lhs = lhs.to_float64();
        (*output)["value"] = detail::math_op(d_lhs, d_rhs, op_str);
        (*output)["type"] = "double";
      }
      else
      {
        int i_rhs = rhs.to_int32();
        int i_lhs = lhs.to_int32();
        (*output)["value"] = detail::math_op(i_lhs, i_rhs, op_str);
        (*output)["type"] = "int";
      }
    }
    else
    {
      if(detail::is_scalar(l_type) != detail::is_scalar(r_type))
      {
        msg << "' " << l_type << " " << op_str << " " << r_type << "'";
        ASCENT_ERROR("Mixed vector and scalar quantities not implemented / supported: " << msg.str());
      }

      double res[3];
      detail::vector_op(lhs.value(),
                        rhs.value(),
                        op_str,
                        res);


      (*output)["value"].set(res, 3);
      (*output)["type"] = "vector";
    }
  }
  else if (detail::is_logic(op_str))
  {
    bool b_rhs = rhs.to_uint8();
    bool b_lhs = lhs.to_uint8();
    (*output)["value"] = detail::logic_op(b_rhs, b_lhs, op_str);
    (*output)["type"] = "bool";
  }
  else
  {
    double d_rhs = rhs.to_float64();
    double d_lhs = lhs.to_float64();
    (*output)["value"] = detail::comp_op(d_lhs, d_rhs, op_str);
    (*output)["type"] = "bool";
  }

  //std::cout<<" operation "<<op_str<<"\n";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
IfExpr::IfExpr()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
IfExpr::~IfExpr()
{
// empty
}

//-----------------------------------------------------------------------------
void
IfExpr::declare_interface(Node &i)
{
  i["type_name"]   = "expr_if";
  i["port_names"].append() = "condition";
  i["port_names"].append() = "if";
  i["port_names"].append() = "else";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
IfExpr::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
IfExpr::execute()
{
  Node *n_condition = input<Node>("condition");
  Node *n_if = input<Node>("if");
  Node *n_else = input<Node>("else");

  Node *output;
  if((*n_condition)["value"].to_uint8() == 1)
  {
    output = n_if;
  }
  else
  {
    output = n_else;
  }

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ScalarMin::ScalarMin()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
ScalarMin::~ScalarMin()
{
// empty
}

//-----------------------------------------------------------------------------
void
ScalarMin::declare_interface(Node &i)
{
  i["type_name"]   = "scalar_min";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "arg2";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ScalarMin::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
ScalarMin::execute()

{
  Node *arg1 = input<Node>("arg1");
  Node *arg2 = input<Node>("arg2");

  conduit::Node *output = new conduit::Node();

  if((*arg1)["type"].as_string() == "double"
    || (*arg2)["type"].as_string() == "double")
  {
    double d_rhs = (*arg1)["value"].to_float64();
    double d_lhs = (*arg2)["value"].to_float64();
    (*output)["value"] = std::min(d_lhs, d_rhs);
    (*output)["type"] = "double";
  }
  else
  {
    int i_rhs = (*arg1)["value"].to_int32();
    int i_lhs = (*arg2)["value"].to_int32();
    (*output)["value"] = std::min(i_lhs, i_rhs);
    (*output)["type"] = "int";
  }

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

  conduit::Node *output = new conduit::Node();

  if((*arg1)["type"].as_string() == "double"
    || (*arg2)["type"].as_string() == "double")
  {
    double d_rhs = (*arg1)["value"].to_float64();
    double d_lhs = (*arg2)["value"].to_float64();
    (*output)["value"] = std::max(d_lhs, d_rhs);
    (*output)["type"] = "double";
  }
  else
  {
    int i_rhs = (*arg1)["value"].to_int32();
    int i_lhs = (*arg2)["value"].to_int32();
    (*output)["value"] = std::max(i_lhs, i_rhs);
    (*output)["type"] = "int";
  }

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldMin::FieldMin()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
FieldMin::~FieldMin()
{
// empty
}

//-----------------------------------------------------------------------------
void
FieldMin::declare_interface(Node &i)
{
  i["type_name"]   = "field_min";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldMin::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
FieldMin::execute()

{

  Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *output = new conduit::Node();

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("FieldMin: field '"<<field<<"' is not a scalar field");
  }

  conduit::Node n_min = field_min(*dataset, field);

  (*output)["type"] = "value_position";
  (*output)["attrs/value/value"] = n_min["value"];
  (*output)["attrs/value/type"] = "double";
  (*output)["attrs/position/value"] = n_min["position"];
  (*output)["attrs/position/type"] = "vector";

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

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *output = new conduit::Node();

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("FieldMax: field '"<<field<<"' is not a scalar field");
  }

  conduit::Node n_max = field_max(*dataset, field);

  (*output)["type"] = "value_position";
  (*output)["attrs/value/value"] = n_max["value"];
  (*output)["attrs/value/type"] = "double";
  (*output)["attrs/position/value"] = n_max["position"];
  (*output)["attrs/position/type"] = "vector";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldAvg::FieldAvg()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
FieldAvg::~FieldAvg()
{
// empty
}

//-----------------------------------------------------------------------------
void
FieldAvg::declare_interface(Node &i)
{
    i["type_name"]   = "field_avg";
    i["port_names"].append() = "arg1";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldAvg::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
FieldAvg::execute()

{
  Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *output = new conduit::Node();

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("FieldAvg: field '"<<field<<"' is not a scalar field");
  }

  conduit::Node n_avg = field_avg(*dataset, field);

  (*output)["value"] = n_avg["value"];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Cycle::Cycle()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Cycle::~Cycle()
{
// empty
}

//-----------------------------------------------------------------------------
void
Cycle::declare_interface(Node &i)
{
  i["type_name"]   = "cycle";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Cycle::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
Cycle::execute()

{
  conduit::Node *output = new conduit::Node();

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  conduit::Node state = get_state_var(*dataset, "cycle");
  if(!state.dtype().is_number())
  {
    ASCENT_ERROR("Expressions: cycle() is not a number");
  }

  (*output)["type"] = "int";
  (*output)["value"] = state;
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
History::History()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
History::~History()
{
// empty
}

//-----------------------------------------------------------------------------
void
History::declare_interface(Node &i)
{
  i["type_name"]   = "history";
  i["port_names"].append() = "expr_name";
  i["port_names"].append() = "index";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
History::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
History::execute()
{
  conduit::Node *output = new conduit::Node();
  conduit::Node history = (*input<Node>("expr_name"))["attrs/history"];
  int index = (*input<Node>("index"))["value"].as_int32();

  const int entries = history.number_of_children();
  if(entries - index - 1 < 0)
  {
    ASCENT_ERROR("History: found only "<<entries<<" entries, cannot get "<<index<<" entries ago.");
  }

  // grab the value from index cycles ago
  (*output) = history.child(entries - index - 1);
  set_output<conduit::Node>(output);

}

//-----------------------------------------------------------------------------
Vector::Vector()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Vector::~Vector()
{
// empty
}

//-----------------------------------------------------------------------------
void
Vector::declare_interface(Node &i)
{
  i["type_name"]   = "vector";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "arg2";
  i["port_names"].append() = "arg3";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Vector::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
Vector::execute()

{
  Node *arg1 = input<Node>("arg1");
  Node *arg2 = input<Node>("arg2");
  Node *arg3 = input<Node>("arg3");

  double vec[3] = {0., 0., 0.};
  vec[0] = (*arg1)["value"].to_float64();
  vec[1] = (*arg2)["value"].to_float64();
  vec[2] = (*arg3)["value"].to_float64();

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "vector";
  (*output)["value"].set(vec,3);;
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Magnitude::Magnitude()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Magnitude::~Magnitude()
{
// empty
}

//-----------------------------------------------------------------------------
void
Magnitude::declare_interface(Node &i)
{
  i["type_name"]   = "magnitude";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Magnitude::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}


//-----------------------------------------------------------------------------
void
Magnitude::execute()

{
  Node *arg1 = input<Node>("arg1");

  double res = 0.;
  const double *vec = (*arg1)["value"].value();
  res = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "double";
  (*output)["value"] = res;
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Field::Field()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Field::~Field()
{
// empty
}

//-----------------------------------------------------------------------------
void
Field::declare_interface(Node &i)
{
  i["type_name"]   = "field";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Field::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Field::execute()
{
  Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  if(!graph().workspace().registry().has_entry("dataset"))
  {
    ASCENT_ERROR("Field: Missing dataset");
  }

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  if(!has_field(*dataset, field))
  {
    std::vector<std::string> names = dataset->child(0)["fields"].child_names();
    std::stringstream ss;
    ss<<"[";
    for(int i = 0; i < names.size(); ++i)
    {
      ss<<" "<<names[i];
    }
    ss<<"]";
    ASCENT_ERROR("Field: dataset does not contain field '"<<field<<"'"
                 <<" known = "<<ss.str());
  }

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = field;
  (*output)["type"] = "field";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Histogram::Histogram()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Histogram::~Histogram()
{
// empty
}

//-----------------------------------------------------------------------------
void
Histogram::declare_interface(Node &i)
{
  i["type_name"]   = "histogram";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "num_bins";
  i["port_names"].append() = "min_val";
  i["port_names"].append() = "max_val";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Histogram::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Histogram::execute()

{
  Node *arg1 = input<Node>("arg1");
  // optional inputs
  const Node *n_bins = input<Node>("num_bins");
  const Node *n_max = input<Node>("max_val");
  const Node *n_min = input<Node>("min_val");

  if((*arg1)["type"].as_string() != "field")
  {
    ASCENT_ERROR("Histogram: arg1 must be a field");
  }

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *dataset = graph().workspace().registry().fetch<Node>("dataset");

  int num_bins = 256;
  if(!n_bins->dtype().is_empty())
  {
    num_bins = (*n_bins)["value"].as_int32();
  }

  double min_val;
  double max_val;

  // handle the optional inputs
  if(!n_max->dtype().is_empty())
  {
    max_val = (*n_max)["value"].to_float64();
  }
  else
  {
    max_val = field_max(*dataset, field)["value"].to_float64();
  }

  if(!n_min->dtype().is_empty())
  {
    min_val = (*n_min)["value"].to_float64();
  }
  else
  {
    min_val = field_min(*dataset, field)["value"].to_float64();
  }

  if(min_val >= max_val)
  {
    ASCENT_ERROR("Histogram: min value ("<<min_val<<") must be smaller than max ("<<max_val<<")");
  }

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "histogram";
  (*output)["attrs/value/value"] = field_histogram(*dataset, field, min_val, max_val, num_bins)["value"];
  (*output)["attrs/value/type"] = "array";
  (*output)["attrs/min_val/value"] = min_val;
  (*output)["attrs/min_val/type"] = "double";
  (*output)["attrs/max_val/value"] = max_val;
  (*output)["attrs/max_val/type"] = "double";
  (*output)["attrs/num_bins/value"] = num_bins;
  (*output)["attrs/num_bins/type"] = "int";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Entropy::Entropy()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Entropy::~Entropy()
{
// empty
}

//-----------------------------------------------------------------------------
void
Entropy::declare_interface(Node &i)
{
  i["type_name"]   = "entropy";
  i["port_names"].append() = "hist";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Entropy::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Entropy::execute()

{
  const conduit::Node *hist = input<conduit::Node>("hist");

  if((*hist)["type"].as_string() != "histogram")
  {
    ASCENT_ERROR("Entropy: hist must be a histogram");
  }

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = field_entropy(*hist)["value"];
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Pdf::Pdf()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Pdf::~Pdf()
{
// empty
}

//-----------------------------------------------------------------------------
void
Pdf::declare_interface(Node &i)
{
  i["type_name"]   = "pdf";
  i["port_names"].append() = "hist";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Pdf::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Pdf::execute()

{
  const conduit::Node *hist = input<conduit::Node>("hist");

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "histogram";
  (*output)["attrs/value/value"] = field_pdf(*hist)["value"];
  (*output)["attrs/value/type"] = "array";
  (*output)["attrs/min_val"] = (*hist)["attrs/min_val"];
  (*output)["attrs/max_val"] = (*hist)["attrs/max_val"];
  (*output)["attrs/num_bins"] = (*hist)["attrs/num_bins"];
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Cdf::Cdf()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Cdf::~Cdf()
{
// empty
}

//-----------------------------------------------------------------------------
void
Cdf::declare_interface(Node &i)
{
  i["type_name"]   = "cdf";
  i["port_names"].append() = "hist";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Cdf::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Cdf::execute()

{
  const conduit::Node *hist = input<conduit::Node>("hist");

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "histogram";
  (*output)["attrs/value/value"] = field_cdf(*hist)["value"];
  (*output)["attrs/value/type"] = "array";
  (*output)["attrs/min_val"] = (*hist)["attrs/min_val"];
  (*output)["attrs/max_val"] = (*hist)["attrs/max_val"];
  (*output)["attrs/num_bins"] = (*hist)["attrs/num_bins"];
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Quantile::Quantile()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Quantile::~Quantile()
{
// empty
}

//-----------------------------------------------------------------------------
void
Quantile::declare_interface(Node &i)
{
  i["type_name"]   = "quantile";
  i["port_names"].append() = "cdf";
  i["port_names"].append() = "val";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Quantile::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Quantile::execute()

{
  const conduit::Node *n_cdf = input<conduit::Node>("cdf");
  const conduit::Node *n_val = input<conduit::Node>("val");

  const double val = (*n_val)["value"].as_float64();

  if(val < 0 || val > 1)
  {
    ASCENT_ERROR("Quantile: val must be between 0 and 1");
  }

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = quantile(*n_cdf, val)["value"];
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
BinByIndex::BinByIndex()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BinByIndex::~BinByIndex()
{
// empty
}

//-----------------------------------------------------------------------------
void
BinByIndex::declare_interface(Node &i)
{
  i["type_name"]   = "bin_by_index";
  i["port_names"].append() = "hist";
  i["port_names"].append() = "bin";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BinByIndex::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
BinByIndex::execute()

{
  const conduit::Node *n_bin = input<conduit::Node>("bin");
  const conduit::Node *n_hist = input<conduit::Node>("hist");

  int num_bins = (*n_hist)["attrs/num_bins/value"].as_int32(); 
  int bin = (*n_bin)["value"].as_int32();

  if(bin < 0 || bin > num_bins - 1)
  {
    ASCENT_ERROR("BinByIndex: bin index must be within the bounds of hist [0, "<<num_bins - 1<<"]");
  }

  conduit::Node *output = new conduit::Node();
  const double *bins = (*n_hist)["attrs/value/value"].value();
  (*output)["value"] = bins[bin];
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
BinByValue::BinByValue()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BinByValue::~BinByValue()
{
// empty
}

//-----------------------------------------------------------------------------
void
BinByValue::declare_interface(Node &i)
{
  i["type_name"]   = "bin_by_value";
  i["port_names"].append() = "hist";
  i["port_names"].append() = "val";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BinByValue::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
BinByValue::execute()

{
  const conduit::Node *n_val = input<conduit::Node>("val");
  const conduit::Node *n_hist = input<conduit::Node>("hist");

  double val = (*n_val)["value"].to_float64();
  double min_val = (*n_hist)["attrs/min_val/value"].to_float64(); 
  double max_val = (*n_hist)["attrs/max_val/value"].to_float64(); 
  int num_bins = (*n_hist)["attrs/num_bins/value"].as_int32(); 

  if(val < min_val || val > max_val)
  {
    ASCENT_ERROR("BinByValue: val must within the bounds of hist ["<<min_val<<", "<<max_val<<"]");
  }

  const double inv_delta = num_bins / (max_val - min_val);
  int bin = static_cast<int>((val - min_val) * inv_delta);

  conduit::Node *output = new conduit::Node();
  const double *bins = (*n_hist)["attrs/value/value"].value();
  (*output)["value"] = bins[bin];
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ArrayAccess::ArrayAccess()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
ArrayAccess::~ArrayAccess()
{
// empty
}

//-----------------------------------------------------------------------------
void
ArrayAccess::declare_interface(Node &i)
{
  i["type_name"]   = "expr_array";
  i["port_names"].append() = "array";
  i["port_names"].append() = "index";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ArrayAccess::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
  info.reset();
  bool res = true;
  // name will be empty if it's accessed by index
  if(!params.has_path("by_name"))
  {
    info["errors"].append() = "Missing required string parameter 'by_name'";
    res = false;
  }
  return res;
}


//-----------------------------------------------------------------------------
void
ArrayAccess::execute()
{
  Node *n_array = input<Node>("array");
  Node *n_index = input<Node>("index");
  bool by_name = params()["by_name"].to_uint8();

  conduit::Node *output = new conduit::Node();
  if(by_name)
  {
    std::string attr = (*n_index)["value"].as_string();
    (*output) = (*n_array)["attrs"][attr];
  }
  else
  {
    int index = (*n_index)["value"].as_int32();
    int length = (*n_array)["value"].dtype().number_of_elements();
    if(index > length - 1)
    {
      ASCENT_ERROR("ArrayAccess: array index out of bounds: [0,"<<length-1<<"]");
    }
    const double *arr = (*n_array)["value"].value();
    (*output)["value"] = arr[index];
    (*output)["type"] = "double";
  }

  set_output<conduit::Node>(output);
}
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions --
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





