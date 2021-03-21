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
#include "ascent_blueprint_architect.hpp"
#include "ascent_conduit_reductions.hpp"
#include <ascent_logging.hpp>
#include <utils/ascent_mpi_utils.hpp>
#include <flow_graph.hpp>
#include <flow_timer.hpp>
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
// We want to allow some objects to have basic
// attributes like vectors, but since its a base
// type, its overly burdensome to always set these
// as the return types in every filter. Thus, do this.
void fill_attrs(conduit::Node &obj)
{
  const std::string type = obj["type"].as_string();
  if(type == "vector")
  {
    double *vals = obj["value"].value();
    obj["attrs/x/value"] = vals[0];
    obj["attrs/x/type"] = "double";
    obj["attrs/y/value"] = vals[1];
    obj["attrs/y/type"] = "double";
    obj["attrs/z/value"] = vals[2];
    obj["attrs/z/type"] = "double";
  }
}

bool
is_math(const std::string &op)
{
  return op == "*" || op == "+" || op == "/" || op == "-" || op == "%";
}

bool
is_logic(const std::string &op)
{
  return op == "or" || op == "and" || op == "not";
}

bool
is_scalar(const std::string &type)
{
  return type == "int" || type == "double" || type == "scalar";
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
    ASCENT_ERROR("Unsupported vector op " << op);
  }
}

template <typename T>
T
math_op(const T lhs, const T rhs, const std::string &op)
{
  ASCENT_ERROR("unknown type: " << typeid(T).name());
}

template <>
double
math_op(const double lhs, const double rhs, const std::string &op)
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
    ASCENT_ERROR("unknown math op " << op << " for type double");
  }
  return res;
}

template <>
int
math_op(const int lhs, const int rhs, const std::string &op)
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
    ASCENT_ERROR("unknown math op " << op << " for type int");
  }
  return res;
}

bool
comp_op(const double lhs, const double rhs, const std::string &op)
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
    ASCENT_ERROR("unknown comparison op " << op);
  }
  return res;
}

bool
logic_op(const bool lhs, const bool rhs, const std::string &op)
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
    // a dummy lhs is being passed
    res = !rhs;
  }
  else
  {
    ASCENT_ERROR("unknown boolean op " << op);
  }
  return res;
}

} // namespace detail

//-----------------------------------------------------------------------------
NullArg::NullArg() : Filter()
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
  i["type_name"] = "null_arg";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
NullArg::verify_params(const conduit::Node &params, conduit::Node &info)
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
Identifier::Identifier() : Filter()
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
  i["type_name"] = "expr_identifier";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Identifier::verify_params(const conduit::Node &params, conduit::Node &info)
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

  const conduit::Node *const cache =
      graph().workspace().registry().fetch<Node>("cache");
  if(!cache->has_path(i_name))
  {
    ASCENT_ERROR("Unknown expression identifier: '" << i_name << "'");
  }

  const int entries = (*cache)[i_name].number_of_children();
  if(entries < 1)
  {
    ASCENT_ERROR("Expression identifier: needs at least one entry");
  }
  // grab the last one calculated so we have type info
  (*output) = (*cache)[i_name].child(entries - 1);
  // we need to keep the name to retrieve the chache
  // if history is called.
  (*output)["name"] = i_name;
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Boolean::Boolean() : Filter()
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
  i["type_name"] = "expr_bool";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Boolean::verify_params(const conduit::Node &params, conduit::Node &info)
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
Integer::Integer() : Filter()
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
  i["type_name"] = "expr_integer";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Integer::verify_params(const conduit::Node &params, conduit::Node &info)
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
Double::Double() : Filter()
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
  i["type_name"] = "expr_double";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Double::verify_params(const conduit::Node &params, conduit::Node &info)
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
String::String() : Filter()
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
  i["type_name"] = "expr_string";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
String::verify_params(const conduit::Node &params, conduit::Node &info)
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
BinaryOp::BinaryOp() : Filter()
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
  i["type_name"] = "expr_binary_op";
  i["port_names"].append() = "lhs";
  i["port_names"].append() = "rhs";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BinaryOp::verify_params(const conduit::Node &params, conduit::Node &info)
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
  const conduit::Node *n_lhs = input<Node>("lhs");
  const conduit::Node *n_rhs = input<Node>("rhs");

  const conduit::Node &lhs = (*n_lhs)["value"];
  const conduit::Node &rhs = (*n_rhs)["value"];

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
        ASCENT_ERROR(
            "Mixed vector and scalar quantities not implemented / supported: "
            << msg.str());
      }

      double res[3];
      detail::vector_op(lhs.value(), rhs.value(), op_str, res);

      (*output)["value"].set(res, 3);
      (*output)["type"] = "vector";
    }
  }
  else if(detail::is_logic(op_str))
  {
    bool b_lhs = lhs.to_uint8();
    bool b_rhs = rhs.to_uint8();
    (*output)["value"] = detail::logic_op(b_lhs, b_rhs, op_str);
    (*output)["type"] = "bool";
  }
  else
  {
    double d_rhs = rhs.to_float64();
    double d_lhs = lhs.to_float64();
    (*output)["value"] = detail::comp_op(d_lhs, d_rhs, op_str);
    (*output)["type"] = "bool";
  }

  // std::cout<<" operation "<<op_str<<"\n";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
IfExpr::IfExpr() : Filter()
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
  i["type_name"] = "expr_if";
  i["port_names"].append() = "condition";
  i["port_names"].append() = "if";
  i["port_names"].append() = "else";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
IfExpr::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
IfExpr::execute()
{
  conduit::Node *n_condition = input<Node>("condition");
  conduit::Node *n_if = input<Node>("if");
  conduit::Node *n_else = input<Node>("else");

  conduit::Node *output;
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
ArrayAccess::ArrayAccess() : Filter()
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
  i["type_name"] = "expr_array";
  i["port_names"].append() = "array";
  i["port_names"].append() = "index";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ArrayAccess::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ArrayAccess::execute()
{
  const conduit::Node *n_array = input<Node>("array");
  const conduit::Node *n_index = input<Node>("index");

  conduit::Node *output = new conduit::Node();

  int index = (*n_index)["value"].as_int32();
  int length = (*n_array)["value"].dtype().number_of_elements();
  if(index > length - 1)
  {
    ASCENT_ERROR("ArrayAccess: array index out of bounds: [0," << length - 1
                                                               << "]");
  }
  const double *arr = (*n_array)["value"].value();
  (*output)["value"] = arr[index];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
DotAccess::DotAccess() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
DotAccess::~DotAccess()
{
  // empty
}

//-----------------------------------------------------------------------------
void
DotAccess::declare_interface(Node &i)
{
  i["type_name"] = "expr_dot";
  i["port_names"].append() = "obj";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DotAccess::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  if(!params.has_path("name"))
  {
    info["errors"].append() = "DotAccess: Missing required parameter 'name'";
    res = false;
  }
  return res;
}

//-----------------------------------------------------------------------------
void
DotAccess::execute()
{
  conduit::Node *n_obj = input<Node>("obj");
  std::string name = params()["name"].as_string();

  conduit::Node *output = new conduit::Node();

  // fills attrs for basic types like vectors
  detail::fill_attrs(*n_obj);

  // TODO test accessing non-existant attribute
  if(!n_obj->has_path("attrs/" + name))
  {
    n_obj->print();
    std::stringstream ss;
    if(n_obj->has_path("attrs"))
    {
      std::string attr_yaml = (*n_obj)["attrs"].to_yaml();
      if(attr_yaml == "")
      {
        ss<<" No known attribtues.";
      }
      else
      {
        ss<<" Known attributes: "<<attr_yaml;
      }
    }
    else
    {
      ss<<" No known attributes.";
    }

    ASCENT_ERROR("'"<<name << "' is not a valid object attribute for"
                      <<" type '"<<(*n_obj)["type"].as_string()<<"'."
                      <<ss.str());
  }

  (*output) = (*n_obj)["attrs/" + name];

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ExpressionList::ExpressionList() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
ExpressionList::~ExpressionList()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ExpressionList::declare_interface(Node &i)
{
  i["type_name"] = "expr_list";
  // We can't have an arbitrary number of input ports so we choose 256
  for(int item_num = 0; item_num < 256; ++item_num)
  {
    std::stringstream ss;
    ss << "item" << item_num;
    i["port_names"].append() = ss.str();
  }
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ExpressionList::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ExpressionList::execute()
{
  conduit::Node *output = new conduit::Node();

  for(int item_num = 0; item_num < 256; ++item_num)
  {
    std::stringstream ss;
    ss << "item" << item_num;
    const conduit::Node *n_item = input<Node>(ss.str());
    if(n_item->dtype().is_empty())
    {
      break;
    }
    output->append() = *n_item;
  }

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ArrayMin::ArrayMin() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
ArrayMin::~ArrayMin()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ArrayMin::declare_interface(Node &i)
{
  i["type_name"] = "array_min";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ArrayMin::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ArrayMin::execute()
{
  conduit::Node *output = new conduit::Node();
  (*output)["value"] = array_min((*input<Node>("arg1"))["value"]);
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ScalarMin::ScalarMin() : Filter()
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
  i["type_name"] = "scalar_min";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "arg2";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ScalarMin::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ScalarMin::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");
  const conduit::Node *arg2 = input<Node>("arg2");

  conduit::Node *output = new conduit::Node();

  if((*arg1)["type"].as_string() == "double" ||
     (*arg2)["type"].as_string() == "double")
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
ScalarMax::ScalarMax() : Filter()
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
  i["type_name"] = "scalar_max";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "arg2";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ScalarMax::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ScalarMax::execute()
{

  const conduit::Node *arg1 = input<Node>("arg1");
  const conduit::Node *arg2 = input<Node>("arg2");

  conduit::Node *output = new conduit::Node();

  if((*arg1)["type"].as_string() == "double" ||
     (*arg2)["type"].as_string() == "double")
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
FieldMin::FieldMin() : Filter()
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
  i["type_name"] = "field_min";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldMin::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
FieldMin::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *output = new conduit::Node();

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("FieldMin: field '" << field << "' is not a scalar field");
  }

  conduit::Node n_min = field_min(*dataset, field);

  (*output)["type"] = "value_position";
  (*output)["attrs/value/value"] = n_min["value"];
  (*output)["attrs/value/type"] = "double";
  (*output)["attrs/position/value"] = n_min["position"];
  (*output)["attrs/position/type"] = "vector";
  // information about the element/field
  (*output)["attrs/element/rank"] = n_min["rank"];
  (*output)["attrs/element/domain_index"] = n_min["domain_id"];
  (*output)["attrs/element/index"] = n_min["index"];
  (*output)["attrs/element/assoc"] = n_min["assoc"];

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ArrayMax::ArrayMax() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
ArrayMax::~ArrayMax()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ArrayMax::declare_interface(Node &i)
{
  i["type_name"] = "array_max";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ArrayMax::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ArrayMax::execute()
{
  conduit::Node *output = new conduit::Node();
  (*output)["value"] = array_max((*input<Node>("arg1"))["value"]);
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldMax::FieldMax() : Filter()
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
  i["type_name"] = "field_max";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldMax::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
FieldMax::execute()
{

  const conduit::Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *output = new conduit::Node();

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("FieldMax: field '" << field << "' is not a scalar field");
  }

  conduit::Node n_max = field_max(*dataset, field);

  (*output)["type"] = "value_position";
  (*output)["attrs/value/value"] = n_max["value"];
  (*output)["attrs/value/type"] = "double";
  (*output)["attrs/position/value"] = n_max["position"];
  (*output)["attrs/position/type"] = "vector";
  // information about the element/field
  (*output)["attrs/element/rank"] = n_max["rank"];
  (*output)["attrs/element/domain_index"] = n_max["domain_id"];
  (*output)["attrs/element/index"] = n_max["index"];
  (*output)["attrs/element/assoc"] = n_max["assoc"];

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ArrayAvg::ArrayAvg() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
ArrayAvg::~ArrayAvg()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ArrayAvg::declare_interface(Node &i)
{
  i["type_name"] = "array_avg";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ArrayAvg::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ArrayAvg::execute()
{
  conduit::Node *output = new conduit::Node();
  conduit::Node sum = array_sum((*input<Node>("arg1"))["value"]);
  (*output)["value"] = sum["value"].to_float64() / sum["count"].to_float64();
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldAvg::FieldAvg() : Filter()
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
  i["type_name"] = "field_avg";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldAvg::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
FieldAvg::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  conduit::Node *output = new conduit::Node();

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("FieldAvg: field '" << field << "' is not a scalar field");
  }

  conduit::Node n_avg = field_avg(*dataset, field);

  (*output)["value"] = n_avg["value"];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Cycle::Cycle() : Filter()
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
  i["type_name"] = "cycle";
  i["port_names"] = DataType::empty();
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Cycle::verify_params(const conduit::Node &params, conduit::Node &info)
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

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

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
History::History() : Filter()
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
  i["type_name"] = "history";
  i["port_names"].append() = "expr_name";
  i["port_names"].append() = "absolute_index";
  i["port_names"].append() = "relative_index";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
History::verify_params(const conduit::Node &params, conduit::Node &info)
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

  const std::string expr_name  = (*input<Node>("expr_name"))["name"].as_string();

  const conduit::Node *const cache =
      graph().workspace().registry().fetch<Node>("cache");

  if(!cache->has_path(expr_name))
  {
    ASCENT_ERROR("History: unknown identifier "<<  expr_name);
  }
  const conduit::Node &history = (*cache)[expr_name];

  const conduit::Node *n_absolute_index = input<Node>("absolute_index");
  const conduit::Node *n_relative_index = input<Node>("relative_index");


  if(!n_absolute_index->dtype().is_empty() &&
     !n_relative_index->dtype().is_empty())
  {
    ASCENT_ERROR(
        "History: Specify only one of relative_index or absolute_index.");
  }


  const int entries = history.number_of_children();
  if(!n_relative_index->dtype().is_empty())
  {
    int relative_index = (*n_relative_index)["value"].to_int32();
    if(relative_index >= entries)
    {
      // clamp to first if its gone too far
      relative_index = 0;
    }
    if(relative_index < 0)
    {
      ASCENT_ERROR("History: relative_index must be a non-negative integer.");
    }
    // grab the value from relative_index cycles ago
    (*output) = history.child(entries - relative_index - 1);
  }
  else
  {
    int absolute_index = 0;

    if(!n_absolute_index->has_path("value"))
    {
      ASCENT_ERROR(
          "History: internal error. absolute index does not have child value");
    }
    absolute_index = (*n_absolute_index)["value"].to_int32();

    if(absolute_index >= entries)
    {
      ASCENT_ERROR("History: found only " << entries
                                          << " entries, cannot get entry at "
                                          << absolute_index);
    }
    if(absolute_index < 0)
    {
      ASCENT_ERROR("History: absolute_index must be a non-negative integer.");
    }

    (*output) = history.child(absolute_index);
  }

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Vector::Vector() : Filter()
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
  i["type_name"] = "vector";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "arg2";
  i["port_names"].append() = "arg3";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Vector::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Vector::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");
  const conduit::Node *arg2 = input<Node>("arg2");
  const conduit::Node *arg3 = input<Node>("arg3");

  double vec[3] = {0., 0., 0.};
  vec[0] = (*arg1)["value"].to_float64();
  vec[1] = (*arg2)["value"].to_float64();
  vec[2] = (*arg3)["value"].to_float64();

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "vector";
  (*output)["value"].set(vec, 3);
  ;
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Magnitude::Magnitude() : Filter()
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
  i["type_name"] = "magnitude";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Magnitude::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Magnitude::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");

  double res = 0.;
  const double *vec = (*arg1)["value"].value();
  res = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "double";
  (*output)["value"] = res;
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Field::Field() : Filter()
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
  i["type_name"] = "field";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Field::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Field::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");

  const std::string field = (*arg1)["value"].as_string();

  if(!graph().workspace().registry().has_entry("dataset"))
  {
    ASCENT_ERROR("Field: Missing dataset");
  }

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  if(!has_field(*dataset, field))
  {
    std::vector<std::string> names = dataset->child(0)["fields"].child_names();
    std::stringstream ss;
    ss << "[";
    for(int i = 0; i < names.size(); ++i)
    {
      ss << " " << names[i];
    }
    ss << "]";
    ASCENT_ERROR("Field: dataset does not contain field '"
                 << field << "'"
                 << " known = " << ss.str());
  }

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = field;
  (*output)["type"] = "field";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Axis::Axis() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
Axis::~Axis()
{
  // empty
}

//-----------------------------------------------------------------------------
void
Axis::declare_interface(Node &i)
{
  i["type_name"] = "axis";
  i["port_names"].append() = "name";
  i["port_names"].append() = "min_val";
  i["port_names"].append() = "max_val";
  i["port_names"].append() = "num_bins";
  i["port_names"].append() = "bins";
  i["port_names"].append() = "clamp";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Axis::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Axis::execute()
{
  const std::string name = (*input<Node>("name"))["value"].as_string();
  // uniform binning
  const conduit::Node *n_min = input<Node>("min_val");
  const conduit::Node *n_max = input<Node>("max_val");
  const conduit::Node *n_num_bins = input<Node>("num_bins");
  // rectilinear binning
  const conduit::Node *n_bins_list = input<Node>("bins");
  // clamp
  const conduit::Node *n_clamp = input<conduit::Node>("clamp");

  if(!graph().workspace().registry().has_entry("dataset"))
  {
    ASCENT_ERROR("Field: Missing dataset");
  }
  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, name) && !is_xyz(name))
  {
    std::string known;
    if(dataset->number_of_children() > 0 )
    {
      std::vector<std::string> names = dataset->child(0)["fields"].child_names();
      std::stringstream ss;
      ss << "[";
      for(size_t i = 0; i < names.size(); ++i)
      {
        ss << " '" << names[i]<<"'";
      }
      ss << "]";
      known = ss.str();
    }

    ASCENT_ERROR("Axis: Axes must be scalar fields or x/y/z. Dataset does not "
                 "contain scalar field '"
                 << name << "'. Possible field names "<<known<<".");
  }

  conduit::Node *output;

  if(!n_bins_list->dtype().is_empty())
  {
    // ensure none of the uniform binning arguments are passed
    if(!n_min->dtype().is_empty() || !n_max->dtype().is_empty() ||
       !n_num_bins->dtype().is_empty())
    {
      ASCENT_ERROR("Axis: Only pass in arguments for uniform or rectilinear "
                   "binning, not both.");
    }

    int bins_len = n_bins_list->number_of_children();

    if(bins_len < 2)
    {
      ASCENT_ERROR("Axis: bins must have at least 2 items.");
    }

    output = new conduit::Node();
    (*output)["value/" + name + "/bins"].set(
        conduit::DataType::c_double(bins_len));
    double *bins = (*output)["value/" + name + "/bins"].value();

    for(int i = 0; i < bins_len; ++i)
    {
      const conduit::Node &bin = n_bins_list->child(i);
      if(!detail::is_scalar(bin["type"].as_string()))
      {
        delete output;
        ASCENT_ERROR("Axis: bins must be a list of scalars.");
      }
      bins[i] = bin["value"].to_float64();
      if(i != 0 && bins[i - 1] >= bins[i])
      {
        delete output;
        ASCENT_ERROR("Axis: bins of strictly increasing scalars.");
      }
    }
  }
  else
  {
    output = new conduit::Node();

    double min_val;
    bool min_found = false;
    if(!n_min->dtype().is_empty())
    {
      min_val = (*n_min)["value"].to_float64();
      (*output)["value/" + name + "/min_val"] = min_val;
      min_found = true;
    }
    else if(!is_xyz(name))
    {
      min_val = field_min(*dataset, name)["value"].to_float64();
      (*output)["value/" + name + "/min_val"] = min_val;
      min_found = true;
    }

    double max_val;
    bool max_found = false;
    if(!n_max->dtype().is_empty())
    {
      max_val = (*n_max)["value"].to_float64();
      max_found = true;
      (*output)["value/" + name + "/max_val"] = max_val;
    }
    else if(!is_xyz(name))
    {
      // add 1 because the last bin isn't inclusive
      max_val = field_max(*dataset, name)["value"].to_float64() + 1.0;
      (*output)["value/" + name + "/max_val"] = max_val;
      max_found = true;
    }

    (*output)["value/" + name + "/num_bins"] = 256;
    if(!n_num_bins->dtype().is_empty())
    {
      (*output)["value/" + name + "/num_bins"] =
          (*n_num_bins)["value"].to_int32();
    }

    if(min_found && max_found && min_val >= max_val)
    {
      delete output;
      ASCENT_ERROR("Axis: axis with name '"
                   << name << "': min_val (" << min_val
                   << ") must be smaller than max_val (" << max_val << ")");
    }
  }

  (*output)["value/" + name + "/clamp"] = false;
  if(!n_clamp->dtype().is_empty())
  {
    (*output)["value/" + name + "/clamp"] = (*n_clamp)["value"].to_uint8();
  }

  (*output)["value/" + name];
  (*output)["type"] = "axis";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Histogram::Histogram() : Filter()
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
  i["type_name"] = "histogram";
  i["port_names"].append() = "arg1";
  i["port_names"].append() = "num_bins";
  i["port_names"].append() = "min_val";
  i["port_names"].append() = "max_val";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Histogram::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Histogram::execute()
{
  const conduit::Node *arg1 = input<Node>("arg1");
  // optional inputs
  const conduit::Node *n_bins = input<Node>("num_bins");
  const conduit::Node *n_max = input<Node>("max_val");
  const conduit::Node *n_min = input<Node>("min_val");

  const std::string field = (*arg1)["value"].as_string();

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  if(!is_scalar_field(*dataset, field))
  {
    ASCENT_ERROR("Histogram: axis for histogram must be a scalar field. "
                 "Invalid axis field: '"
                 << field << "'.");
  }

  // handle the optional inputs
  int num_bins = 256;
  if(!n_bins->dtype().is_empty())
  {
    num_bins = (*n_bins)["value"].as_int32();
  }

  double min_val;
  double max_val;

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
    ASCENT_ERROR("Histogram: min value ("
                 << min_val << ") must be smaller than max (" << max_val
                 << ")");
  }

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "histogram";
  (*output)["attrs/value/value"] =
      field_histogram(*dataset, field, min_val, max_val, num_bins)["value"];
  (*output)["attrs/value/type"] = "array";
  (*output)["attrs/min_val/value"] = min_val;
  (*output)["attrs/min_val/type"] = "double";
  (*output)["attrs/max_val/value"] = max_val;
  (*output)["attrs/max_val/type"] = "double";
  (*output)["attrs/num_bins/value"] = num_bins;
  (*output)["attrs/num_bins/type"] = "int";
  (*output)["attrs/clamp/value"] = true;
  (*output)["attrs/clamp/type"] = "bool";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Binning::Binning() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
Binning::~Binning()
{
  // empty
}

//-----------------------------------------------------------------------------
void
Binning::declare_interface(Node &i)
{
  i["type_name"] = "binning";
  i["port_names"].append() = "reduction_var";
  i["port_names"].append() = "reduction_op";
  i["port_names"].append() = "bin_axes";
  i["port_names"].append() = "empty_bin_val";
  i["port_names"].append() = "component";
  i["port_names"].append() = "output";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Binning::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Binning::execute()
{
  const std::string reduction_var =
      (*input<Node>("reduction_var"))["value"].as_string();
  const std::string reduction_op =
      (*input<Node>("reduction_op"))["value"].as_string();
  const conduit::Node *n_axes_list = input<Node>("bin_axes");
  // optional arguments
  const conduit::Node *n_empty_bin_val = input<conduit::Node>("empty_bin_val");
  const conduit::Node *n_component = input<conduit::Node>("component");
  const conduit::Node *n_output_opt = input<conduit::Node>("output");

  std::string component = "";
  if(!n_component->dtype().is_empty())
  {
    component = (*n_component)["value"].as_string();
  }

  conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  // verify n_axes_list and put the values in n_bin_axes
  conduit::Node n_bin_axes;
  int num_axes = n_axes_list->number_of_children();
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = n_axes_list->child(i);
    if(axis["type"].as_string() != "axis")
    {
      ASCENT_ERROR("Binning: bin_axes must be a list of axis");
    }
    n_bin_axes.update(axis["value"]);
  }

  // verify reduction_var
  if(reduction_var.empty())
  {
    if(reduction_op != "sum" && reduction_op != "pdf")
    {
      ASCENT_ERROR("Binning: reduction_var can only be left empty if "
                   "reduction_op is 'sum' or 'pdf'.");
    }
  }
  else if(!is_xyz(reduction_var))
  {
    if(!has_field(*dataset, reduction_var))
    {
      std::string known;
      if(dataset->number_of_children() > 0 )
      {
        std::vector<std::string> names = dataset->child(0)["fields"].child_names();
        std::stringstream ss;
        ss << "[";
        for(size_t i = 0; i < names.size(); ++i)
        {
          ss << " '" << names[i]<<"'";
        }
        ss << "]";
        known = ss.str();
      }
      ASCENT_ERROR("Binning: reduction variable '"
                   << reduction_var
                   << "' must be a scalar field in the dataset or x/y/z or empty."
                   << " known = " << known);
    }

    bool scalar = is_scalar_field(*dataset, reduction_var);
    if(!scalar && component == "")
    {
      ASCENT_ERROR("Binning: reduction variable '"
                   << reduction_var <<"'"
                   << " has multiple components and no 'component' is"
                   << " specified."
                   << " known components = "
                   << possible_components(*dataset, reduction_var));
    }
    if(scalar && component != "")
    {
      ASCENT_ERROR("Binning: reduction variable '"
                   << reduction_var <<"'"
                   << " is a scalar(i.e., has not components "
                   << " but 'component' " << " '"<<component<<"' was"
                   << " specified. Remove the 'component' argument"
                   << " or choose a vector variable.");
    }
    if(!has_component(*dataset, reduction_var, component))
    {
      ASCENT_ERROR("Binning: reduction variable '"
                   << reduction_var << "'"
                   << " does not have component '"<<component<<"'."
                   << " known components = "
                   << possible_components(*dataset, reduction_var));

    }
  }

  // verify reduction_op
  if(reduction_op != "sum" && reduction_op != "min" && reduction_op != "max" &&
     reduction_op != "avg" && reduction_op != "pdf" && reduction_op != "std" &&
     reduction_op != "var" && reduction_op != "rms")
  {
    ASCENT_ERROR(
        "Unknown reduction_op: '"
        << reduction_op
        << "'. Known reduction operators are: cnt, sum, min, max, avg, pdf, "
           "std, var, rms");
  }

  double empty_bin_val = 0;
  if(!n_empty_bin_val->dtype().is_empty())
  {
    empty_bin_val = (*n_empty_bin_val)["value"].to_float64();
  }

  const conduit::Node &n_binning = binning(*dataset,
                                           n_bin_axes,
                                           reduction_var,
                                           reduction_op,
                                           empty_bin_val,
                                           component);

  conduit::Node *output = new conduit::Node();
  (*output)["type"] = "binning";
  (*output)["attrs/value/value"] = n_binning["value"];
  (*output)["attrs/value/type"] = "array";
  (*output)["attrs/reduction_var/value"] = reduction_var;
  (*output)["attrs/reduction_var/type"] = "string";
  (*output)["attrs/reduction_op/value"] = reduction_op;
  (*output)["attrs/reduction_op/type"] = "string";
  (*output)["attrs/bin_axes/value"] = n_bin_axes;
  //(*output)["attrs/bin_axes/type"] = "list";
  (*output)["attrs/association/value"] = n_binning["association"];
  (*output)["attrs/association/type"] = "string";
  set_output<conduit::Node>(output);

  if(!n_output_opt->dtype().is_empty())
  {
    const std::string &output_opt = (*n_output_opt)["value"].as_string();
    if(output_opt != "none" && output_opt != "bins" && output_opt != "mesh")
    {
      ASCENT_ERROR("Unknown ouput_opt: '"
                   << output_opt
                   << "'. Known output options are: 'none', 'bins', 'mesh'.");
    }
    if(output_opt == "bins")
    {
      conduit::Node n_binning_mesh;
      binning_mesh(*output, n_binning_mesh);
      n_binning_mesh["state/cycle"] = 100;
      n_binning_mesh["state/domain_id"] = 0;
      dataset->child(0).update(n_binning_mesh);
    }
    else if(output_opt == "mesh")
    {
      paint_binning(*output, *dataset);
    }
  }
}

//-----------------------------------------------------------------------------
Entropy::Entropy() : Filter()
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
  i["type_name"] = "entropy";
  i["port_names"].append() = "hist";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Entropy::verify_params(const conduit::Node &params, conduit::Node &info)
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
Pdf::Pdf() : Filter()
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
  i["type_name"] = "pdf";
  i["port_names"].append() = "hist";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Pdf::verify_params(const conduit::Node &params, conduit::Node &info)
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
Cdf::Cdf() : Filter()
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
  i["type_name"] = "cdf";
  i["port_names"].append() = "hist";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Cdf::verify_params(const conduit::Node &params, conduit::Node &info)
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
Quantile::Quantile() : Filter()
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
  i["type_name"] = "quantile";
  i["port_names"].append() = "cdf";
  i["port_names"].append() = "q";
  i["port_names"].append() = "interpolation";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Quantile::verify_params(const conduit::Node &params, conduit::Node &info)
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
  const conduit::Node *n_val = input<conduit::Node>("q");
  // optional inputs
  const conduit::Node *n_interpolation = input<conduit::Node>("interpolation");

  const double val = (*n_val)["value"].as_float64();

  if(val < 0 || val > 1)
  {
    ASCENT_ERROR("Quantile: val must be between 0 and 1");
  }

  // handle the optional inputs
  std::string interpolation = "linear";
  if(!n_interpolation->dtype().is_empty())
  {
    interpolation = (*n_interpolation)["value"].as_string();
    if(interpolation != "linear" && interpolation != "lower" &&
       interpolation != "higher" && interpolation != "midpoint" &&
       interpolation != "nearest")
    {
      ASCENT_ERROR("Known interpolation types are: linear, lower, higher, "
                   "midpoint, nearest");
    }
  }

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = quantile(*n_cdf, val, interpolation)["value"];
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
BinByIndex::BinByIndex() : Filter()
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
  i["type_name"] = "bin_by_index";
  i["port_names"].append() = "hist";
  i["port_names"].append() = "bin";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BinByIndex::verify_params(const conduit::Node &params, conduit::Node &info)
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
    ASCENT_ERROR("BinByIndex: bin index must be within the bounds of hist [0, "
                 << num_bins - 1 << "]");
  }

  conduit::Node *output = new conduit::Node();
  const double *bins = (*n_hist)["attrs/value/value"].value();
  (*output)["value"] = bins[bin];
  (*output)["type"] = "double";
  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
BinByValue::BinByValue() : Filter()
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
  i["type_name"] = "bin_by_value";
  i["port_names"].append() = "hist";
  i["port_names"].append() = "val";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BinByValue::verify_params(const conduit::Node &params, conduit::Node &info)
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
    ASCENT_ERROR("BinByValue: val must within the bounds of hist ["
                 << min_val << ", " << max_val << "]");
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
FieldSum::FieldSum() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
FieldSum::~FieldSum()
{
  // empty
}

//-----------------------------------------------------------------------------
void
FieldSum::declare_interface(Node &i)
{
  i["type_name"] = "field_sum";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldSum::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
FieldSum::execute()
{
  std::string field = (*input<Node>("arg1"))["value"].as_string();
  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = field_sum(*dataset, field)["value"];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldNanCount::FieldNanCount() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
FieldNanCount::~FieldNanCount()
{
  // empty
}

//-----------------------------------------------------------------------------
void
FieldNanCount::declare_interface(Node &i)
{
  i["type_name"] = "field_nan_count";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldNanCount::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
FieldNanCount::execute()
{
  std::string field = (*input<Node>("arg1"))["value"].as_string();
  conduit::Node *dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = field_nan_count(*dataset, field)["value"];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
FieldInfCount::FieldInfCount() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
FieldInfCount::~FieldInfCount()
{
  // empty
}

//-----------------------------------------------------------------------------
void
FieldInfCount::declare_interface(Node &i)
{
  i["type_name"] = "field_inf_count";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
FieldInfCount::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
FieldInfCount::execute()
{
  std::string field = (*input<Node>("arg1"))["value"].as_string();
  conduit::Node *dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  conduit::Node *output = new conduit::Node();
  (*output)["value"] = field_inf_count(*dataset, field)["value"];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
ArraySum::ArraySum() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
ArraySum::~ArraySum()
{
  // empty
}

//-----------------------------------------------------------------------------
void
ArraySum::declare_interface(Node &i)
{
  i["type_name"] = "array_sum";
  i["port_names"].append() = "arg1";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
ArraySum::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
ArraySum::execute()
{
  conduit::Node *output = new conduit::Node();
  (*output)["value"] = array_sum((*input<Node>("arg1"))["value"])["value"];
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}
//-----------------------------------------------------------------------------
PointAndAxis::PointAndAxis() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
PointAndAxis::~PointAndAxis()
{
  // empty
}

//-----------------------------------------------------------------------------
void
PointAndAxis::declare_interface(Node &i)
{
  i["type_name"] = "point_and_axis";
  i["port_names"].append() = "binning";
  i["port_names"].append() = "axis";
  i["port_names"].append() = "threshold";
  i["port_names"].append() = "point";
  i["port_names"].append() = "miss_value";
  i["port_names"].append() = "direction";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
PointAndAxis::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
PointAndAxis::execute()
{
  conduit::Node &in_binning = *input<Node>("binning");
  conduit::Node &in_axis =  *input<Node>("axis");
  conduit::Node &in_threshold =  *input<Node>("threshold");
  conduit::Node &in_point =  *input<Node>("point");
  conduit::Node &n_miss_val =  *input<Node>("miss_value");
  conduit::Node &n_dir =  *input<Node>("direction");
  conduit::Node *output = new conduit::Node();

  const int num_axes = in_binning["attrs/bin_axes"].number_of_children();
  if(num_axes > 1)
  {
    ASCENT_ERROR("point_and_axis: only one axis is implemented");
  }

  int direction = 1;
  if(!n_dir.dtype().is_empty())
  {
    direction = n_dir["value"].to_int32();
    if(direction != 1 && direction != -1)
    {
      ASCENT_ERROR("point_and_axis: invalid direction `"<<direction<<"'."
                  <<" Valid directions are 1 or -1.");
    }
  }

  const double point = in_point["value"].to_float64();
  const double threshold = in_threshold["value"].to_float64();

  const conduit::Node &axis = in_binning["attrs/bin_axes/value"].child(0);
  const int num_bins = axis["num_bins"].to_int32();
  const double min_val = axis["min_val"].to_float64();
  const double max_val = axis["max_val"].to_float64();
  const double bin_size = (max_val - min_val) / double(num_bins);

  double *bins = in_binning["attrs/value/value"].value();
  double min_dist = std::numeric_limits<double>::max();
  int index = -1;
  for(int i = 0; i < num_bins; ++i)
  {
    double val = bins[i];
    if(val > threshold)
    {
      double left = min_val + double(i) * bin_size;
      double right = min_val + double(i+1) * bin_size;
      double center = left + (right-left) / 2.0;
      double dist = center - point;
      // skip if distance is behind
      bool behind = dist * double(direction) < 0;

      if(!behind && dist < min_dist)
      {
        min_dist = dist;
        index = i;
      }
    }
  }

  double bin_value = std::numeric_limits<double>::quiet_NaN();

  if(!n_miss_val.dtype().is_empty())
  {
    bin_value = n_miss_val["value"].to_float64();
  }

  // init with miss
  double bin_min = bin_value;
  double bin_max = bin_value;
  double bin_center = bin_value;

  if(index != -1)
  {
    bin_value = bins[index];
    bin_min = min_val + double(index) * bin_size;
    bin_max = min_val + double(index+1) * bin_size;
    bin_center = bin_min + (bin_max-bin_min) / 2.0;
  }

  (*output)["type"] = "bin";

  (*output)["attrs/value/value"] = bin_value;
  (*output)["attrs/value/type"] = "double";

  (*output)["attrs/min/value"] = bin_min;
  (*output)["attrs/min/type"] = "double";

  (*output)["attrs/max/value"] = bin_max;
  (*output)["attrs/max/type"] = "double";

  (*output)["attrs/center/value"] = bin_center;
  (*output)["attrs/center/type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
MaxFromPoint::MaxFromPoint() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
MaxFromPoint::~MaxFromPoint()
{
  // empty
}

//-----------------------------------------------------------------------------
void
MaxFromPoint::declare_interface(Node &i)
{
  i["type_name"] = "max_from_point";
  i["port_names"].append() = "binning";
  i["port_names"].append() = "axis";
  i["port_names"].append() = "point";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
MaxFromPoint::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
MaxFromPoint::execute()
{
  conduit::Node &in_binning = *input<Node>("binning");
  conduit::Node &in_axis =  *input<Node>("axis");
  conduit::Node &in_point =  *input<Node>("point");
  conduit::Node *output = new conduit::Node();

  const int num_axes = in_binning["attrs/bin_axes"].number_of_children();
  if(num_axes > 1)
  {
    ASCENT_ERROR("max_from_point: only one axis is implemented");
  }

  const double point = in_point["value"].to_float64();

  const conduit::Node &axis = in_binning["attrs/bin_axes/value"].child(0);
  const int num_bins = axis["num_bins"].to_int32();
  const double min_val = axis["min_val"].to_float64();
  const double max_val = axis["max_val"].to_float64();
  const double bin_size = (max_val - min_val) / double(num_bins);

  double *bins = in_binning["attrs/value/value"].value();
  double max_bin_val = std::numeric_limits<double>::lowest();
  double dist_value = 0;
  double min_dist = std::numeric_limits<double>::max();
  int index = -1;
  for(int i = 0; i < num_bins; ++i)
  {
    double val = bins[i];
    if(val >= max_bin_val)
    {
      double left = min_val + double(i) * bin_size;
      double right = min_val + double(i+1) * bin_size;
      double center = left + (right-left) / 2.0;
      double dist = fabs(center - point);
      if(val > max_bin_val ||
         ((dist < min_dist) && val == max_bin_val))
      {
        min_dist = dist;
        max_bin_val = val;
        dist_value = center - point;
        index = i;
      }
    }
  }

  double loc[3] = {0.0, 0.0, 0.0};
  std::string axis_str = in_axis["value"].as_string();

  if(axis_str == "z")
  {
    loc[2] = dist_value;
  }
  else if (axis_str == "y")
  {
    loc[1] = dist_value;
  }
  else
  {
    loc[0] = dist_value;
  }

  (*output)["type"] = "value_position";
  (*output)["attrs/value/value"] = max_bin_val;
  (*output)["attrs/value/type"] = "double";
  (*output)["attrs/position/value"].set(loc,3);
  (*output)["attrs/position/type"] = "vector";

  (*output)["value"] = min_dist;
  (*output)["type"] = "double";

  set_output<conduit::Node>(output);
}

//-----------------------------------------------------------------------------
Bin::Bin() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
Bin::~Bin()
{
  // empty
}

//-----------------------------------------------------------------------------
void
Bin::declare_interface(Node &i)
{
  i["type_name"] = "bin";
  i["port_names"].append() = "binning";
  i["port_names"].append() = "index";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Bin::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Bin::execute()
{
  conduit::Node &in_binning = *input<Node>("binning");
  conduit::Node &in_index =  *input<Node>("index");
  conduit::Node *output = new conduit::Node();

  const int num_axes = in_binning["attrs/bin_axes"].number_of_children();
  if(num_axes > 1)
  {
    ASCENT_ERROR("bin: only one axis is implemented");
  }

  int bindex = in_index["value"].to_int32();

  const conduit::Node &axis = in_binning["attrs/bin_axes/value"].child(0);
  const int num_bins = axis["num_bins"].to_int32();

  if(bindex < 0 || bindex >= num_bins)
  {
    ASCENT_ERROR("bin: invalid bin "<<bindex<<"."
                <<" Number of bins "<<num_bins);
  }

  const double min_val = axis["min_val"].to_float64();
  const double max_val = axis["max_val"].to_float64();
  const double bin_size = (max_val - min_val) / double(num_bins);
  double *bins = in_binning["attrs/value/value"].value();

  double left = min_val + double(bindex) * bin_size;
  double right = min_val + double(bindex+1) * bin_size;
  double center = left + (right-left) / 2.0;
  double val = bins[bindex];

  (*output)["type"] = "bin";

  (*output)["attrs/value/value"] = val;
  (*output)["attrs/value/type"] = "double";

  (*output)["attrs/min/value"] = left;
  (*output)["attrs/min/type"] = "double";

  (*output)["attrs/max/value"] = right;
  (*output)["attrs/max/type"] = "double";

  (*output)["attrs/center/value"] = center;
  (*output)["attrs/center/type"] = "double";

  set_output<conduit::Node>(output);
}
//-----------------------------------------------------------------------------
Bounds::Bounds() : Filter()
{
  // empty
}

//-----------------------------------------------------------------------------
Bounds::~Bounds()
{
  // empty
}

//-----------------------------------------------------------------------------
void
Bounds::declare_interface(Node &i)
{
  i["type_name"] = "bounds";
  i["port_names"].append() = "topology";
  i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
Bounds::verify_params(const conduit::Node &params, conduit::Node &info)
{
  info.reset();
  bool res = true;
  return res;
}

//-----------------------------------------------------------------------------
void
Bounds::execute()
{
  conduit::Node &n_topology = *input<Node>("topology");
  conduit::Node *output = new conduit::Node();

  const conduit::Node *const dataset =
      graph().workspace().registry().fetch<Node>("dataset");

  std::set<std::string> topos;

  if(!n_topology.dtype().is_empty())
  {
    std::string topo = n_topology["value"].as_string();
    if(!has_topology(*dataset, topo))
    {
      std::set<std::string> names = topology_names(*dataset);
      std::stringstream msg;
      msg<<"Unknown topology: '"<<topo<<"'. Known topologies: [";
      for(auto &name : names)
      {
        msg<<" "<<name;
      }
      msg<<" ]";
      ASCENT_ERROR(msg.str());
    }
    topos.insert(topo);
  }
  else
  {
    topos = topology_names(*dataset);
  }

  double inf = std::numeric_limits<double>::infinity();
  double min_vec[3] = {inf, inf, inf};
  double max_vec[3] = {-inf, -inf, -inf};
  for(auto &topo_name : topos)
  {
    conduit::Node n_aabb = global_bounds(*dataset, topo_name);
    double *t_min = n_aabb["min_coords"].as_float64_ptr();
    double *t_max = n_aabb["max_coords"].as_float64_ptr();
    for(int i = 0; i < 3; ++i)
    {
      min_vec[i] = std::min(t_min[i],min_vec[i]);
      max_vec[i] = std::max(t_max[i],max_vec[i]);
    }
  }

  (*output)["type"] = "aabb";
  (*output)["attrs/min/value"].set(min_vec, 3);
  (*output)["attrs/min/type"] = "vector";
  (*output)["attrs/max/value"].set(max_vec, 3);
  (*output)["attrs/max/type"] = "vector";

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
