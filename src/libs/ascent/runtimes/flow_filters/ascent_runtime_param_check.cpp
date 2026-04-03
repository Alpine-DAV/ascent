//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_param_check.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_param_check.hpp"
#include "ascent_expression_eval.hpp"
#include "expressions/ascent_expressions_ast.hpp"
#include "expressions/ascent_expressions_tokens.hpp"
#include "expressions/ascent_expressions_parser.hpp"
#include <ascent_logging.hpp>
#include <flow_schema_validator.hpp>

#include <algorithm>

using namespace conduit;

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
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{
//-----------------------------------------------------------------------------
//Parse the ParamSpec struct
ParamSpec 
assign_param_spec(const conduit::Node &n, DataObject *data_object)
{
  ParamSpec spec;

  // If it's a string: "min" / "max"
  if(n.dtype().is_string())
  {
    std::string s = n.as_string();
    if(s == "min") spec.mode = ParamVal::BoundsMin;
    else if(s == "max") spec.mode = ParamVal::BoundsMax;
    else ASCENT_ERROR("reflect axis must be a number or 'min'/'max' (got '" << s << "')");
    return spec;
  }

  // Otherwise treat it as numeric / expression supported by get_float64
  spec.mode  = ParamVal::Value;
  spec.value = get_float64(n, data_object);
  return spec;
}

//-----------------------------------------------------------------------------
// this detects if the syntax is valid, not
// whether the expression will actually work
bool is_valid_expression(const std::string &expr, std::string &err_msg)
{
  bool res = true;
  try
  {
    scan_string(expr.c_str());
  }
  catch(const char *msg)
  {
    err_msg = msg;
    res = false;
  }
  return res;
}

void ascent_register_flow_schema_hooks()
{
    flow::schema::register_format_checker("expression", &is_valid_expression);
}

//-----------------------------------------------------------------------------

conduit::Node &string_schema(conduit::Node &schema_node,
                             size_t minLength,
                             size_t maxLength)
{
  schema_node.reset();
  schema_node["type"] = "string";
  if(minLength != 0) schema_node["minLength"] = minLength;
  if(maxLength != std::numeric_limits<std::size_t>::max()) schema_node["maxLength"] = maxLength;
  return schema_node;
}

//-----------------------------------------------------------------------------

conduit::Node &string_enum_schema(conduit::Node &schema_node, const std::vector<std::string> &options)
{
  string_schema(schema_node);
  for (const auto& value: options)
  {
    schema_node["enum"].append() = value;
  }
  return schema_node;
}

conduit::Node &bool_schema(conduit::Node &schema_node)
{
    return string_enum_schema(schema_node, {"true", "false"});
}

//-----------------------------------------------------------------------------

conduit::Node &expression_schema(conduit::Node &schema_node)
{
  string_schema(schema_node);
  schema_node["format"] = "expression";
  return schema_node;
}

//-----------------------------------------------------------------------------

conduit::Node &number_schema(conduit::Node &schema_node,
                             bool supports_expressions,
                             int minimum,
                             int maximum,
                             int exclusiveMinimum,
                             int exclusiveMaximum)
{
  schema_node.reset();
  if (supports_expressions)
  {
    number_schema(schema_node["oneOf"].append(), false, minimum, maximum, exclusiveMinimum, exclusiveMaximum);
    expression_schema(schema_node["oneOf"].append());
  }
  else
  {
    schema_node["type"] = "number";

    if(exclusiveMinimum != std::numeric_limits<int>::lowest()) schema_node["exclusiveMinimum"] = exclusiveMinimum;
    else if(minimum != std::numeric_limits<int>::lowest()) schema_node["minimum"] = minimum;

    if(exclusiveMaximum != std::numeric_limits<int>::max()) schema_node["exclusiveMaximum"] = exclusiveMaximum;
    else if(maximum != std::numeric_limits<int>::max()) schema_node["maximum"] = maximum;
  }
  return schema_node;
}

conduit::Node &integer_schema(conduit::Node &schema_node,
                             bool supports_expressions,
                             int minimum,
                             int maximum,
                             int exclusiveMinimum,
                             int exclusiveMaximum)
{
  if (supports_expressions)
  {
    integer_schema(schema_node["oneOf"].append(), false, minimum, maximum, exclusiveMinimum, exclusiveMaximum);
    expression_schema(schema_node["oneOf"].append());
  }
  else
  {
    schema_node["type"] = "integer";

    if(exclusiveMinimum != std::numeric_limits<int>::lowest()) schema_node["exclusiveMinimum"] = exclusiveMinimum;
    else if(minimum != std::numeric_limits<int>::lowest()) schema_node["minimum"] = minimum;

    if(exclusiveMaximum != std::numeric_limits<int>::max()) schema_node["exclusiveMaximum"] = exclusiveMaximum;
    else if(maximum != std::numeric_limits<int>::max()) schema_node["maximum"] = maximum;
  }
  return schema_node;
}

//-----------------------------------------------------------------------------

conduit::Node &vec3_schema(conduit::Node &schema_node,
                           const std::string var1,
                           const std::string var2,
                           const std::string var3,
                           bool supports_expressions)
{
  schema_node.reset();
  
  schema_node["type"] = "object";
  schema_node["additionalProperties"] = false;

  number_schema(schema_node["properties/" + var1], supports_expressions);
  number_schema(schema_node["properties/" + var2], supports_expressions);
  number_schema(schema_node["properties/" + var3], supports_expressions);

  schema_node["required"].append() = var1;
  schema_node["required"].append() = var2;
  schema_node["required"].append() = var3;

  return schema_node;
}

conduit::Node &vec3_schema(conduit::Node &schema_node, bool supports_expressions)
{
  return vec3_schema(schema_node, "x", "y", "z", supports_expressions);
}

conduit::Node &vec3_schema_anyOf(conduit::Node &schema_node,
                                 const std::string var1,
                                 const std::string var2,
                                 const std::string var3,
                                 bool supports_expressions)
{
  schema_node.reset();
  
  schema_node["type"] = "object";
  schema_node["additionalProperties"] = false;

  number_schema(schema_node["properties/" + var1], supports_expressions);
  number_schema(schema_node["properties/" + var2], supports_expressions);
  number_schema(schema_node["properties/" + var3], supports_expressions);

  conduit::Node &var1_required = schema_node["anyOf"].append();
  var1_required["type"] = "object";
  var1_required["required"].append() = var1;

  conduit::Node &var2_required = schema_node["anyOf"].append();
  var2_required["type"] = "object";
  var2_required["required"].append() = var2;

  conduit::Node &var3_required = schema_node["anyOf"].append();
  var3_required["type"] = "object";
  var3_required["required"].append() = var3;

  return schema_node;
}

conduit::Node &vec3_schema_anyOf(conduit::Node &schema_node, bool supports_expressions)
{
  return vec3_schema_anyOf(schema_node, "x", "y", "z", supports_expressions);
}

//-----------------------------------------------------------------------------

conduit::Node &array_schema(conduit::Node &schema_node, const conduit::Node &item_schema)
{
  schema_node.reset();
  
  schema_node["type"] = "array";
  if (!item_schema.dtype().is_empty())
  {
    schema_node["items"].set(item_schema);
  }

  return schema_node;
}

conduit::Node array_schema()
{
  conduit::Node n;
  n["type"] = "array";
  return n;
}

//-----------------------------------------------------------------------------

conduit::Node ignore_schema()
{
    conduit::Node n;
    n["type"] = "object";
    n["constraints/skip"] = true;
    return n;
}

//-----------------------------------------------------------------------------
bool
check_numeric(const std::string path,
              const conduit::Node &params,
              conduit::Node &info,
              bool required,
              bool supports_expressions)
{
  bool res = true;
  if(!params.has_path(path) && required)
  {
    info["errors"].append() = "Missing required numeric parameter '" + path + "'";
    res = false;
  }

  if(params.has_path(path))
  {

    bool is_expr = false;
    std::string err_msg;
    if(params[path].dtype().is_string() && supports_expressions)
    {
      // check to see if this is a valid expression

      is_expr = is_valid_expression(params[path].as_string(), err_msg);
    }

    if(!params[path].dtype().is_number() && !is_expr)
    {
      if(supports_expressions)
      {
        std::string msg = "Expected numeric parameter '" + path +
                          " : " + params[path].to_yaml()
                             + "'  is not numeric and is not a valid expression."
                             + " Error message '" + err_msg + "'";
        info["errors"].append() = msg;
      }
      else
      {
        std::string msg = "Expected numeric parameter '" + path +
                          " : " + params[path].to_yaml()
                             + "'  is not numeric and does not support expressions";
      }
      res = false;
    }
  }
  return res;
}

//-----------------------------------------------------------------------------
bool
check_string(const std::string path,
             const conduit::Node &params,
             conduit::Node &info,
             bool required)
{
  schema_node.reset();
  schema_node["type"] = "array";
  return schema_node;
}

//-----------------------------------------------------------------------------

conduit::Node &ignore_schema(conduit::Node &schema_node)
{
  schema_node.reset();
  
  schema_node["type"] = "object";
  schema_node["constraints/skip"] = true;

  return schema_node;
}

//-----------------------------------------------------------------------------
template<typename T>
T conduit_cast(const conduit::Node &node);

//-----------------------------------------------------------------------------
template<>
int conduit_cast<int>(const conduit::Node &node)
{
  return node.to_int32();
}

//-----------------------------------------------------------------------------
template<>
double conduit_cast<double>(const conduit::Node &node)
{
  return node.to_float64();
}

//-----------------------------------------------------------------------------
template<>
float conduit_cast<float>(const conduit::Node &node)
{
  return node.to_float32();
}

//-----------------------------------------------------------------------------
template<typename T>
T get_value(const conduit::Node &node, DataObject *dataset)
{
  T value = 0;
  if(node.dtype().is_empty())
  {
    // don't silently return a value from an empty node
    ASCENT_ERROR("Cannot get value from and empty node");
  }

  // check to see if this is an expression
  if(node.dtype().is_string())
  {
    if(dataset == nullptr)
    {
      ASCENT_ERROR("Numeric parameter is an expression(string)"
                   <<" but we can not evaluate the expression."
                   <<" This is usually for a parameter that is"
                   <<" not meant to have an expression. expression '"
                   <<node.to_string()<<"'");

    }
    // TODO: we want to zero copy this
    conduit::Node * bp_dset = dataset->as_low_order_bp().get();
    expressions::ExpressionEval eval(bp_dset);
    std::string expr = node.as_string();
    conduit::Node res = eval.evaluate(expr);

    if(!res.has_path("value"))
    {
      ASCENT_ERROR("expression '"<<expr
                   <<"': failed to extract a value from the result."
                   <<" '"<<res.to_yaml()<<"'");
    }

    if(res["value"].dtype().number_of_elements() != 1)
    {
      ASCENT_ERROR("expression '"<<expr
                   <<"' resulted in multiple values."
                   <<" Expected scalar. '"<<res.to_yaml()<<"'");
    }
    value = res["value"].to_float64();
  }
  else
  {
    value = conduit_cast<T>(node);
  }
  return value;
}

//-----------------------------------------------------------------------------
double get_float64(const conduit::Node &node, DataObject *dataset)
{
  return get_value<double>(node, dataset);
}

//-----------------------------------------------------------------------------
float get_float32(const conduit::Node &node, DataObject *dataset)
{
  return get_value<float>(node, dataset);
}

//-----------------------------------------------------------------------------
int get_int32(const conduit::Node &node, DataObject *dataset)
{
  return get_value<int>(node, dataset);
}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
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




