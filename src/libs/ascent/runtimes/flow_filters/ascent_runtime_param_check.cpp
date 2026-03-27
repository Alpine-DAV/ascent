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

//-----------------------------------------------------------------------------

conduit::Node string_schema(optional_param<int> minLength,
                            optional_param<int> maxLength)
{
  conduit::Node n;
  n["type"] = "string";
  if(minLength) n["minLength"] = *minLength;
  if(maxLength) n["maxLength"] = *maxLength;
  return n;
}

conduit::Node string_enum_schema(std::vector<std::string> options)
{
  conduit::Node n = string_schema();
  for (const auto& value: options)
  {
    n["enum"].append() = value;
  }
  return n;
}

conduit::Node bool_schema()
{
    return string_enum_schema({"true", "false"});
}

//-----------------------------------------------------------------------------

conduit::Node expression_schema()
{
  conduit::Node n = string_schema();
  n["format"] = "expression";
  return n;
}

//-----------------------------------------------------------------------------

conduit::Node number_schema(bool supports_expressions,
                            optional_param<int> minimum,
                            optional_param<int> maximum,
                            optional_param<int> exclusiveMinimum,
                            optional_param<int> exclusiveMaximum)
{
  conduit::Node n;
  if (supports_expressions)
  {
    n["oneOf"].append().set(number_schema(false, minimum, maximum, exclusiveMinimum, exclusiveMaximum));
    n["oneOf"].append().set(expression_schema());
  }
  else
  {
    n["type"] = "number";

    if(exclusiveMinimum) n["exclusiveMinimum"] = *exclusiveMinimum;
    else if(minimum) n["minimum"] = *minimum;

    if(exclusiveMaximum) n["exclusiveMaximum"] = *exclusiveMaximum;
    else if(maximum) n["maximum"] = *maximum;
  }
  return n;
}

conduit::Node integer_schema(bool supports_expressions,
                             optional_param<int> minimum,
                             optional_param<int> maximum,
                             optional_param<int> exclusiveMinimum,
                             optional_param<int> exclusiveMaximum)
{
  conduit::Node n;
  if (supports_expressions)
  {
    n["oneOf"].append().set(integer_schema(false, minimum, maximum, exclusiveMinimum, exclusiveMaximum));
    n["oneOf"].append().set(expression_schema());
  }
  else
  {
    n["type"] = "integer";

    if(exclusiveMinimum) n["exclusiveMinimum"] = *exclusiveMinimum;
    else if(minimum) n["minimum"] = *minimum;

    if(exclusiveMaximum) n["exclusiveMaximum"] = *exclusiveMaximum;
    else if(maximum) n["maximum"] = *maximum;
  }
  return n;
}

//-----------------------------------------------------------------------------

conduit::Node vec3_schema(const std::string var1,
                          const std::string var2,
                          const std::string var3,
                          bool supports_expressions)
{
  conduit::Node n;
  n["type"] = "object";
  n["additionalProperties"] = false;

  n["properties/" + var1].set(number_schema(supports_expressions));
  n["properties/" + var2].set(number_schema(supports_expressions));
  n["properties/" + var3].set(number_schema(supports_expressions));

  n["required"].append() = var1;
  n["required"].append() = var2;
  n["required"].append() = var3;

  return n;
}

conduit::Node vec3_schema(bool supports_expressions)
{
  return vec3_schema("x", "y", "z", supports_expressions);
}

conduit::Node vec3_schema_anyOf(const std::string var1,
                                const std::string var2,
                                const std::string var3,
                                bool supports_expressions)
{
  conduit::Node n;
  n["type"] = "object";
  n["additionalProperties"] = false;

  n["properties/" + var1].set(number_schema(supports_expressions));
  n["properties/" + var2].set(number_schema(supports_expressions));
  n["properties/" + var3].set(number_schema(supports_expressions));

  conduit::Node var1_required;
  var1_required["type"] = "object";
  var1_required["required"] = var1;
  n["anyOf"].append().set(var1_required);

  conduit::Node var2_required;
  var2_required["type"] = "object";
  var2_required["required"] = var2;
  n["anyOf"].append().set(var2_required);

  conduit::Node var3_required;
  var3_required["type"] = "object";
  var3_required["required"] = var3;
  n["anyOf"].append().set(var3_required);

  return n;
}

conduit::Node vec3_schema_anyOf(bool supports_expressions)
{
  return vec3_schema_anyOf("x", "y", "z", supports_expressions);
}

//-----------------------------------------------------------------------------

conduit::Node array_schema(const conduit::Node &item_schema)
{
  conduit::Node n;
  n["type"] = "array";
  n["items"].set(item_schema);
  return n;
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





