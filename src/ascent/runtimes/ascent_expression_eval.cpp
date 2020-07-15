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
/// file: ascent_expression_eval.hpp
///
//-----------------------------------------------------------------------------

#include "ascent_expression_eval.hpp"
#include "expressions/ascent_blueprint_architect.hpp"
#include "expressions/ascent_derived_jit.hpp"
#include "expressions/ascent_expression_filters.hpp"
#include "expressions/ascent_expressions_ast.hpp"
#include "expressions/ascent_expressions_parser.hpp"
#include "expressions/ascent_expressions_tokens.hpp"

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
// -- begin ascent::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

conduit::Node ExpressionEval::m_cache;
conduit::Node g_function_table;
conduit::Node g_object_table;

void
register_builtin()
{
  flow::Workspace::register_filter_type<expressions::NullArg>();
  flow::Workspace::register_filter_type<expressions::Boolean>();
  flow::Workspace::register_filter_type<expressions::Double>();
  flow::Workspace::register_filter_type<expressions::Integer>();
  flow::Workspace::register_filter_type<expressions::Identifier>();
  flow::Workspace::register_filter_type<expressions::History>();
  flow::Workspace::register_filter_type<expressions::BinaryOp>();
  flow::Workspace::register_filter_type<expressions::String>();
  flow::Workspace::register_filter_type<expressions::ExpressionList>();
  flow::Workspace::register_filter_type<expressions::IfExpr>();
  flow::Workspace::register_filter_type<expressions::ScalarMax>();
  flow::Workspace::register_filter_type<expressions::ScalarMin>();
  flow::Workspace::register_filter_type<expressions::FieldMax>();
  flow::Workspace::register_filter_type<expressions::FieldMin>();
  flow::Workspace::register_filter_type<expressions::FieldAvg>();
  flow::Workspace::register_filter_type<expressions::FieldNanCount>();
  flow::Workspace::register_filter_type<expressions::FieldInfCount>();
  flow::Workspace::register_filter_type<expressions::FieldSum>();
  flow::Workspace::register_filter_type<expressions::ArrayMax>();
  flow::Workspace::register_filter_type<expressions::ArrayMin>();
  flow::Workspace::register_filter_type<expressions::ArrayAvg>();
  flow::Workspace::register_filter_type<expressions::ArraySum>();
  flow::Workspace::register_filter_type<expressions::Vector>();
  flow::Workspace::register_filter_type<expressions::Magnitude>();
  flow::Workspace::register_filter_type<expressions::Field>();
  flow::Workspace::register_filter_type<expressions::Axis>();
  flow::Workspace::register_filter_type<expressions::Histogram>();
  flow::Workspace::register_filter_type<expressions::Binning>();
  flow::Workspace::register_filter_type<expressions::Entropy>();
  flow::Workspace::register_filter_type<expressions::Pdf>();
  flow::Workspace::register_filter_type<expressions::Cdf>();
  flow::Workspace::register_filter_type<expressions::Quantile>();
  flow::Workspace::register_filter_type<expressions::BinByValue>();
  flow::Workspace::register_filter_type<expressions::BinByIndex>();
  flow::Workspace::register_filter_type<expressions::Cycle>();
  flow::Workspace::register_filter_type<expressions::ArrayAccess>();
  flow::Workspace::register_filter_type<expressions::DotAccess>();
  flow::Workspace::register_filter_type<expressions::JitFilter>();

  initialize_functions();
  initialize_objects();
}

ExpressionEval::ExpressionEval(conduit::Node *data) : m_data(data)
{
}

void
count_params()
{
  const int num_functions = g_function_table.number_of_children();
  for(int i = 0; i < num_functions; ++i)
  {
    conduit::Node &function = g_function_table.child(i);
    const int num_overloads = function.number_of_children();
    for(int o = 0; o < num_overloads; ++o)
    {
      conduit::Node &sig = function.child(o);
      const int num_args = sig["args"].number_of_children();
      int req = 0;
      int opt = 0;
      bool seen_opt = false;
      for(int a = 0; a < num_args; ++a)
      {
        const conduit::Node &arg = sig["args"].child(a);
        if(arg.has_path("optional"))
        {
          seen_opt = true;
          opt++;
        }
        else
        {
          req++;
          if(seen_opt)
          {
            function.print();
            ASCENT_ERROR("Function: optional parameters must come after "
                         "required params");
          }
        }
      }
      sig["req_count"] = req;
      sig["opt_count"] = opt;
    }
  }
}

void
initialize_functions()
{
  // functions
  g_function_table.reset();
  conduit::Node *functions = &g_function_table;

  // -------------------------------------------------------------

  conduit::Node &array_avg_sig = (*functions)["avg"].append();
  array_avg_sig["return_type"] = "double";
  array_avg_sig["filter_name"] = "array_avg"; // matches the filter's type_name
  array_avg_sig["args/arg1/type"] = "array"; // arg names match input port names
  array_avg_sig["description"] = "Return the average of an array.";

  // -------------------------------------------------------------

  conduit::Node &field_avg_sig = (*functions)["avg"].append();
  field_avg_sig["return_type"] = "double";
  field_avg_sig["filter_name"] = "field_avg";
  field_avg_sig["derived_support"] =
      "true"; // function is supported in derived jit
  field_avg_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_avg_sig["description"] = "Return the field average of a mesh variable.";

  // -------------------------------------------------------------

  conduit::Node &field_nan_sig = (*functions)["field_nan_count"].append();
  field_nan_sig["return_type"] = "double";
  field_nan_sig["filter_name"] = "field_nan_count";
  field_nan_sig["derived_support"] = "true";
  field_nan_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_nan_sig["description"] =
      "Return the number  of NaNs in a mesh variable.";

  // -------------------------------------------------------------

  conduit::Node &field_inf_sig = (*functions)["field_inf_count"].append();
  field_inf_sig["return_type"] = "double";
  field_inf_sig["filter_name"] = "field_inf_count";
  field_inf_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_inf_sig["description"] =
      "Return the number  of -inf and +inf in a mesh variable.";

  // -------------------------------------------------------------

  conduit::Node &scalar_max_sig = (*functions)["max"].append();
  scalar_max_sig["return_type"] = "double";
  scalar_max_sig["filter_name"] = "scalar_max";
  scalar_max_sig["args/arg1/type"] = "scalar";
  scalar_max_sig["args/arg2/type"] = "scalar";
  scalar_max_sig["description"] = "Return the maximum of two scalars.";

  // -------------------------------------------------------------

  conduit::Node &field_max_sig = (*functions)["max"].append();
  field_max_sig["return_type"] = "value_position";
  field_max_sig["filter_name"] = "field_max";
  field_max_sig["args/arg1/type"] = "field";
  field_max_sig["description"] =
      "Return the maximum value from the meshvar. Its position is also stored "
      "and is accessible via the `position` function.";

  // -------------------------------------------------------------

  conduit::Node &array_max_sig = (*functions)["max"].append();
  array_max_sig["return_type"] = "double";
  array_max_sig["filter_name"] = "array_max";
  array_max_sig["args/arg1/type"] = "array";
  array_max_sig["description"] = "Return the maximum of an array.";

  // -------------------------------------------------------------

  conduit::Node &field_scalar_max_sig = (*functions)["max"].append();
  field_scalar_max_sig["return_type"] = "derived_field";
  field_scalar_max_sig["filter_name"] = "field_field_max";
  field_scalar_max_sig["args/arg1/type"] = "field";
  field_scalar_max_sig["args/arg2/type"] = "scalar";
  field_scalar_max_sig["jitable"];
  field_scalar_max_sig["description"] =
      "Return a derived field that is the max of a field and a scalar.";

  // -------------------------------------------------------------

  // same as above but scalar goes first, field goes second
  conduit::Node &scalar_field_max_sig = (*functions)["max"].append();
  scalar_field_max_sig["return_type"] = "derived_field";
  scalar_field_max_sig["filter_name"] = "field_field_max";
  scalar_field_max_sig["args/arg1/type"] = "scalar";
  scalar_field_max_sig["args/arg2/type"] = "field";
  scalar_field_max_sig["jitable"];
  scalar_field_max_sig["description"] =
      "Return a derived field that is the max of a scalar and a field. Same "
      "functionality as above but the order of the arguments is switched.";

  // -------------------------------------------------------------

  // same as above but scalar goes first, field goes second
  conduit::Node &field_field_max_sig = (*functions)["max"].append();
  field_field_max_sig["return_type"] = "derived_field";
  field_field_max_sig["filter_name"] = "field_field_max";
  field_field_max_sig["args/arg1/type"] = "field";
  field_field_max_sig["args/arg2/type"] = "field";
  field_field_max_sig["jitable"];
  field_field_max_sig["description"] =
      "Return a derived field that is the max of two fields.";

  // -------------------------------------------------------------

  conduit::Node &field_min_sig = (*functions)["min"].append();
  field_min_sig["return_type"] = "value_position";
  field_min_sig["filter_name"] = "field_min";
  field_min_sig["args/arg1/type"] = "field";
  field_min_sig["description"] =
      "Return the minimum value from the meshvar. Its position is also stored "
      "and is accessible via the `position` function.";

  // -------------------------------------------------------------

  conduit::Node &scalar_min_sig = (*functions)["min"].append();
  scalar_min_sig["return_type"] = "double";
  scalar_min_sig["filter_name"] = "scalar_min";
  scalar_min_sig["args/arg1/type"] = "scalar";
  scalar_min_sig["args/arg2/type"] = "scalar";
  scalar_min_sig["description"] = "Return the minimum of two scalars.";

  // -------------------------------------------------------------

  conduit::Node &array_min_sig = (*functions)["min"].append();
  array_min_sig["return_type"] = "double";
  array_min_sig["filter_name"] = "array_min";
  array_min_sig["args/arg1/type"] = "array";
  array_min_sig["description"] = "Return the minimum of an array.";

  // -------------------------------------------------------------

  conduit::Node &field_sum_sig = (*functions)["sum"].append();
  field_sum_sig["return_type"] = "double";
  field_sum_sig["filter_name"] = "field_sum";
  field_sum_sig["derived_support"] = "true";
  field_sum_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_sum_sig["description"] = "Return the sum of a field.";

  // -------------------------------------------------------------

  conduit::Node &array_sum_sig = (*functions)["sum"].append();
  array_sum_sig["return_type"] = "double";
  array_sum_sig["filter_name"] = "array_sum";
  array_sum_sig["args/arg1/type"] = "array";
  array_sum_sig["description"] = "Return the sum of an array.";

  // -------------------------------------------------------------

  conduit::Node &cycle_sig = (*functions)["cycle"].append();
  cycle_sig["return_type"] = "int";
  cycle_sig["filter_name"] = "cycle";
  cycle_sig["args"] = conduit::DataType::empty();
  cycle_sig["description"] = "Return the current simulation cycle.";

  // -------------------------------------------------------------

  conduit::Node &vector = (*functions)["vector"].append();
  vector["return_type"] = "vector";
  vector["filter_name"] = "vector";
  vector["args/arg1/type"] = "scalar";
  vector["args/arg2/type"] = "scalar";
  vector["args/arg3/type"] = "scalar";
  vector["description"] = "Return the 3D position vector for the input value.";

  // -------------------------------------------------------------

  conduit::Node &mag_sig = (*functions)["magnitude"].append();
  mag_sig["return_type"] = "double";
  mag_sig["filter_name"] = "magnitude";
  mag_sig["args/arg1/type"] = "vector";
  mag_sig["description"] = "Return the magnitude of the input vector.";

  // -------------------------------------------------------------

  conduit::Node &hist_sig = (*functions)["histogram"].append();
  hist_sig["return_type"] = "histogram";
  hist_sig["filter_name"] = "histogram";
  hist_sig["args/arg1/type"] = "field";
  // In a flow filter, these become parameters
  hist_sig["args/num_bins/type"] = "int";
  hist_sig["args/num_bins/optional"];
  hist_sig["args/num_bins/description"] = "defaults to ``256``";

  hist_sig["args/min_val/type"] = "scalar";
  hist_sig["args/min_val/optional"];
  hist_sig["args/min_val/description"] = "defaults to ``min(arg1)``";

  hist_sig["args/max_val/type"] = "scalar";
  hist_sig["args/max_val/optional"];
  hist_sig["args/max_val/description"] = "defaults to ``max(arg1)``";

  hist_sig["description"] = "Return a histogram of the mesh variable. Return a "
                            "histogram of the mesh variable.";

  // -------------------------------------------------------------

  conduit::Node &history_sig = (*functions)["history"].append();
  history_sig["return_type"] = "anytype";
  history_sig["filter_name"] = "history";

  history_sig["args/expr_name/type"] = "anytype";
  history_sig["args/expr_name/description"] =
      "`expr_name` should be the name of an expression that was evaluated in "
      "the past.";

  history_sig["args/relative_index/type"] = "int";
  history_sig["args/relative_index/optional"];
  history_sig["args/relative_index/description"] = "The number of evaluations \
  ago. This should be less than the number of past evaluations. For example, \
  ``history(pressure, relative_index=1)`` returns the value of pressure one \
  evaluation ago.";

  history_sig["args/absolute_index/type"] = "int";
  history_sig["args/absolute_index/optional"];
  history_sig["args/absolute_index/description"] =
      "The index in the evaluation \
  history. This should be less than the number of past evaluations. For \
  example, ``history(pressure, absolute_index=0)`` returns the value of \
  pressure from the first time it was evaluated.";

  history_sig["description"] = "As the simulation progresses the expressions \
  are evaluated repeatedly. The history function allows you to get the value of \
  previous evaluations. For example, if we want to evaluate the difference \
  between the original state of the simulation and the current state then we \
  can use an absolute index of 0 to compare the initial value with the \
  current value: ``val - history(val, absolute_index=0)``. Another example is if \
  you want to evaluate the relative change between the previous state and the \
  current state: ``val - history(val, relative_index=1)``.\n\n \
  .. note:: Exactly one of ``relative_index`` or ``absolute_index`` must be \
  passed. If the argument name is not specified ``relative_index`` will be \
  used.";

  // -------------------------------------------------------------

  conduit::Node &entropy_sig = (*functions)["entropy"].append();
  entropy_sig["return_type"] = "double";
  entropy_sig["filter_name"] = "entropy";
  entropy_sig["args/hist/type"] = "histogram";
  entropy_sig["description"] =
      "Return the Shannon entropy given a histogram of the field.";

  // -------------------------------------------------------------

  conduit::Node &pdf_sig = (*functions)["pdf"].append();
  pdf_sig["return_type"] = "histogram";
  pdf_sig["filter_name"] = "pdf";
  pdf_sig["args/hist/type"] = "histogram";
  pdf_sig["description"] =
      "Return the probability distribution function (pdf) from a histogram.";

  // -------------------------------------------------------------

  conduit::Node &cdf_sig = (*functions)["cdf"].append();
  cdf_sig["return_type"] = "histogram";
  cdf_sig["filter_name"] = "cdf";
  cdf_sig["args/hist/type"] = "histogram";
  cdf_sig["description"] =
      "Return the cumulative distribution function (cdf) from a histogram.";

  // -------------------------------------------------------------

  // gets histogram bin by index
  conduit::Node &bin_by_index_sig = (*functions)["bin"].append();
  bin_by_index_sig["return_type"] = "double";
  bin_by_index_sig["filter_name"] = "bin_by_index";
  bin_by_index_sig["args/hist/type"] = "histogram";
  bin_by_index_sig["args/bin/type"] = "int";
  bin_by_index_sig["description"] =
      "Return the value of the bin at index `bin` of a histogram.";

  // -------------------------------------------------------------

  // gets histogram bin by value
  conduit::Node &bin_by_value_sig = (*functions)["bin"].append();
  bin_by_value_sig["return_type"] = "double";
  bin_by_value_sig["filter_name"] = "bin_by_value";
  bin_by_value_sig["args/hist/type"] = "histogram";
  bin_by_value_sig["args/val/type"] = "scalar";
  bin_by_value_sig["description"] =
      "Return the value of the bin with axis-value `val` on the histogram.";

  // -------------------------------------------------------------

  conduit::Node &field_sig = (*functions)["field"].append();
  field_sig["return_type"] = "field";
  field_sig["filter_name"] = "field";
  field_sig["args/arg1/type"] = "string";
  field_sig["description"] = "Return a mesh field given a its name.";

  // -------------------------------------------------------------

  conduit::Node &quantile_sig = (*functions)["quantile"].append();
  quantile_sig["return_type"] = "double";
  quantile_sig["filter_name"] = "quantile";
  quantile_sig["args/cdf/type"] = "histogram";
  quantile_sig["args/cdf/description"] = "CDF of a histogram.";

  quantile_sig["args/q/type"] = "double";
  quantile_sig["args/q/description"] = "Quantile between 0 and 1 inclusive.";

  quantile_sig["args/interpolation/type"] = "string";
  quantile_sig["args/interpolation/optional"];
  quantile_sig["args/interpolation/description"] =
      "Specifies the interpolation \
  method to use when the quantile lies between two data points ``i < j``: \n\n \
  - linear (default): ``i + (j - i) * fraction``, where fraction is the \
  fractional part of the index surrounded by ``i`` and ``j``. \n \
  - lower: ``i``. \n \
  - higher: ``j``. \n \
  - nearest: ``i`` or ``j``, whichever is nearest. \n \
  - midpoint: ``(i + j) / 2``";

  quantile_sig["description"] = "Return the `q`-th quantile of the data along \
  the axis of `cdf`. For example, if `q` is 0.5 the result is the value on the \
  x-axis which 50\% of the data lies below.";

  // -------------------------------------------------------------

  conduit::Node &axis_sig = (*functions)["axis"].append();
  axis_sig["return_type"] = "axis";
  axis_sig["filter_name"] = "axis";
  axis_sig["args/name/type"] = "string";
  axis_sig["args/name/description"] = "The name of a scalar field on the mesh "
                                      "or one of ``'x'``, ``'y'``, or ``'z'``.";
  // rectilinear binning
  axis_sig["args/bins/type"] = "list";
  axis_sig["args/bins/optional"];
  axis_sig["args/bins/description"] =
      "A strictly increasing list of scalars containing the values for each "
      "tick. Used to specify a rectilinear axis.";
  // uniform binning
  axis_sig["args/min_val/type"] = "scalar";
  axis_sig["args/min_val/optional"];
  axis_sig["args/min_val/description"] =
      "Minimum value of the axis (i.e. the value of the first tick).";
  axis_sig["args/max_val/type"] = "scalar";
  axis_sig["args/max_val/optional"];
  axis_sig["args/max_val/description"] =
      "Maximum value of the axis (i.e. the value of the last tick).";
  axis_sig["args/num_bins/type"] = "int";
  axis_sig["args/num_bins/optional"];
  axis_sig["args/num_bins/description"] =
      "Number of bins on the axis (i.e. the number of ticks minus 1).";
  axis_sig["description"] =
      "Defines a uniform or rectilinear axis. When used for binning the bins "
      "are inclusive on the lower boundary and exclusive on the higher "
      "boundary of each bin. Either specify only ``bins`` or a subset of the "
      "``min_val``, ``max_val``, ``num_bins`` options.";
  axis_sig["args/clamp/type"] = "bool";
  axis_sig["args/clamp/optional"];
  axis_sig["args/clamp/description"] =
      "Defaults to ``False``. If ``True``, values outside the axis should be "
      "put into the bins on the boundaries.";
  // -------------------------------------------------------------

  conduit::Node &binning_sig = (*functions)["binning"].append();
  binning_sig["return_type"] = "binning";
  binning_sig["filter_name"] = "binning";
  binning_sig["args/reduction_var/type"] = "string";
  binning_sig["args/reduction_var/description"] =
      "The variable being reduced. Either the name of a scalar field on the "
      "mesh or one of ``'x'``, ``'y'``, or ``'z'``.";
  binning_sig["args/reduction_op/type"] = "string";
  binning_sig["args/reduction_op/description"] =
      "The reduction operator to use when \
  putting values in bins. Available reductions are: \n\n \
  - cnt: number of elements in a bin \n \
  - min: minimum value in a bin \n \
  - max: maximum value in a bin \n \
  - sum: sum of values in a bin \n \
  - avg: average of values in a bin \n \
  - pdf: probability distribution function over all bins \n \
  - std: standard deviation of values in a bin \n \
  - var: variance of values in a bin \n \
  - rms: root mean square of values in a bin";
  binning_sig["args/bin_axes/type"] = "list";
  binning_sig["args/bin_axes/description"] =
      "List of Axis objects which define the bin axes.";
  binning_sig["args/empty_bin_val/type"] = "scalar";
  binning_sig["args/empty_bin_val/optional"];
  binning_sig["args/empty_bin_val/description"] =
      "The value that empty bins should have. Defaults to 0.";
  binning_sig["args/output/type"] = "string";
  binning_sig["args/output/optional"];
  binning_sig["args/output/description"] =
      "Defaults to ``'none'``. If set to ``'bins'`` a binning with 3 or fewer "
      "dimensions will be output as a new topology on the dataset. This is "
      "useful for directly visualizing the binning. If set to ``'mesh'`` the "
      "bins will be \"painted\" back onto the original mesh as a new field.";
  binning_sig["description"] = "Returns a multidimensional data binning.";

  // -------------------------------------------------------------

  count_params();
  // functions->save("functions.json", "json");
  // TODO: validate that there are no ambiguities
}

void
initialize_objects()
{
  // object type definitions
  g_object_table.reset();
  conduit::Node *objects = &g_object_table;

  conduit::Node &histogram = (*objects)["histogram/attrs"];
  histogram["value/type"] = "array";
  histogram["min_val/type"] = "double";
  histogram["max_val/type"] = "double";
  histogram["num_bins/type"] = "int";

  conduit::Node &value_position = (*objects)["value_position/attrs"];
  value_position["value/type"] = "double";
  value_position["position/type"] = "vector";

  // objects->save("objects.json", "json");
}

conduit::Node
ExpressionEval::evaluate(const std::string expr, std::string expr_name)
{

  if(expr_name == "")
  {
    expr_name = expr;
  }

  // used to eliminate common subexpressions
  // ex: (x - min) / (max - min) then min should only be evaluated once
  conduit::Node subexpr_cache;
  w.registry().add<conduit::Node>("subexpr_cache", &subexpr_cache, -1);
  w.registry().add<conduit::Node>("dataset", m_data, -1);
  w.registry().add<conduit::Node>("cache", &m_cache, -1);
  w.registry().add<conduit::Node>("function_table", &g_function_table, -1);
  w.registry().add<conduit::Node>("object_table", &g_object_table, -1);
  int cycle = get_state_var(*m_data, "cycle").to_int32();
  w.registry().add<int>("cycle", &cycle, -1);

  try
  {
    scan_string(expr.c_str());
  }
  catch(const char *msg)
  {
    w.reset();
    ASCENT_ERROR("Expression parsing error: " << msg << " in '" << expr << "'");
  }

  ASTExpression *expression = get_result();

  conduit::Node root;

  try
  {
    root = expression->build_graph(w);
    // if root is a derived field add a JitFilter to execute it
    if(root["type"].as_string() == "derived_field")
    {
      conduit::Node params;
      params["func"] = "execute";
      params["execute"] = true;
      params["inputs/derived_field/type"] = "derived_field";
      params["inputs/derived_field/port"] = 0;
      w.graph().add_filter("jit_filter", "jit_execute", params);
      // src, dest, port
      w.graph().connect(root["filter_name"].as_string(), "jit_execute", 0);
      detail::null_ports(w, "jit_execute", 1, 256);
      root["filter_name"] = "jit_execute";
      root["type"] = "field";
    }
    w.execute();
  }
  catch(std::exception &e)
  {
    delete expression;
    w.reset();
    ASCENT_ERROR("Error while executing expression '" << expr
                                                      << "': " << e.what());
  }
  std::string filter_name = root["filter_name"].as_string();

  conduit::Node *n_res = w.registry().fetch<conduit::Node>(filter_name);
  conduit::Node return_val = *n_res;

  std::stringstream cache_entry;
  cache_entry << expr_name << "/" << cycle;

  // this causes an invalid read in conduit in the expression tests
  // m_cache[cache_entry.str()] = *n_res;
  m_cache[cache_entry.str()] = return_val;

  delete expression;
  w.reset();
  return return_val;
}

void
ExpressionEval::evaluate_derived(const std::string expr, std::string expr_name)
{

  if(expr_name == "")
  {
    expr_name = expr;
  }

  conduit::Node subexpr_cache;
  w.registry().add<conduit::Node>("subexpr_cache", &subexpr_cache, -1);
  w.registry().add<conduit::Node>("dataset", m_data, -1);
  w.registry().add<conduit::Node>("cache", &m_cache, -1);
  w.registry().add<conduit::Node>("function_table", &g_function_table, -1);
  w.registry().add<conduit::Node>("object_table", &g_object_table, -1);
  int cycle = get_state_var(*m_data, "cycle").to_int32();
  w.registry().add<int>("cycle", &cycle, -1);

  try
  {
    scan_string(expr.c_str());
  }
  catch(const char *msg)
  {
    w.reset();
    ASCENT_ERROR("JIT Expression parsing error: " << msg << " in '" << expr
                                                  << "'");
  }

  ASTExpression *expression = get_result();

  conduit::Node root;

  try
  {
    // expression->access();
    // root = expression->build_graph(w);
    conduit::Node info;
    bool can = expression->can_jit();
    std::cout << "Expression = " << expr << "\n";
    std::cout << "CAN string " << can << "\n";
    std::string res = expression->build_jit(info, w);
    if(info.has_path("pre-execute"))
    {
      // std::cout<<w.graph().to_dot()<<"\n";
      // w.graph().save_dot_html("ascent_jit_pre_execute_graph.html");
      w.execute();
      int results = info["pre-execute"].number_of_children();
      for(int i = 0; i < results; ++i)
      {
        const std::string name =
            info["pre-execute"].child(i)["filter_name"].as_string();
        conduit::Node *n_res = w.registry().fetch<conduit::Node>(name);
        std::cout << "***************\n";
        n_res->print();
        info["constants/" + name + "/value"] = (*n_res)["attrs/value/value"];
      }
    }
    expression->access();
    std::cout << "Res: " << res << "\n";
    ;
    do_it(*m_data, res, info);
  }
  catch(std::exception &e)
  {
    delete expression;
    w.reset();
    ASCENT_ERROR("Error while executing expression '" << expr
                                                      << "': " << e.what());
  }
  // std::string filter_name = root["filter_name"].as_string();

  // conduit::Node *n_res = w.registry().fetch<conduit::Node>(filter_name);
  // conduit::Node return_val = *n_res;

  // std::stringstream cache_entry;
  // cache_entry<<expr_name<<"/"<<cycle;

  //// this causes an invalid read in conduit in the expression tests
  ////m_cache[cache_entry.str()] = *n_res;
  // m_cache[cache_entry.str()] = return_val;

  // delete expression;
  // w.reset();
  // return return_val;
}

//-----------------------------------------------------------------------------
const conduit::Node &
ExpressionEval::get_cache()
{
  return m_cache;
}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::expressions--
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
