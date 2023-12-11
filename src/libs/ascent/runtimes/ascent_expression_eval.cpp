//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_expression_eval.hpp
///
//-----------------------------------------------------------------------------

#include <ascent_config.h>
#include "ascent_expression_eval.hpp"
#include "ascent_data_logger.hpp"
#include "expressions/ascent_blueprint_architect.hpp"
#include "expressions/ascent_expression_filters.hpp"
#include "expressions/ascent_expressions_ast.hpp"
#include "expressions/ascent_expressions_parser.hpp"
#include "expressions/ascent_expressions_tokens.hpp"

// JIT headers
#include "expressions/ascent_derived_jit.hpp"
#include "expressions/ascent_expression_jit_filters.hpp"

#ifdef ASCENT_JIT_ENABLED
// Needed for logging functions
#include "expressions/ascent_array_registry.hpp"
#endif

#include <ctime>
#include <flow_timer.hpp>
#include <stdio.h>
#include <stdlib.h>

#ifdef ASCENT_MPI_ENABLED
#include <conduit_relay_mpi.hpp>
#include <mpi.h>
#endif
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

conduit::Node g_function_table;
conduit::Node g_object_table;

Cache ExpressionEval::m_cache;

double
Cache::last_known_time()
{
  double res = 0;
  if(m_data.has_path("last_known_time"))
  {
    res = m_data["last_known_time"].as_float64();
  }
  return res;
}

bool
Cache::filtered()
{
  return m_filtered;
}

void
Cache::last_known_time(double time)
{
  m_data["last_known_time"] = time;
}

void
Cache::filter_time(double ftime)
{
  const int num_entries = m_data.number_of_children();
  int removal_count = 0;
  for(int i = 0; i < num_entries; ++i)
  {
    conduit::Node &entry = m_data.child(i);
    if(entry.name() == "last_known_time" ||
       entry.name() == "session_cache_info")
    {
      continue;
    }

    bool invalid_time = true;
    while(invalid_time && entry.number_of_children() > 0)
    {
      int last = entry.number_of_children() - 1;

      if(!entry.child(last).has_path("time"))
      {
        // if there is no time, we can reason about
        // anything
        entry.remove(last);
        removal_count++;
      }
      else if(entry.child(last)["time"].to_float64() >= ftime)
      {
        entry.remove(last);
        removal_count++;
      }
      else
      {
        invalid_time = false;
      }
    }
  }

  // clean up entries with no children
  bool clean = false;
  while(!clean)
  {
    const int size = m_data.number_of_children();
    bool removed = false;
    for(int i = 0; i < size; ++i)
    {
      if(m_data.child(i).number_of_children() == 0)
      {
        m_data.remove(i);
        removed = true;
        break;
      }
    }
    clean = !removed;
  }

  time_t t;
  char curr_time[100];
  time(&t);

  std::strftime(curr_time, sizeof(curr_time), "%A %c", std::localtime(&t));
  std::stringstream msg;
  msg << "Time travel detected at " << curr_time << '\n';
  msg << "Removed all expression cache entries (" << removal_count << ")"
      << " after simulation time " << ftime << ".";
  m_data["ascent_cache_info"].append() = msg.str();
  m_filtered = true;
}

bool
Cache::loaded()
{
  return m_loaded;
}

void
Cache::load(const std::string &dir, const std::string &session)
{
  m_rank = 0;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &m_rank);
#endif

  std::string file_name = session;
  std::string session_file = conduit::utils::join_path(dir, file_name);
  m_session_file = session_file;

  bool exists = conduit::utils::is_file(session_file);

  if(m_rank == 0 && exists)
  {
    m_data.load(session_file + ".yaml", "yaml");
  }

#ifdef ASCENT_MPI_ENABLED
  if(exists)
  {
    conduit::relay::mpi::broadcast_using_schema(m_data, 0, mpi_comm);
  }
#endif
  m_loaded = true;
}

void Cache::save()
{
  // the session file can be blank during testing,
  // since its not actually opening ascent
  if(m_rank == 0 && !m_data.dtype().is_empty() && m_session_file != "")
  {
    m_data.save(m_session_file+".yaml","yaml");
  }
}

void Cache::save(const std::string &filename)
{
  // the session file can be blank during testing,
  // since its not actually opening ascent
  if(m_rank == 0 &&
     !m_data.dtype().is_empty())
  {
    m_data.save(filename+".yaml","yaml");
  }
}

void Cache::save(const std::string &filename,
                 const std::vector<std::string> &selection)
{
  conduit::Node data;
  for(const auto &expr : selection)
  {
    if(m_data.has_path(expr))
    {
      data[expr].set_external(m_data[expr]);
    }
  }
  // the session file can be blank during testing,
  // since its not actually opening ascent
  // or there might not be match
  if(m_rank == 0 &&
     !data.dtype().is_empty())
  {
    data.save(filename+".yaml","yaml");
  }
}

Cache::~Cache()
{
  save();
}

void
register_builtin()
{
  // basic lang:
  // boolean, integer, double, string, nan, null, identifier, dot access,
  // if (conditional), binary operations (math, logical, etc)
  flow::Workspace::register_filter_type<expressions::ExprBoolean>();
  flow::Workspace::register_filter_type<expressions::ExprInteger>();
  flow::Workspace::register_filter_type<expressions::ExprDouble>();
  flow::Workspace::register_filter_type<expressions::ExprString>();
  flow::Workspace::register_filter_type<expressions::ExprNan>();
  flow::Workspace::register_filter_type<expressions::ExprNull>();
  flow::Workspace::register_filter_type<expressions::ExprIdentifier>();
  flow::Workspace::register_filter_type<expressions::ExprObjectDotAccess>();
  flow::Workspace::register_filter_type<expressions::ExprIf>();
  flow::Workspace::register_filter_type<expressions::ExprBinaryOp>();

  // scalar ops
  //  max, min, abs, exp, log, pow
  flow::Workspace::register_filter_type<expressions::ExprScalarMax>();
  flow::Workspace::register_filter_type<expressions::ExprScalarMin>();
  flow::Workspace::register_filter_type<expressions::ExprScalarAbs>();
  flow::Workspace::register_filter_type<expressions::ExprScalarPow>();
  flow::Workspace::register_filter_type<expressions::ExprScalarExp>();
  flow::Workspace::register_filter_type<expressions::ExprScalarLog>();

  // vec ops
  //   vector, vector_magnitude
  flow::Workspace::register_filter_type<expressions::ExprVector>();
  flow::Workspace::register_filter_type<expressions::ExprVectorMagnitude>();

  // array ops
  // access, replace, max, min, avg, sum
  flow::Workspace::register_filter_type<expressions::ExprArrayAccess>();
  flow::Workspace::register_filter_type<expressions::ExprArrayReplace>();
  flow::Workspace::register_filter_type<expressions::ExprArrayReductionMax>();
  flow::Workspace::register_filter_type<expressions::ExprArrayReductionMin>();
  flow::Workspace::register_filter_type<expressions::ExprArrayReductionAvg>();
  flow::Workspace::register_filter_type<expressions::ExprArrayReductionSum>();

  // history ops
  //  history, history_range, history_gradient, history_range_gradient
  flow::Workspace::register_filter_type<expressions::ExprHistory>();
  flow::Workspace::register_filter_type<expressions::ExprHistoryRange>();
  flow::Workspace::register_filter_type<expressions::ExprHistoryGradient>();
  flow::Workspace::register_filter_type<expressions::ExprHistoryGradientRange>();

  // histogram ops
  // histogram, entropy, pdf, cdf, quantile, bin_by_value, bin_by_index
  flow::Workspace::register_filter_type<expressions::ExprHistogram>();
  flow::Workspace::register_filter_type<expressions::ExprHistogramEntropy>();
  flow::Workspace::register_filter_type<expressions::ExprHistogramPDF>();
  flow::Workspace::register_filter_type<expressions::ExprHistogramCDF>();
  flow::Workspace::register_filter_type<expressions::ExprHistogramCDFQuantile>();
  flow::Workspace::register_filter_type<expressions::ExprHistogramBinByValue>();
  flow::Workspace::register_filter_type<expressions::ExprHistogramBinByIndex>();

  // mesh ops
  //   cycle, time, field, topology, bounds, lineout
  flow::Workspace::register_filter_type<expressions::ExprMeshCycle>();
  flow::Workspace::register_filter_type<expressions::ExprMeshTime>();
  flow::Workspace::register_filter_type<expressions::ExprMeshField>();
  flow::Workspace::register_filter_type<expressions::ExprMeshTopology>();
  flow::Workspace::register_filter_type<expressions::ExprMeshBounds>();
  flow::Workspace::register_filter_type<expressions::ExprMeshLineout>();

  // mesh field reduce ops
  //  max reduce, min reduce, avg reduce, nan count, inf count, sum reduce
  flow::Workspace::register_filter_type<expressions::ExprMeshFieldReductionMax>();
  flow::Workspace::register_filter_type<expressions::ExprMeshFieldReductionMin>();
  flow::Workspace::register_filter_type<expressions::ExprMeshFieldReductionAvg>();
  flow::Workspace::register_filter_type<expressions::ExprMeshFieldReductionSum>();
  flow::Workspace::register_filter_type<expressions::ExprMeshFieldReductionNanCount>();
  flow::Workspace::register_filter_type<expressions::ExprMeshFieldReductionInfCount>();

  // binning ops
  //  binning, binning_axis, bin_by_index, point_and_axis, max_from_point
  flow::Workspace::register_filter_type<expressions::ExprMeshBinning>();
  flow::Workspace::register_filter_type<expressions::ExprMeshBinningAxis>();
  flow::Workspace::register_filter_type<expressions::ExprMeshBinningBinByIndex>();
  flow::Workspace::register_filter_type<expressions::ExprMeshBinningPointAndAxis>();
  flow::Workspace::register_filter_type<expressions::ExprMeshBinningMaxFromPoint>();

  initialize_functions();
  initialize_objects();
}

ExpressionEval::ExpressionEval(conduit::Node *data)
{
  // wrap the pointer in a data object we can assume that this
  // is a valid multi-domain dataset
  conduit::Node *data_node = new conduit::Node();
  data_node->set_external(*data);
  m_data_object.reset(data_node);
}

ExpressionEval::ExpressionEval(DataObject &dataset)
  : m_data_object(dataset)
{
}

void
ExpressionEval::load_cache(const std::string &dir, const std::string &session)
{
  // the cache is static so don't load if we already have
  if(!m_cache.loaded())
  {
    m_cache.load(dir, session);
  }
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
  conduit::Node &array_replace_sig = (*functions)["replace"].append();
  array_replace_sig["return_type"] = "array";
  array_replace_sig["filter_name"] = "expr_array_replace"; // matches the filter's type_name
  array_replace_sig["args/arg1/type"] = "array";
  array_replace_sig["args/find/type"] = "double";
  array_replace_sig["args/find/description"] = "Value in the array to find and replace.";
  array_replace_sig["args/replace/type"] = "double";
  array_replace_sig["args/replace/description"] = "Replacement value.";
  array_replace_sig["description"] = "Find and replace zero or more values in an array.";

  // -------------------------------------------------------------
  conduit::Node &nan_sig = (*functions)["nan"].append();
  nan_sig["return_type"] = "double";
  nan_sig["filter_name"] = "expr_nan"; // matches the filter's type_name
  nan_sig["args"] = conduit::DataType::empty();
  nan_sig["description"] = "Generates a NaN value.";


  //---------------------------------------------------------------------------

  // gradient is an interesting case. Currently function matching picks the first
  // match and breaks so this if we don't list the most restrictive first, bad
  // things will happen (get a syntax error or something).
  conduit::Node &field_gradient_sig = (*functions)["gradient"].append();
  field_gradient_sig["return_type"] = "jitable";
  field_gradient_sig["filter_name"] = "expr_jit_mesh_field_gradient";
  field_gradient_sig["args/field/type"] = "field";
  field_gradient_sig["description"] =
      "Return a derived field that is the gradient of a field.";
  field_gradient_sig["jitable"];

  //---------------------------------------------------------------------------
  conduit::Node &scalar_gradient_sig = (*functions)["gradient"].append();
  scalar_gradient_sig["return_type"] = "double";
  scalar_gradient_sig["filter_name"] = "expr_history_gradient";

  //scalar_gradient_sig["args/expr_name/type"] = "string";
  scalar_gradient_sig["args/expr_name/type"] = "anytype";
  scalar_gradient_sig["args/expr_name/description"] =
      "`expr_name` should be the name of an expression that was evaluated in "
      "the past.";

  scalar_gradient_sig["args/window_length/type"] = "scalar";
  scalar_gradient_sig["args/window_length/optional"];
  scalar_gradient_sig["args/window_length/description"] = "The number of time points ago to use as the x0 for the gradient calculation. defaults to ``1`` \
    (calculate the gradient from the previous time point to now).";

  scalar_gradient_sig["args/window_length_unit/type"] =  "string";
  scalar_gradient_sig["args/window_length_unit/optional"];
  scalar_gradient_sig["args/window_length_unit/description"] = "Can be one of three values: ``\"index\"``, ``\"time\"`` or ``\"cycle\".`` \
    Indicates whether the window length is in units of number of expression execution points, simulation time, or simulation cycles. \
    Defaults to ``index`` (the window_length is in number of expression execution points).";

  scalar_gradient_sig["description"] = "Return the temporal gradient of the given expression for the current point in time.";

  // -------------------------------------------------------------
  // NOTE: 2023-07-13 CYRUS
  // `history_gradient` is a new user facing name for what was `gradient` applied to history
  conduit::Node &hist_gradient_sig = (*functions)["history_gradient"].append();
  // same sig as old "gradient_range"
  hist_gradient_sig.set(scalar_gradient_sig);

  // -------------------------------------------------------------

  conduit::Node &array_gradient_sig = (*functions)["gradient_range"].append();
  array_gradient_sig["return_type"] = "array";
  array_gradient_sig["filter_name"] = "expr_history_gradient_range";

  array_gradient_sig["args/expr_name/type"] = "anytype";
  array_gradient_sig["args/expr_name/description"] =
      "`expr_name` should be the name of an expression that was evaluated in "
      "the past.";

  array_gradient_sig["args/first_relative_index/type"] = "int";
  array_gradient_sig["args/first_relative_index/optional"];
  array_gradient_sig["args/first_relative_index/description"] = "The the first number of evaluations ago for which to calculate the temporal gradient. \
  The index is relative, with ``first_relative_index=1`` corresponding to one evaluation ago. Example usage: \
  gradient_range(pressure, first_relative_index=1, last_relative_index=10). This will calculate the temporal gradient for the previous 10 evaluations.";

  array_gradient_sig["args/last_relative_index/type"] = "int";
  array_gradient_sig["args/last_relative_index/optional"];
  array_gradient_sig["args/last_relative_index/description"] = "The the last number of evaluations ago for which to calculate the temporal gradient. \
  The index is relative, with ``last_relative_index=1`` corresponding to one evaluation ago. Example usage: \
  gradient_range(pressure, first_relative_index=1, last_relative_index=10). This will calculate the temporal gradient for the previous 10 evaluations.";

  array_gradient_sig["args/first_absolute_index/type"] = "int";
  array_gradient_sig["args/first_absolute_index/optional"];
  array_gradient_sig["args/first_absolute_index/description"] =
      "The first index in the evaluation \
  history for which to calculate the temporal gradient. This should be less than the number of past evaluations. For \
  example, ``gradient_range(pressure, first_absolute_index=0, last_absolute_index=10)`` calculates the temporal gradient of pressure from the first 10 times it was evaluated.";

  array_gradient_sig["args/last_absolute_index/type"] = "int";
  array_gradient_sig["args/last_absolute_index/optional"];
  array_gradient_sig["args/last_absolute_index/description"] =
      "The last index in the evaluation \
  history for which to calculate the temporal gradient. This should be less than the number of past evaluations. For \
  example, ``gradient_range(pressure, first_absolute_index=0, last_absolute_index=10)`` calculates the temporal gradient of pressure from the first 10 times it was evaluated.";

  array_gradient_sig["args/first_absolute_time/type"] = "scalar";
  array_gradient_sig["args/first_absolute_time/optional"];
  array_gradient_sig["args/first_absolute_time/description"] =
      "The first simulation time for which to calculate the temporal gradient. For \
  example, ``gradient_range(pressure, first_absolute_time=0, last_absolute_time=0.1)`` calculates the temporal gradient of \
  pressure from the first 0.1 units of simulation time.";

  array_gradient_sig["args/last_absolute_time/type"] = "scalar";
  array_gradient_sig["args/last_absolute_time/optional"];
  array_gradient_sig["args/last_absolute_time/description"] =
      "The last simulation time for which to calculate the temporal gradient. For \
  example, ``gradient_range(pressure, first_absolute_time=0, last_absolute_time=0.1)`` calculates the temporal gradient of \
  pressure from the first 0.1 units of simulation time.";

  array_gradient_sig["args/first_absolute_cycle/type"] = "scalar";
  array_gradient_sig["args/first_absolute_cycle/optional"];
  array_gradient_sig["args/first_absolute_cycle/description"] =
      "The first simulation cycle for which to calculate the temporal gradient. For \
  example, ``gradient_range(pressure, first_absolute_cycle=0, last_absolute_cycle=1)`` calculates the temporal gradient of \
  pressure from the first two cycles.";

  array_gradient_sig["args/last_absolute_cycle/type"] = "scalar";
  array_gradient_sig["args/last_absolute_cycle/optional"];
  array_gradient_sig["args/last_absolute_cycle/description"] =
      "The last simulation cycle for which to calculate the temporal gradient. For \
  example, ``gradient_range(pressure, first_absolute_cycle=0, last_absolute_cycle=1)`` calculate the temporal gradient of \
  pressure from the first two cycles.";

  array_gradient_sig["description"] = "As the simulation progresses the expressions \
  are evaluated repeatedly. The gradient_range function allows you to get the temporal gradient from a range of \
  previous evaluations. For example, if we want to evaluate the difference \
  between the original state of the simulation and the current state then we \
  can use an first absolute index of 0 and a last absolute index of 10 to compare the initial values with the \
  current value: ``gradient(val) - avg(gradient_range(val, first_absolute_index=0, last_absolute_index=10)``. Another example is if \
  you want to evaluate the relative change between the previous states and the \
  current state: ``gradient(val) - avg(gradient_range(val, first_relative_index=1, last_relative_index=10))``\
  We can alternatively evaluate the difference between a particular range of time in the simulation, \
  such as the first 10 seconds, and the current state: ``gradient(val) - avg(gradient_range(val, first_absolute_time=1, last_absolute_index=10))`` \
  or for the first 10 cycles of the simulation ``gradient(val) - avg(gradient_range(val, first_absolute_cycle=0, last_absolute_cycle=9))``.\n\n \
  .. note:: Exactly one of the following pairs of values must be provided: 1). first_absolute_index and last_absolute_index, 2). \
  first_relative_index and last_relative_index, 3). first_absolute_time and last_absolute_time, or 4). first_absolute_cycle and last_absolute_cycle.";

  // -------------------------------------------------------------
  // NOTE: 2023-07-13 CYRUS
  // `history_gradient_range` is a new user facing name for what was `gradient_range` applied to history
  conduit::Node &hist_gradient_range_sig = (*functions)["history_gradient_range"].append();
  // same sig as old "gradient_range"
  hist_gradient_range_sig.set(array_gradient_sig);


  //---------------------------------------------------------------------------
  // abs()
  //---------------------------------------------------------------------------
  conduit::Node &abs_sig = (*functions)["abs"].append();
  abs_sig["return_type"] = "scalar";
  abs_sig["filter_name"] = "expr_scalar_abs";
  abs_sig["args/arg1/type"] = "scalar";
  abs_sig["description"] = "Return the absolute value of the input.";

  //---------------------------------------------------------------------------
  // exp()
  //---------------------------------------------------------------------------
  conduit::Node &exp_sig = (*functions)["exp"].append();
  exp_sig["return_type"] = "double";
  exp_sig["filter_name"] = "expr_scalar_exp";
  exp_sig["args/arg1/type"] = "scalar";
  exp_sig["description"] = "Return the base e exponential.";

  //---------------------------------------------------------------------------
  // pow()
  //---------------------------------------------------------------------------
  conduit::Node &pow_sig = (*functions)["pow"].append();
  pow_sig["return_type"] = "double";
  pow_sig["filter_name"] = "expr_scalar_pow";
  pow_sig["args/arg1/type"] = "scalar";
  pow_sig["args/arg2/type"] = "scalar";
  pow_sig["description"] =
    "Returns base raised to the power exponent."
    " pow(base, exponent)";

  //---------------------------------------------------------------------------
  // log()
  //---------------------------------------------------------------------------
  conduit::Node &log_sig = (*functions)["log"].append();
  log_sig["return_type"] = "double";
  log_sig["filter_name"] = "expr_scalar_log";
  log_sig["args/arg1/type"] = "scalar";
  log_sig["description"] =
    "Returns the natural logarithm of the argument";

  // -------------------------------------------------------------
  // vector()
  // -------------------------------------------------------------
  conduit::Node &vector = (*functions)["vector"].append();
  vector["return_type"] = "vector";
  vector["filter_name"] = "expr_vector";
  vector["args/arg1/type"] = "scalar";
  vector["args/arg2/type"] = "scalar";
  vector["args/arg3/type"] = "scalar";
  vector["description"] = "Return the 3D position vector for the input value.";

  // -------------------------------------------------------------
  // magnitude()
  // -------------------------------------------------------------
  conduit::Node &mag_sig = (*functions)["magnitude"].append();
  mag_sig["return_type"] = "double";
  mag_sig["filter_name"] = "expr_vector_magnitude";
  mag_sig["args/arg1/type"] = "vector";
  mag_sig["description"] = "Return the magnitude of the input vector.";


  //---------------------------------------------------------------------------
  //  field_nan_count() 
  //---------------------------------------------------------------------------
  conduit::Node &field_nan_sig = (*functions)["field_nan_count"].append();
  field_nan_sig["return_type"] = "double";
  field_nan_sig["filter_name"] = "expr_mesh_field_reduction_nan_count";
  field_nan_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_nan_sig["description"] = "Return the number  of NaNs in a field.";

  //---------------------------------------------------------------------------
  //  field_inf_count() 
  //---------------------------------------------------------------------------
  conduit::Node &field_inf_sig = (*functions)["field_inf_count"].append();
  field_inf_sig["return_type"] = "double";
  field_inf_sig["filter_name"] = "expr_mesh_field_reduction_inf_count";
  field_inf_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_inf_sig["description"] =
      "Return the number  of -inf and +inf in a field.";


  //---------------------------------------------------------------------------
  //  min() 
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  conduit::Node &scalar_min_sig = (*functions)["min"].append();
  scalar_min_sig["return_type"] = "double";
  scalar_min_sig["filter_name"] = "expr_scalar_min";
  scalar_min_sig["args/arg1/type"] = "scalar";
  scalar_min_sig["args/arg2/type"] = "scalar";
  scalar_min_sig["description"] = "Return the minimum of two scalars.";

  //---------------------------------------------------------------------------
  conduit::Node &array_min_sig = (*functions)["min"].append();
  array_min_sig["return_type"] = "double";
  array_min_sig["filter_name"] = "expr_array_reduction_min";
  array_min_sig["args/arg1/type"] = "array";
  array_min_sig["description"] = "Return the minimum of an array.";

  //---------------------------------------------------------------------------
  conduit::Node &field_min_sig = (*functions)["min"].append();
  field_min_sig["return_type"] = "value_position";
  field_min_sig["filter_name"] = "expr_mesh_field_reduction_min";
  field_min_sig["args/arg1/type"] = "field";
  field_min_sig["description"] =
      "Return the minimum value from the meshvar. Its position is also "
      "stored "
      "and is accessible via the `position` function.";

  //---------------------------------------------------------------------------
  //  max()
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  conduit::Node &scalar_max_sig = (*functions)["max"].append();
  scalar_max_sig["return_type"] = "double";
  scalar_max_sig["filter_name"] = "expr_scalar_max";
  scalar_max_sig["args/arg1/type"] = "scalar";
  scalar_max_sig["args/arg2/type"] = "scalar";
  scalar_max_sig["description"] = "Return the maximum of two scalars.";

  //---------------------------------------------------------------------------
  conduit::Node &array_max_sig = (*functions)["max"].append();
  array_max_sig["return_type"] = "double";
  array_max_sig["filter_name"] = "expr_array_reduction_max";
  array_max_sig["args/arg1/type"] = "array";
  array_max_sig["description"] = "Return the maximum of an array.";

  //---------------------------------------------------------------------------
  conduit::Node &field_max_sig = (*functions)["max"].append();
  field_max_sig["return_type"] = "value_position";
  field_max_sig["filter_name"] = "expr_mesh_field_reduction_max";
  field_max_sig["args/arg1/type"] = "field";
  field_max_sig["description"] =
      "Return the maximum value from the meshvar. Its position is also "
      "stored "
      "and is accessible via the `position` function.";

  //---------------------------------------------------------------------------
  //  avg()
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  conduit::Node &array_avg_sig = (*functions)["avg"].append();
  array_avg_sig["return_type"] = "double";
  array_avg_sig["filter_name"] = "expr_array_reduction_avg"; // matches the filter's type_name
  array_avg_sig["args/arg1/type"] = "array"; // arg names match input port names
  array_avg_sig["description"] = "Return the average of an array.";

  // -------------------------------------------------------------
  conduit::Node &field_avg_sig = (*functions)["avg"].append();
  field_avg_sig["return_type"] = "double";
  field_avg_sig["filter_name"] = "expr_mesh_field_reduction_avg";
  field_avg_sig["args/arg1/type"] = "field";
  field_avg_sig["description"] = "Return the field average of a field.";

  //---------------------------------------------------------------------------
  //  sum()
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  conduit::Node &array_sum_sig = (*functions)["sum"].append();
  array_sum_sig["return_type"] = "double";
  array_sum_sig["filter_name"] = "expr_array_reduction_sum";
  array_sum_sig["args/arg1/type"] = "array";
  array_sum_sig["description"] = "Return the sum of an array.";

  //---------------------------------------------------------------------------
  conduit::Node &field_sum_sig = (*functions)["sum"].append();
  field_sum_sig["return_type"] = "double";
  field_sum_sig["filter_name"] = "expr_mesh_field_reduction_sum";
  field_sum_sig["args/arg1/type"] = "field";
  field_sum_sig["description"] = "Return the sum of a field.";

  //---------------------------------------------------------------------------
  //  cycle()
  //---------------------------------------------------------------------------
  conduit::Node &cycle_sig = (*functions)["cycle"].append();
  cycle_sig["return_type"] = "int";
  cycle_sig["filter_name"] = "expr_mesh_cycle";
  cycle_sig["args"] = conduit::DataType::empty();
  cycle_sig["description"] = "Return the current simulation cycle.";

  //---------------------------------------------------------------------------
  //  time()
  //---------------------------------------------------------------------------
  conduit::Node &time_sig = (*functions)["time"].append();
  time_sig["return_type"] = "double";
  time_sig["filter_name"] = "expr_mesh_time";
  time_sig["args"] = conduit::DataType::empty();
  time_sig["description"] = "Return the current simulation time.";

  // -------------------------------------------------------------
  conduit::Node &hist_sig = (*functions)["histogram"].append();
  hist_sig["return_type"] = "histogram";
  hist_sig["filter_name"] = "expr_histogram";
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

  hist_sig["description"] = "Return a histogram of the field.";

  //---------------------------------------------------------------------------
  // history()
  //---------------------------------------------------------------------------
  conduit::Node &history_sig = (*functions)["history"].append();
  history_sig["return_type"] = "anytype";
  history_sig["filter_name"] = "expr_history";

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

  //---------------------------------------------------------------------------
  // history_range()
  //---------------------------------------------------------------------------
  conduit::Node &history_range_sig = (*functions)["history_range"].append();
  // history_range_sig["return_type"] = "anytype";
  history_range_sig["return_type"] = "array";
  history_range_sig["filter_name"] = "expr_history_range";

  history_range_sig["args/expr_name/type"] = "anytype";
  history_range_sig["args/expr_name/description"] =
      "`expr_name` should be the name of an expression that was evaluated in "
      "the past.";

  history_range_sig["args/first_relative_index/type"] = "int";
  history_range_sig["args/first_relative_index/optional"];
  history_range_sig["args/first_relative_index/description"] = "The the first number of evaluations ago for which to retrieve past expression values. \
  The index is relative, with ``first_relative_index=1`` corresponding to one evaluation ago. Example usage: \
  history_range(pressure, first_relative_index=1, last_relative_index=10). This will retrieve the value \
  for the previous 10 evaluations.";

  history_range_sig["args/last_relative_index/type"] = "int";
  history_range_sig["args/last_relative_index/optional"];
  history_range_sig["args/last_relative_index/description"] = "The the last number of evaluations ago for which to retrieve past expression values. \
  The index is relative, with ``last_relative_index=1`` corresponding to one evaluation ago. Example usage: \
  history_range(pressure, first_relative_index=1, last_relative_index=10). This will retrieve the value \
  for the previous 10 evaluations.";

  history_range_sig["args/first_absolute_index/type"] = "int";
  history_range_sig["args/first_absolute_index/optional"];
  history_range_sig["args/first_absolute_index/description"] =
      "The first index in the evaluation \
  history for which to retrieve values. This should be less than the number of past evaluations. For \
  example, ``history_range(pressure, first_absolute_index=0, last_absolute_index=10)`` returns the value of \
  pressure from the first 10 times it was evaluated.";

  history_range_sig["args/last_absolute_index/type"] = "int";
  history_range_sig["args/last_absolute_index/optional"];
  history_range_sig["args/last_absolute_index/description"] =
      "The last index in the evaluation \
  history for which to retrieve values. This should be less than the number of past evaluations. For \
  example, ``history_range(pressure, first_absolute_index=0, last_absolute_index=10)`` returns the value of \
  pressure from the first 10 times it was evaluated.";

  history_range_sig["args/first_absolute_time/type"] = "scalar";
  history_range_sig["args/first_absolute_time/optional"];
  history_range_sig["args/first_absolute_time/description"] =
      "The first simulation time for which to retrieve values. For \
  example, ``history_range(pressure, first_absolute_time=0, last_absolute_time=0.1)`` returns the value of \
  pressure from the first 0.1 units of simulation time.";

  history_range_sig["args/last_absolute_time/type"] = "scalar";
  history_range_sig["args/last_absolute_time/optional"];
  history_range_sig["args/last_absolute_time/description"] =
      "The last simulation time for which to retrieve values. For \
  example, ``history_range(pressure, first_absolute_time=0, last_absolute_time=0.1)`` returns the value of \
  pressure from the first 0.1 units of simulation time.";

  history_range_sig["args/first_absolute_cycle/type"] = "scalar";
  history_range_sig["args/first_absolute_cycle/optional"];
  history_range_sig["args/first_absolute_cycle/description"] =
      "The first simulation cycle for which to retrieve values. For \
  example, ``history_range(pressure, first_absolute_cycle=0, last_absolute_cycle=1)`` returns the value of \
  pressure from the first two cycles.";

  history_range_sig["args/last_absolute_cycle/type"] = "scalar";
  history_range_sig["args/last_absolute_cycle/optional"];
  history_range_sig["args/last_absolute_cycle/description"] =
      "The last simulation cycle for which to retrieve values. For \
  example, ``history_range(pressure, first_absolute_cycle=0, last_absolute_cyclee=1)`` returns the value of \
  pressure from the first two cycles.";

  history_range_sig["description"] = "As the simulation progresses the expressions \
  are evaluated repeatedly. The history_range function allows you to get the value from a range of \
  previous evaluations. For example, if we want to evaluate the difference \
  between the original state of the simulation and the current state then we \
  can use an first absolute index of 0 and a last absolute index of 10 to compare the initial values with the \
  current value: ``val - avg(history_range(val, first_absolute_index=0, last_absolute_index=10)``. Another example is if \
  you want to evaluate the relative change between the previous states and the \
  current state: ``val - avg(history_range(val, first_relative_index=1, last_relative_index=10))``\
  We can alternatively evaluate the difference between a particular range of time in the simulation, \
  such as the first 10 seconds, and the current state: ``val - avg(history_range(val, first_absolute_time=1, last_absolute_index=10))`` \
  or for the first 10 cycles of the simulation ``val - avg(history_range(val, first_absolute_cycle=0, last_absolute_cycle=9))``.\n\n \
  .. note:: Exactly one of the following pairs of values must be provided: 1). first_absolute_index and last_absolute_index, 2). \
  first_relative_index and last_relative_index, 3). first_absolute_time and last_absolute_time, or 4). first_absolute_cycle and last_absolute_cycle.";

  // -------------------------------------------------------------
  // entropy()
  // -------------------------------------------------------------
  conduit::Node &entropy_sig = (*functions)["entropy"].append();
  entropy_sig["return_type"] = "double";
  entropy_sig["filter_name"] = "expr_histogram_entropy";
  entropy_sig["args/hist/type"] = "histogram";
  entropy_sig["description"] =
      "Return the Shannon entropy given a histogram of the field.";

  //---------------------------------------------------------------------------
  // pdf()
  //---------------------------------------------------------------------------
  conduit::Node &pdf_sig = (*functions)["pdf"].append();
  pdf_sig["return_type"] = "histogram";
  pdf_sig["filter_name"] = "expr_histogram_pdf";
  pdf_sig["args/hist/type"] = "histogram";
  pdf_sig["description"] =
      "Return the probability distribution function (pdf) from a histogram.";

  //---------------------------------------------------------------------------
  // cdf()
  //---------------------------------------------------------------------------
  conduit::Node &cdf_sig = (*functions)["cdf"].append();
  cdf_sig["return_type"] = "histogram";
  cdf_sig["filter_name"] = "expr_histogram_cdf";
  cdf_sig["args/hist/type"] = "histogram";
  cdf_sig["description"] =
      "Return the cumulative distribution function (cdf) from a histogram.";

  //---------------------------------------------------------------------------
  // quantile()
  //---------------------------------------------------------------------------
  conduit::Node &quantile_sig = (*functions)["quantile"].append();
  quantile_sig["return_type"] = "double";
  quantile_sig["filter_name"] = "expr_histogram_cdf_quantile";
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
  x-axis which 50 percent of the data lies below.";

  //---------------------------------------------------------------------------
  // bin()
  //---------------------------------------------------------------------------
  // gets histogram bin by index
  conduit::Node &hist_bin_by_index_sig = (*functions)["bin"].append();
  hist_bin_by_index_sig["return_type"] = "double";
  hist_bin_by_index_sig["filter_name"] = "expr_histogram_bin_by_index";
  hist_bin_by_index_sig["args/hist/type"] = "histogram";
  hist_bin_by_index_sig["args/bin/type"] = "int"; // index?
  hist_bin_by_index_sig["description"] =
      "Return the value of the bin at index `bin` of a histogram.";

  //---------------------------------------------------------------------------
  // gets histogram bin by value
  conduit::Node &hist_bin_by_value_sig = (*functions)["bin"].append();
  hist_bin_by_value_sig["return_type"] = "double";
  hist_bin_by_value_sig["filter_name"] = "expr_histogram_bin_by_value";
  hist_bin_by_value_sig["args/hist/type"] = "histogram";
  hist_bin_by_value_sig["args/val/type"] = "scalar";
  hist_bin_by_value_sig["description"] =
      "Return the value of the bin with axis-value `val` on the histogram.";

  //---------------------------------------------------------------------------
  // field()
  //---------------------------------------------------------------------------
  conduit::Node &field_sig = (*functions)["field"].append();
  field_sig["return_type"] = "field";
  field_sig["filter_name"] = "expr_mesh_field";
  field_sig["args/field_name/type"] = "string";
  field_sig["args/component/type"] = "string";
  field_sig["args/component/optional"];
  field_sig["args/component/description"] =
      "Used to specify a single component if the field is a vector field.";
  field_sig["description"] = "Return a mesh field given its name.";

  //---------------------------------------------------------------------------
  // topo()
  //---------------------------------------------------------------------------
  conduit::Node &topo_sig = (*functions)["topo"].append();
  topo_sig["return_type"] = "topology";
  topo_sig["filter_name"] = "expr_mesh_topology";
  topo_sig["args/arg1/type"] = "string";
  topo_sig["description"] = "Return a mesh topology given its name.";

  //---------------------------------------------------------------------------
  // topology()
  //---------------------------------------------------------------------------
  conduit::Node &topology_sig = (*functions)["topology"].append();
  topology_sig["return_type"] = "topology";
  topology_sig["filter_name"] = "expr_mesh_topology";
  topology_sig["args/arg1/type"] = "string";
  topology_sig["description"] = "Return a mesh topology given a its name.";

  //---------------------------------------------------------------------------
  // bounds()
  //---------------------------------------------------------------------------
  conduit::Node &bounds_sig = (*functions)["bounds"].append();
  bounds_sig["return_type"] = "aabb";
  bounds_sig["filter_name"] = "expr_mesh_bounds";
  bounds_sig["args/topology/type"] = "string";
  bounds_sig["args/topology/optional"];
  bounds_sig["description"] = "Returns the spatial bounds of a mesh.";

  //---------------------------------------------------------------------------
  // lineout()
  //---------------------------------------------------------------------------
  conduit::Node &lineout = (*functions)["lineout"].append();
  lineout["return_type"] = "array";
  lineout["filter_name"] = "expr_mesh_lineout";
  lineout["args/samples/type"] = "int";
  lineout["args/start/type"] = "vector";
  lineout["args/end/type"] = "vector";
  lineout["args/fields/type"] = "list";
  lineout["args/fields/optional"];
  lineout["args/empty_val/type"] = "double";
  lineout["args/empty_val/optional"];
  lineout["description"] = "returns a sampled based line out";




  //---------------------------------------------------------------------------
  // point_and_axis()
  //---------------------------------------------------------------------------
  conduit::Node &point_and_axis_sig = (*functions)["point_and_axis"].append();
  point_and_axis_sig["return_type"] = "bin";
  point_and_axis_sig["filter_name"] = "expr_mesh_binning_point_and_axis";
  point_and_axis_sig["args/binning/type"] = "binning";
  point_and_axis_sig["args/axis/type"] = "string";
  point_and_axis_sig["args/threshold/type"] = "double";
  point_and_axis_sig["args/point/type"] = "double";
  point_and_axis_sig["args/miss_value/type"] = "scalar";
  point_and_axis_sig["args/miss_value/optional"];
  point_and_axis_sig["args/direction/type"] = "int";
  point_and_axis_sig["args/direction/optional"];
  point_and_axis_sig["description"] =
      "returns the first values in"
      " a binning that exceeds a threshold from the given point.";

  conduit::Node &bin_sig = (*functions)["bin"].append();
  bin_sig["return_type"] = "bin";
  bin_sig["filter_name"] = "expr_mesh_binning_bin_by_index";
  bin_sig["args/binning/type"] = "binning";
  bin_sig["args/index/type"] = "int";
  bin_sig["description"] = "returns a bin from a binning by index";

  //---------------------------------------------------------------------------
  // max_from_point()
  //---------------------------------------------------------------------------
  conduit::Node &max_from_point_sig = (*functions)["max_from_point"].append();
  max_from_point_sig["return_type"] = "value_position";
  max_from_point_sig["filter_name"] = "expr_mesh_binning_max_from_point";
  max_from_point_sig["args/binning/type"] = "binning";
  max_from_point_sig["args/axis/type"] = "string";
  max_from_point_sig["args/point/type"] = "double";
  max_from_point_sig["description"] =
      "returns the closest max"
      " value from a reference point on an axis";


  //---------------------------------------------------------------------------
  // axis()
  //---------------------------------------------------------------------------
  
  //---------------------------------------------------------------------------
  conduit::Node &axis_sig = (*functions)["axis"].append();
  axis_sig["return_type"] = "axis";
  axis_sig["filter_name"] = "expr_mesh_binning_axis";
  axis_sig["args/name/type"] = "string";
  axis_sig["args/name/description"] =
      "The name of a scalar field on the mesh "
      "or one of ``'x'``, ``'y'``, or ``'z'``. `name` can also be the empty "
      "string `''` if `reduction_op` is either `sum` or `pdf` to mean we want "
      "to count the number of elements in the bin as our reduction variable.";
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
      "Minimum value of the axis (i.e. the value of the first tick). Defaults "
      "to ``min(name)`` for fields and for ``'x'``, ``'y'``, or ``'z'`` the "
      "minimum value on the topology.";
  axis_sig["args/max_val/type"] = "scalar";
  axis_sig["args/max_val/optional"];
  axis_sig["args/max_val/description"] =
      "Maximum value of the axis (i.e. the value of the last tick).Defaults to "
      "``max(name)`` for fields and for ``'x'``, ``'y'``, or ``'z'`` the "
      "maximum value on the topology.";
  axis_sig["args/num_bins/type"] = "int";
  axis_sig["args/num_bins/optional"];
  axis_sig["args/num_bins/description"] =
      "Number of bins on the axis (i.e. the number of ticks minus 1). Defaults "
      "to ``256``.";
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

  //---------------------------------------------------------------------------
  conduit::Node &axis_sig2 = (*functions)["axis"].append();
  axis_sig2["return_type"] = "axis";
  axis_sig2["filter_name"] = "expr_mesh_binning_axis";
  axis_sig2["args/var/type"] = "string";
  axis_sig2["args/var/description"] = "One of the strings ``'x', 'y', 'z'`` "
                                      "corresponding to a spacial coordinate.";
  // rectilinear binning
  axis_sig2["args/bins/type"] = "list";
  axis_sig2["args/bins/optional"];
  // uniform binning
  axis_sig2["args/min_val/type"] = "scalar";
  axis_sig2["args/min_val/optional"];
  axis_sig2["args/max_val/type"] = "scalar";
  axis_sig2["args/max_val/optional"];
  axis_sig2["args/num_bins/type"] = "int";
  axis_sig2["args/num_bins/optional"];
  axis_sig2["args/clamp/type"] = "bool";
  axis_sig2["args/clamp/optional"];
  axis_sig2["description"] =
      "Same as the above function except that ``reduction_var`` should be one "
      "of the strings ``'x', 'y', 'z'``";

  //---------------------------------------------------------------------------
  // binning()
  //---------------------------------------------------------------------------
  conduit::Node &binning_sig = (*functions)["binning"].append();
  binning_sig["return_type"] = "binning";
  binning_sig["filter_name"] = "expr_mesh_binning";
  binning_sig["args/reduction_var/type"] = "string";
  binning_sig["args/reduction_var/description"] =
      "The variable being reduced. Either the name of a scalar field on the "
      "mesh or one of ``'x'``, ``'y'``, or ``'z'``.";
  binning_sig["args/reduction_op/type"] = "string";
  binning_sig["args/reduction_op/description"] =
      "The reduction operator to use when \
  putting values in bins. Available reductions are: \n\n \
  - min: minimum value in a bin \n \
  - max: maximum value in a bin \n \
  - sum: sum of values in a bin \n \
  - avg: average of values in a bin \n \
  - pdf: probability distribution function \n \
  - std: standard deviation of values in a bin \n \
  - var: variance of values in a bin \n \
  - rms: root mean square of values in a bin";
  binning_sig["args/bin_axes/type"] = "list";
  binning_sig["args/bin_axes/description"] =
      "List of Axis objects which define the bin axes.";
  binning_sig["args/empty_bin_val/type"] = "scalar";
  binning_sig["args/empty_bin_val/optional"];
  binning_sig["args/empty_bin_val/description"] =
      "The value that empty bins should have. Defaults to ``0``.";
  binning_sig["args/component/type"] = "string";
  binning_sig["args/component/optional"];
  binning_sig["args/component/description"] =
      "the component of a vector field to use for the reduction."
      " Example 'x' for a field defined as 'velocity/x'";
  binning_sig["description"] = "Returns a multidimensional data binning.";

  //---------------------------------------------------------------------------
//
//  conduit::Node &binning_sig2 = (*functions)["binning"].append();
//  binning_sig2["return_type"] = "binning";
//  binning_sig2["filter_name"] = "binning";
//  binning_sig2["args/reduction_var/type"] = "string";
//  binning_sig2["args/reduction_var/description"] =
//      "One of the strings ``'x', 'y', 'z'`` corresponding to a spacial "
//      "coordinate. ``reduction_var`` can be ``'cnt'`` to mean "
//      "\"bin the count\" if ``reduction_op`` is one of ``sum``, ``pdf``, or "
//      "``cdf``.";
//  binning_sig2["args/reduction_op/type"] = "string";
//  binning_sig2["args/bin_axes/type"] = "list";
//  binning_sig2["args/bin_axes/description"] =
//      "List of Axis objects which define the bin axes.";
//  binning_sig2["args/empty_val/type"] = "scalar";
//  binning_sig2["args/empty_val/optional"];
//  binning_sig2["args/topo/type"] = "topology";
//  binning_sig2["args/topo/optional"];
//  binning_sig2["args/topo/description"] =
//      "The topology to bin. Defaults to the "
//      "topology associated with the bin axes. This topology must have "
//      "all the fields used for the axes of ``binning``. It only makes sense "
//      "to specify this when ``bin_axes`` and ``reduction_var`` are a "
//      "subset of ``x``, ``y``, ``z``.";
//  binning_sig2["args/assoc/type"] = "string";
//  binning_sig2["args/assoc/optional"];
//  binning_sig2["args/assoc/description"] =
//      "The association of the resultant field. Defaults to the association "
//      "infered from the bin axes and and reduction variable. It only "
//      "makes sense to specify this when ``bin_axes`` and ``reduction_var`` are "
//      "a subset of ``x``, ``y``, ``z``.";
//  binning_sig2["description"] =
//      "Returns a multidimensional data binning. Same as the above function "
//      "except that ``reduction_var`` should be one of the strings ``'x', 'y', "
//      "'z'`` and the association and topology can be explicitely specified.";
//
  //---------------------------------------------------------------------------

  // 2023-07-07 CYRUS NOTE
  // I DON"T THINK paint_binning or binning_mesh exprs are 
  // fully wired up or tests, I am commenting them out 

  // // this does not jit but binning_value does
  // conduit::Node &paint_binning_sig = (*functions)["paint_binning"].append();
  // paint_binning_sig["return_type"] = "field";
  // paint_binning_sig["filter_name"] = "paint_binning";
  // paint_binning_sig["args/binning/type"] = "binning";
  // paint_binning_sig["args/binning/description"] =
  //     "The values in ``binning`` are used to generate the new field.";
  // paint_binning_sig["args/name/type"] = "string";
  // paint_binning_sig["args/name/optional"];
  // paint_binning_sig["args/name/description"] =
  //     "The name of the new field to be generated. If not specified, a name "
  //     "is automatically generated and the field is treated as a temporary and "
  //     "removed from the dataset when the expression is done executing.";
  // paint_binning_sig["args/default_val/type"] = "scalar";
  // paint_binning_sig["args/default_val/optional"];
  // paint_binning_sig["args/default_val/description"] =
  //     "The value given to elements which do not fall into "
  //     "any of the bins. Defaults to ``0``.";
  // paint_binning_sig["args/topo/type"] = "topology";
  // paint_binning_sig["args/topo/optional"];
  // paint_binning_sig["args/topo/description"] =
  //     " The topology to paint the bin values back onto. Defaults to the "
  //     "topology associated with the bin axes. This topology must have "
  //     "all the fields used for the axes of ``binning``. It only makes sense "
  //     "to specify this when the ``bin_axes`` are a subset of ``x``, ``y``, "
  //     "``z``. Additionally, it must be specified in this case since there is "
  //     "not enough info to infer the topology assuming there are multiple "
  //     "topologies in the dataset.";
  // paint_binning_sig["args/assoc/type"] = "topology";
  // paint_binning_sig["args/assoc/optional"];
  // paint_binning_sig["args/assoc/description"] =
  //     "Defaults to the association inferred from the bin axes and and "
  //     "reduction variable. The association of the resultant field. This "
  //     "topology must have all the fields used for the axes of ``binning``. It "
  //     "only makes sense to specify this when the ``bin_axes`` are a subset of "
  //     "``x``, ``y``, ``z``.";
  // paint_binning_sig["description"] =
  //     "Paints back the bin values onto an existing mesh by binning the "
  //     "elements of the mesh and creating a new field there the value at each "
  //     "element is the value in the bin it falls into.";
  //
  // //---------------------------------------------------------------------------
  //
  // conduit::Node &binning_mesh_sig = (*functions)["binning_mesh"].append();
  // binning_mesh_sig["return_type"] = "field";
  // binning_mesh_sig["filter_name"] = "binning_mesh";
  // binning_mesh_sig["args/binning/type"] = "binning";
  // binning_mesh_sig["args/binning/description"] =
  //     "The values in ``binning`` are used to generate the new field.";
  // binning_mesh_sig["args/name/type"] = "string";
  // binning_mesh_sig["args/name/optional"];
  // binning_mesh_sig["args/name/description"] =
  //     "The name of the new field to be generated, the corresponding topology "
  //     "topology and coordinate sets will be named '``name``_topo' and "
  //     "'``name``_coords' respectively. If not specified, a name is "
  //     "automatically generated and the field is treated as a temporary and "
  //     "removed from the dataset when the expression is done executing.";
  // binning_mesh_sig["description"] =
  //     "A binning with 3 or fewer dimensions will be output as a new element "
  //     "associated field on a new topology on the dataset. This is useful for "
  //     "directly visualizing the binning.";

  //---------------------------------------------------------------------------
  // Jitable Functions
  //---------------------------------------------------------------------------
  // Functions below this line call JitFilter
  // All jitable functions need to have ["jitable"] in order for this to happen
  // filter_name is passed to JitFilter so that it can determine which function
  // to execute

  //---------------------------------------------------------------------------
  // min(field, scalar)
  conduit::Node &field_field_min_sig = (*functions)["min"].append();
  field_field_min_sig["return_type"] = "jitable";
  field_field_min_sig["filter_name"] = "expr_jit_mesh_field_min";
  field_field_min_sig["args/arg1/type"] = "field";
  field_field_min_sig["args/arg2/type"] = "scalar";
  field_field_min_sig["description"] =
      "Return a derived field that is the min of a field and a scalar.";

  //---------------------------------------------------------------------------
  // min(field, field)
  field_field_min_sig["jitable"];
  conduit::Node &field_scalar_min_sig = (*functions)["min"].append();
  field_scalar_min_sig["return_type"] = "jitable";
  field_scalar_min_sig["filter_name"] = "expr_jit_mesh_field_min";
  field_scalar_min_sig["args/arg1/type"] = "field";
  field_scalar_min_sig["args/arg2/type"] = "field";
  field_scalar_min_sig["description"] =
      "Return a derived field that is the min of two fields.";
  field_scalar_min_sig["jitable"];

  //---------------------------------------------------------------------------
  // max(field, scalar)
  conduit::Node &field_scalar_max_sig = (*functions)["max"].append();
  field_scalar_max_sig["return_type"] = "jitable";
  field_scalar_max_sig["filter_name"] = "expr_jit_mesh_field_max";
  field_scalar_max_sig["args/arg1/type"] = "field";
  field_scalar_max_sig["args/arg2/type"] = "scalar";
  field_scalar_max_sig["description"] =
      "Return a derived field that is the max of a field and a scalar.";
  field_scalar_max_sig["jitable"];

  //---------------------------------------------------------------------------
  // max(field, field)
  conduit::Node &field_field_max_sig = (*functions)["max"].append();
  field_field_max_sig["return_type"] = "jitable";
  field_field_max_sig["filter_name"] = "expr_jit_mesh_field_max";
  field_field_max_sig["args/arg1/type"] = "field";
  field_field_max_sig["args/arg2/type"] = "field";
  field_field_max_sig["description"] =
      "Return a derived field that is the max of two fields.";
  field_field_max_sig["jitable"];

  //---------------------------------------------------------------------------
  conduit::Node &field_sin_sig = (*functions)["sin"].append();
  field_sin_sig["return_type"] = "jitable";
  field_sin_sig["filter_name"] = "expr_jit_mesh_field_sin";
  field_sin_sig["args/arg1/type"] = "field";
  field_sin_sig["description"] =
      "Return a derived field that is the sin of a field.";
  field_sin_sig["jitable"];

  //---------------------------------------------------------------------------

  conduit::Node &field_pow_sig = (*functions)["pow"].append();
  field_pow_sig["return_type"] = "jitable";
  field_pow_sig["filter_name"] = "expr_jit_mesh_field_pow";
  field_pow_sig["args/arg1/type"] = "field";
  field_pow_sig["args/arg1/type"] = "scalar";
  field_pow_sig["description"] =
      "Return a derived field that is the pow(field, exponent) of a field.";
  field_pow_sig["jitable"];

  //---------------------------------------------------------------------------

  conduit::Node &field_abs_sig = (*functions)["abs"].append();
  field_abs_sig["return_type"] = "jitable";
  field_abs_sig["filter_name"] = "expr_jit_mesh_field_abs";
  field_abs_sig["args/arg1/type"] = "field";
  field_abs_sig["description"] =
      "Return a derived field that is the absolute value of a field.";
  field_abs_sig["jitable"];

  //---------------------------------------------------------------------------

  conduit::Node &field_sqrt_sig = (*functions)["sqrt"].append();
  field_sqrt_sig["return_type"] = "jitable";
  field_sqrt_sig["filter_name"] = "expr_jit_mesh_field_sqrt";
  field_sqrt_sig["args/arg1/type"] = "field";
  field_sqrt_sig["description"] =
      "Return a derived field that is the square root value of a field.";
  field_sqrt_sig["jitable"];

  //---------------------------------------------------------------------------


  conduit::Node &field_curl_sig = (*functions)["curl"].append();
  field_curl_sig["return_type"] = "jitable";
  field_curl_sig["filter_name"] = "expr_jit_mesh_field_curl";
  field_curl_sig["args/field/type"] = "field";
  field_curl_sig["description"] =
      "Return a derived field that is the curl of a vector field.";
  field_curl_sig["jitable"];

  //---------------------------------------------------------------------------

  conduit::Node &field_magnitude_sig = (*functions)["magnitude"].append();
  field_magnitude_sig["return_type"] = "jitable";
  field_magnitude_sig["filter_name"] = "expr_jit_mesh_field_vector_magnitude";
  field_magnitude_sig["args/vector/type"] = "field";
  field_magnitude_sig["description"] =
      "Return a derived field that is the magnitude of a vector field.";
  field_magnitude_sig["jitable"];

  //---------------------------------------------------------------------------

  conduit::Node &field_vector = (*functions)["vector"].append();
  field_vector["return_type"] = "jitable";
  field_vector["filter_name"] = "expr_jit_mesh_field_vector_compose";
  field_vector["args/arg1/type"] = "field";
  field_vector["args/arg2/type"] = "field";
  field_vector["args/arg3/type"] = "field";
  field_vector["description"] = "Return a vector field on the mesh.";
  field_vector["jitable"];

  //---------------------------------------------------------------------------
  // NOTE:  2023-07-12 cyrush:
  // changed `derived_field` to `constant_field`
  conduit::Node &const_field = (*functions)["constant_field"].append();
  const_field["return_type"] = "jitable";
  const_field["filter_name"] = "expr_jit_mesh_field_constant";
  const_field["args/arg1/type"] = "scalar";
  const_field["args/arg1/description"] =
      "The scalar to be cast to a constant mesh field.";
  const_field["args/topo/type"] = "string";
  const_field["args/topo/optional"];
  const_field["args/topo/description"] =
      "The topology to put the field onto. The language tries to infer "
      "this if not specified.";
  const_field["args/assoc/type"] = "string";
  const_field["args/assoc/optional"];
  const_field["args/assoc/description"] =
      "The association of the field. The language tries to infer "
      "this if not specified.";
  const_field["description"] =
      "Cast a scalar to a constant mesh field (type `jitable`).";
  const_field["jitable"];

  //---------------------------------------------------------------------------
  // NOTE:  2023-07-12 cyrush:
  // changed `derived_field` to `constant_field`
  conduit::Node &const_field2 = (*functions)["constant_field"].append();
  const_field2["return_type"] = "jitable";
  const_field2["filter_name"] = "expr_jit_mesh_field_constant";
  const_field2["args/arg1/type"] = "field";
  const_field2["args/arg1/description"] =
      "The scalar to be cast to a constant mesh field.";
  const_field2["args/topo/type"] = "string";
  const_field2["args/topo/optional"];
  const_field2["args/topo/description"] =
      "The topology to put the field onto. The language tries to infer "
      "this if not specified.";
  const_field2["args/assoc/type"] = "string";
  const_field2["args/assoc/optional"];
  const_field2["args/assoc/description"] =
      "The association of the field. The language tries to infer "
      "this if not specified.";
  const_field2["description"] =
      "Used to explicitly specify the topology and association of a constant "
      "field (e.g. in case it cannot be inferred or needs to be changed).";
  const_field2["jitable"];

  //---------------------------------------------------------------------------
  conduit::Node &recenter_sig = (*functions)["recenter"].append();
  recenter_sig["return_type"] = "jitable";
  recenter_sig["filter_name"] = "expr_jit_mesh_field_recenter";
  recenter_sig["args/field/type"] = "field";
  recenter_sig["args/mode/type"] = "string";
  recenter_sig["args/mode/optional"] = "string";
  recenter_sig["args/mode/description"] =
      "One of ``'toggle', 'vertex', 'element'``. Defaults to ``'toggle'``.";
  recenter_sig["description"] = "Recenter a field from vertex association to "
                                "element association or vice versa.";
  recenter_sig["jitable"];

  //---------------------------------------------------------------------------
  conduit::Node &rand_sig = (*functions)["rand"].append();
  rand_sig["return_type"] = "jitable";
  rand_sig["filter_name"] = "expr_jit_rand";
  rand_sig["description"] = "Return a random number between 0 and 1.";
  rand_sig["jitable"];


  //---------------------------------------------------------------------------

  // essentially the jit version of paint_binning
  conduit::Node &binning_value_sig = (*functions)["binning_value"].append();
  binning_value_sig["return_type"] = "jitable";
  binning_value_sig["filter_name"] = "expr_jit_mesh_binning_value"; // still not a great name
  binning_value_sig["args/binning/type"] = "binning";
  binning_value_sig["args/binning/description"] =
      "The ``binning`` to lookup values in.";
  binning_value_sig["args/default_val/type"] = "scalar";
  binning_value_sig["args/default_val/optional"];
  binning_value_sig["args/default_val/description"] =
      "The value given to elements which do not fall into "
      "any of the bins. Defaults to ``0``.";
  binning_value_sig["args/topo/type"] = "topology";
  binning_value_sig["args/topo/optional"];
  binning_value_sig["args/topo/description"] =
      "The topology to bin. Defaults to the "
      "topology associated with the bin axes. This topology must have "
      "all the fields used for the axes of ``binning``. It only makes sense "
      "to specify this when the ``bin_axes`` are a subset of ``x``, ``y``, "
      "``z``.";
  binning_value_sig["args/assoc/type"] = "topology";
  binning_value_sig["args/assoc/optional"];
  binning_value_sig["args/assoc/description"] =
      "The association of the resultant field. Defaults to the association "
      "infered from the bin axes and and reduction variable. It only "
      "makes sense to specify this when the ``bin_axes`` are a subset of "
      "``x``, ``y``, ``z``.";
  binning_value_sig["description"] =
      "Get the value of a vertex or cell in a given binning. In other words, "
      "bin the cell and return the value found in that bin of ``binning``.";
  binning_value_sig["jitable"];

  //---------------------------------------------------------------------------

  count_params();
  //functions->save("functions.json", "json");
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
  histogram["clamp/type"] = "bool";

  conduit::Node &value_position = (*objects)["value_position/attrs"];
  value_position["value/type"] = "double";
  value_position["position/type"] = "vector";

  conduit::Node &topo = (*objects)["topology/attrs"];
  topo["cell/type"] = "cell";
  topo["cell/description"] = "Holds ``jitable`` cell attributes.";
  topo["vertex/type"] = "vertex";
  topo["vertex/description"] = "Holds ``jitable`` vertex attributes.";

  conduit::Node &cell = (*objects)["cell/attrs"];
  (*objects)["cell/jitable"];
  cell["x/type"] = "jitable";
  cell["x/description"] = "Cell x-coordinate.";
  cell["y/type"] = "jitable";
  cell["y/description"] = "Cell y-coordinate.";
  cell["z/type"] = "jitable";
  cell["z/description"] = "Cell z-coordinate.";
  cell["dx/type"] = "jitable";
  cell["dx/description"] = "Cell dx, only defined for rectilinear topologies.";
  cell["dy/type"] = "jitable";
  cell["dy/description"] = "Cell dy, only defined for rectilinear topologies.";
  cell["dz/type"] = "jitable";
  cell["dz/description"] = "Cell dz, only defined for rectilinear topologies.";
  cell["id/type"] = "jitable";
  cell["id/description"] = "Domain cell id.";
  cell["volume/type"] = "jitable";
  cell["volume/description"] = "Cell volume, only defined for 3D topologies";
  cell["area/type"] = "jitable";
  cell["area/description"] = "Cell area, only defined for 2D topologies";

  conduit::Node &vertex = (*objects)["vertex/attrs"];
  (*objects)["vertex/jitable"];
  vertex["x/type"] = "jitable";
  vertex["x/description"] = "Vertex x-coordinate.";
  vertex["y/type"] = "jitable";
  vertex["y/description"] = "Vertex y-coordinate.";
  vertex["z/type"] = "jitable";
  vertex["z/description"] = "Vertex z-coordinate.";
  vertex["id/type"] = "jitable";
  vertex["id/description"] = "Domain vertex id.";

  conduit::Node &aabb = (*objects)["aabb/attrs"];
  aabb["min/type"] = "vector";
  vertex["min/description"] = "Min coordinate of an axis-aligned bounding box (aabb)";
  aabb["max/type"] = "vector";
  vertex["max/description"] = "Max coordinate of an axis-aligned bounding box (aabb)";

  conduit::Node &vector_atts = (*objects)["vector/attrs"];
  vector_atts["x/type"] = "double";
  vector_atts["y/type"] = "double";
  vector_atts["z/type"] = "double";

  conduit::Node &bin_atts = (*objects)["bin/attrs"];
  bin_atts["min/type"] = "double";
  bin_atts["max/type"] = "double";
  bin_atts["center/type"] = "double";
  bin_atts["value/type"] = "double";

  conduit::Node &jitable = (*objects)["jitable/attrs"];
  jitable["x/type"] = "jitable";
  jitable["y/type"] = "jitable";
  jitable["z/type"] = "jitable";

  // we give field the attributes of jitable since all fields are jitables
  (*objects)["field/attrs"].update(jitable);

  //objects->save("objects.json", "json");
}

conduit::Node
ExpressionEval::evaluate(const std::string expr, std::string expr_name)
{
  ASCENT_DATA_OPEN("expression_eval");
  ASCENT_DATA_ADD("expression", expr);
  flow::Timer expression_timer;
  if(expr_name == "")
  {
    expr_name = expr;
  }

  // stores temporary fields, topos, and coords that need to be removed after
  // the expression runs
  conduit::Node remove;
  w.registry().add<conduit::Node>("remove", &remove, -1);

  w.registry().add<DataObject>("dataset", &m_data_object, -1);
  w.registry().add<conduit::Node>("cache", &m_cache.m_data, -1);
  w.registry().add<conduit::Node>("function_table", &g_function_table, -1);
  w.registry().add<conduit::Node>("object_table", &g_object_table, -1);
  int cycle = get_state_var(*m_data_object.as_node().get(), "cycle").to_int32();
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

  ASTNode *root_node = get_result();

  conduit::Node root;
  conduit::Node symbol_table;

  try
  {
    flow::Timer build_graph_timer;
    // change the execution policy here
    // change false to true to generate a graph with verbose names
    BuildGraphVisitor build_graph(
        w, std::make_shared<const FusePolicy>(), false);
    // BuildGraphVisitor build_graph(
    //     w, std::make_shared<const RoundtripPolicy>(), false);
    root_node->accept(&build_graph);
    root = build_graph.get_output();

    symbol_table = build_graph.table();
    w.registry().add<conduit::Node>("symbol_table", &symbol_table, -1);
    // if root is a derived field add a JitFilter to execute it
    if(root["type"].as_string() == "jitable")
    {
      jit_root(root, expr_name);
    }

    //w.graph().save_dot_html("ascent_expressions_graph.html");
    ASCENT_DATA_ADD("build_graph time", build_graph_timer.elapsed());
    flow::Timer execute_timer;
    w.execute();

    ASCENT_DATA_ADD("execute time", execute_timer.elapsed());
  }
  catch(std::exception &e)
  {
    delete root_node;
    w.reset();
    ASCENT_ERROR("Error while executing expression '" << expr
                                                      << "': " << e.what());
  }
  std::string filter_name = root["filter_name"].as_string();

  conduit::Node *n_res = w.registry().fetch<conduit::Node>(filter_name);
  conduit::Node return_val = *n_res;

  //return_val.print();


  // remove temporary fields, topologies, and coordsets from the dataset
  // TODO: We need a way to delete the intermediate results during execution
  conduit::Node *dataset = m_data_object.as_node().get();
  const int num_domains = dataset->number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = dataset->child(i);
    for(const auto &field_name : remove["fields"].child_names())
    {
      dom["fields"].remove(field_name);
    }
    for(const auto &topo_name : remove["topologies"].child_names())
    {
      dom["topologies"].remove(topo_name);
    }
    for(const auto &coords_name : remove["coordsets"].child_names())
    {
      dom["coordsets"].remove(coords_name);
    }
  }

  //std::cout<<m_data_object.as_node()->to_summary_string()<<"\n";

  // add the sim time
  conduit::Node n_time = get_state_var(*m_data_object.as_node().get(), "time");
  double time = 0;
  bool valid_time = false;
  if(!n_time.dtype().is_empty())
  {
    valid_time = true;
    time = n_time.to_float64();
  }
  return_val["time"] = time;

  // check the cache for signs of time travel
  // i.e., someone could have restarted the simulation from the beginning
  // or from some earlier checkpoint
  // There are a couple conditions:
  // 0) only check one time on startup
  // 1) only filter if we haven't done so before
  // 2) only filter if we detect time travel
  // 3) only filter if we have state/time
  static bool first_execute = true;

  if(first_execute &&
     !m_cache.filtered() &&
     time <= m_cache.last_known_time() &&
     valid_time)
  {
    // remove all cache entries that occur in the future
    m_cache.filter_time(time);
  }
  first_execute = false;

  m_cache.last_known_time(time);

  //return_val.print();
  // add the result to the cache
  {
    std::stringstream cache_entry;
    cache_entry << expr_name << "/" << cycle;
    m_cache.m_data[cache_entry.str()] = return_val;
  }
  // now we might have intermediate symbol, and
  // we also need to add them to the cache
  const int num_symbols = symbol_table.number_of_children();
  for(int i = 0; i < num_symbols; ++i)
  {
    const conduit::Node &symbol = symbol_table.child(i);
    if(symbol.has_path("value"))
    {
      const std::string symbol_name = symbol.name();
      std::stringstream cache_entry;
      cache_entry << symbol_name << "/" << cycle;
      m_cache.m_data[cache_entry.str()] = symbol;
    }
  }

  delete root_node;
  w.reset();
#ifdef ASCENT_JIT_ENABLED
  ASCENT_DATA_ADD("Device high water mark", ArrayRegistry::high_water_mark());
  ASCENT_DATA_ADD("Current Device usage ", ArrayRegistry::device_usage());
  ASCENT_DATA_ADD("Current host usage ", ArrayRegistry::host_usage());
  ArrayRegistry::reset_high_water_mark();
#endif
  ASCENT_DATA_CLOSE();
  return return_val;
}

void ExpressionEval::jit_root(conduit::Node &root, const std::string &expr_name)
{
  // When the root node in the executiuon graph is a jittable
  // result, we have to complile that kernel and execute it
  if(root["type"].as_string() == "jitable")
  {
    conduit::Node params;
    params["func"] = "execute";
    params["filter_name"] = "jit_execute";
    params["field_name"] = expr_name;
    conduit::Node &inp = params["inputs/jitable"];
    inp = root;
    inp["port"] = 0;
    w.graph().add_filter(
        register_jit_filter(
            w, 1, std::make_shared<const AlwaysExecutePolicy>()),
        "jit_execute",
        params);
    // src, dest, port
    w.graph().connect(root["filter_name"].as_string(), "jit_execute", 0);
    root["filter_name"] = "jit_execute";
    root["type"] = "field";
  }
}
//-----------------------------------------------------------------------------
const conduit::Node &
ExpressionEval::get_cache()
{
  return m_cache.m_data;
}

void
ExpressionEval::reset_cache()
{
  m_cache.m_data.reset();
}

void
ExpressionEval::save_cache(const std::string &filename)
{
  m_cache.save(filename);
}

void
ExpressionEval::save_cache()
{
  m_cache.save();
}

void ExpressionEval::get_last(conduit::Node &data)
{
  data.reset();
  const int entries = m_cache.m_data.number_of_children();

  for(int i = 0; i < entries; ++i)
  {
    conduit::Node &entry = m_cache.m_data.child(i);
    const int cycles = entry.number_of_children();
    if(cycles > 0)
    {
      conduit::Node &cycle = entry.child(cycles - 1);
      data[cycle.path()].set_external(cycle);
    }
  }
}
void ExpressionEval::save_cache(const std::string &filename,
                                const std::vector<std::string> &selection)
{
  m_cache.save(filename, selection);
}
//-----------------------------------------------------------------------------
DataObject& ExpressionEval::data_object()
{
  return m_data_object;
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

