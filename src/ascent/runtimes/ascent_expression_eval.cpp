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
#include "ascent_data_logger.hpp"
#include "expressions/ascent_array_registry.hpp"
#include "expressions/ascent_blueprint_architect.hpp"
#include "expressions/ascent_derived_jit.hpp"
#include "expressions/ascent_expression_filters.hpp"
#include "expressions/ascent_expressions_ast.hpp"
#include "expressions/ascent_expressions_parser.hpp"
#include "expressions/ascent_expressions_tokens.hpp"

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

  std::string file_name = session + ".yaml";
  std::string session_file = conduit::utils::join_path(dir, file_name);
  m_session_file = session_file;

  bool exists = conduit::utils::is_file(session_file);

  if(m_rank == 0 && exists)
  {
    m_data.load(session_file, "yaml");
  }

#ifdef ASCENT_MPI_ENABLED
  if(exists)
  {
    conduit::relay::mpi::broadcast_using_schema(m_data, 0, mpi_comm);
  }
#endif
  m_loaded = true;
}

Cache::~Cache()
{
  // the session file can be blank during testing,
  // since its not actually opening ascent
  if(m_rank == 0 && !m_data.dtype().is_empty() && m_session_file != "")
  {
    m_data.save(m_session_file, "yaml");
  }
}

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
  flow::Workspace::register_filter_type<expressions::Topo>();
  flow::Workspace::register_filter_type<expressions::Axis>();
  flow::Workspace::register_filter_type<expressions::Histogram>();
  flow::Workspace::register_filter_type<expressions::Binning>();
  flow::Workspace::register_filter_type<expressions::PaintBinning>();
  flow::Workspace::register_filter_type<expressions::BinningMesh>();
  flow::Workspace::register_filter_type<expressions::Entropy>();
  flow::Workspace::register_filter_type<expressions::Pdf>();
  flow::Workspace::register_filter_type<expressions::Cdf>();
  flow::Workspace::register_filter_type<expressions::Quantile>();
  flow::Workspace::register_filter_type<expressions::BinByValue>();
  flow::Workspace::register_filter_type<expressions::BinByIndex>();
  flow::Workspace::register_filter_type<expressions::Cycle>();
  flow::Workspace::register_filter_type<expressions::ArrayAccess>();
  flow::Workspace::register_filter_type<expressions::DotAccess>();
  flow::Workspace::register_filter_type<expressions::PointAndAxis>();
  flow::Workspace::register_filter_type<expressions::MaxFromPoint>();
  flow::Workspace::register_filter_type<expressions::Bin>();

  initialize_functions();
  initialize_objects();
}

ExpressionEval::ExpressionEval(conduit::Node *data) : m_data(data)
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

  //---------------------------------------------------------------------------

  conduit::Node &array_avg_sig = (*functions)["avg"].append();
  array_avg_sig["return_type"] = "double";
  array_avg_sig["filter_name"] = "array_avg"; // matches the filter's type_name
  array_avg_sig["args/arg1/type"] = "array"; // arg names match input port names
  array_avg_sig["description"] = "Return the average of an array.";

  //---------------------------------------------------------------------------

  conduit::Node &field_avg_sig = (*functions)["avg"].append();
  field_avg_sig["return_type"] = "double";
  field_avg_sig["filter_name"] = "field_avg";
  field_avg_sig["args/arg1/type"] = "field";
  field_avg_sig["description"] = "Return the field average of a mesh variable.";

  //---------------------------------------------------------------------------

  conduit::Node &field_nan_sig = (*functions)["field_nan_count"].append();
  field_nan_sig["return_type"] = "double";
  field_nan_sig["filter_name"] = "field_nan_count";
  field_nan_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_nan_sig["description"] =
      "Return the number  of NaNs in a mesh variable.";

  //---------------------------------------------------------------------------

  conduit::Node &field_inf_sig = (*functions)["field_inf_count"].append();
  field_inf_sig["return_type"] = "double";
  field_inf_sig["filter_name"] = "field_inf_count";
  field_inf_sig["args/arg1/type"] = "field"; // arg names match input port names
  field_inf_sig["description"] =
      "Return the number  of -inf and +inf in a mesh variable.";

  //---------------------------------------------------------------------------

  conduit::Node &scalar_max_sig = (*functions)["max"].append();
  scalar_max_sig["return_type"] = "double";
  scalar_max_sig["filter_name"] = "scalar_max";
  scalar_max_sig["args/arg1/type"] = "scalar";
  scalar_max_sig["args/arg2/type"] = "scalar";
  scalar_max_sig["description"] = "Return the maximum of two scalars.";

  //---------------------------------------------------------------------------

  conduit::Node &field_max_sig = (*functions)["max"].append();
  field_max_sig["return_type"] = "value_position";
  field_max_sig["filter_name"] = "field_max";
  field_max_sig["args/arg1/type"] = "field";
  field_max_sig["description"] =
      "Return the maximum value from the meshvar. Its position is also "
      "stored "
      "and is accessible via the `position` function.";

  //---------------------------------------------------------------------------

  conduit::Node &array_max_sig = (*functions)["max"].append();
  array_max_sig["return_type"] = "double";
  array_max_sig["filter_name"] = "array_max";
  array_max_sig["args/arg1/type"] = "array";
  array_max_sig["description"] = "Return the maximum of an array.";

  //---------------------------------------------------------------------------

  conduit::Node &field_min_sig = (*functions)["min"].append();
  field_min_sig["return_type"] = "value_position";
  field_min_sig["filter_name"] = "field_min";
  field_min_sig["args/arg1/type"] = "field";
  field_min_sig["description"] =
      "Return the minimum value from the meshvar. Its position is also "
      "stored "
      "and is accessible via the `position` function.";

  //---------------------------------------------------------------------------

  conduit::Node &scalar_min_sig = (*functions)["min"].append();
  scalar_min_sig["return_type"] = "double";
  scalar_min_sig["filter_name"] = "scalar_min";
  scalar_min_sig["args/arg1/type"] = "scalar";
  scalar_min_sig["args/arg2/type"] = "scalar";
  scalar_min_sig["description"] = "Return the minimum of two scalars.";

  //---------------------------------------------------------------------------

  conduit::Node &array_min_sig = (*functions)["min"].append();
  array_min_sig["return_type"] = "double";
  array_min_sig["filter_name"] = "array_min";
  array_min_sig["args/arg1/type"] = "array";
  array_min_sig["description"] = "Return the minimum of an array.";

  //---------------------------------------------------------------------------

  conduit::Node &field_sum_sig = (*functions)["sum"].append();
  field_sum_sig["return_type"] = "double";
  field_sum_sig["filter_name"] = "field_sum";
  field_sum_sig["args/arg1/type"] = "field";
  field_sum_sig["description"] = "Return the sum of a field.";

  //---------------------------------------------------------------------------

  conduit::Node &array_sum_sig = (*functions)["sum"].append();
  array_sum_sig["return_type"] = "double";
  array_sum_sig["filter_name"] = "array_sum";
  array_sum_sig["args/arg1/type"] = "array";
  array_sum_sig["description"] = "Return the sum of an array.";

  //---------------------------------------------------------------------------

  conduit::Node &cycle_sig = (*functions)["cycle"].append();
  cycle_sig["return_type"] = "int";
  cycle_sig["filter_name"] = "cycle";
  cycle_sig["args"] = conduit::DataType::empty();
  cycle_sig["description"] = "Return the current simulation cycle.";

  //---------------------------------------------------------------------------

  conduit::Node &vector = (*functions)["vector"].append();
  vector["return_type"] = "vector";
  vector["filter_name"] = "vector";
  vector["args/arg1/type"] = "scalar";
  vector["args/arg2/type"] = "scalar";
  vector["args/arg3/type"] = "scalar";
  vector["description"] = "Return the 3D position vector for the input value.";

  //---------------------------------------------------------------------------

  conduit::Node &mag_sig = (*functions)["magnitude"].append();
  mag_sig["return_type"] = "double";
  mag_sig["filter_name"] = "magnitude";
  mag_sig["args/arg1/type"] = "vector";
  mag_sig["description"] = "Return the magnitude of the input vector.";

  //---------------------------------------------------------------------------

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

  //---------------------------------------------------------------------------

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

  //---------------------------------------------------------------------------

  conduit::Node &entropy_sig = (*functions)["entropy"].append();
  entropy_sig["return_type"] = "double";
  entropy_sig["filter_name"] = "entropy";
  entropy_sig["args/hist/type"] = "histogram";
  entropy_sig["description"] =
      "Return the Shannon entropy given a histogram of the field.";

  //---------------------------------------------------------------------------

  conduit::Node &pdf_sig = (*functions)["pdf"].append();
  pdf_sig["return_type"] = "histogram";
  pdf_sig["filter_name"] = "pdf";
  pdf_sig["args/hist/type"] = "histogram";
  pdf_sig["description"] =
      "Return the probability distribution function (pdf) from a histogram.";

  //---------------------------------------------------------------------------

  conduit::Node &cdf_sig = (*functions)["cdf"].append();
  cdf_sig["return_type"] = "histogram";
  cdf_sig["filter_name"] = "cdf";
  cdf_sig["args/hist/type"] = "histogram";
  cdf_sig["description"] =
      "Return the cumulative distribution function (cdf) from a histogram.";

  //---------------------------------------------------------------------------

  // gets histogram bin by index
  conduit::Node &bin_by_index_sig = (*functions)["bin"].append();
  bin_by_index_sig["return_type"] = "double";
  bin_by_index_sig["filter_name"] = "bin_by_index";
  bin_by_index_sig["args/hist/type"] = "histogram";
  bin_by_index_sig["args/bin/type"] = "int";
  bin_by_index_sig["description"] =
      "Return the value of the bin at index `bin` of a histogram.";

  //---------------------------------------------------------------------------

  // gets histogram bin by value
  conduit::Node &bin_by_value_sig = (*functions)["bin"].append();
  bin_by_value_sig["return_type"] = "double";
  bin_by_value_sig["filter_name"] = "bin_by_value";
  bin_by_value_sig["args/hist/type"] = "histogram";
  bin_by_value_sig["args/val/type"] = "scalar";
  bin_by_value_sig["description"] =
      "Return the value of the bin with axis-value `val` on the histogram.";

  //---------------------------------------------------------------------------

  conduit::Node &field_sig = (*functions)["field"].append();
  field_sig["return_type"] = "field";
  field_sig["filter_name"] = "field";
  field_sig["args/field_name/type"] = "string";
  field_sig["args/component/type"] = "string";
  field_sig["args/component/optional"];
  field_sig["args/component/description"] =
      "Used to specify a single component if the field is a vector field.";
  field_sig["description"] = "Return a mesh field given a its name.";

  //---------------------------------------------------------------------------

  conduit::Node &topo_sig = (*functions)["topo"].append();
  topo_sig["return_type"] = "topo";
  topo_sig["filter_name"] = "topo";
  topo_sig["args/arg1/type"] = "string";
  topo_sig["description"] = "Return a mesh topology given a its name.";

  //---------------------------------------------------------------------------

  conduit::Node &point_and_axis_sig = (*functions)["point_and_axis"].append();
  point_and_axis_sig["return_type"] = "bin";
  point_and_axis_sig["filter_name"] = "point_and_axis";
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
  bin_sig["filter_name"] = "bin";
  bin_sig["args/binning/type"] = "binning";
  bin_sig["args/index/type"] = "int";
  bin_sig["description"] = "returns a bin from a binning by index";

  // -------------------------------------------------------------

  conduit::Node &max_from_point_sig = (*functions)["max_from_point"].append();
  max_from_point_sig["return_type"] = "value_position";
  max_from_point_sig["filter_name"] = "max_from_point";
  max_from_point_sig["args/binning/type"] = "binning";
  max_from_point_sig["args/axis/type"] = "string";
  max_from_point_sig["args/point/type"] = "double";
  max_from_point_sig["description"] =
      "returns the closest max"
      " value from a reference point on an axis";

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

  //---------------------------------------------------------------------------

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

  //---------------------------------------------------------------------------

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
  binning_sig["args/component/type"] = "string";
  binning_sig["args/component/optional"];
  binning_sig["args/component/description"] =
      "the component of a vector field to use for the reduction."
      " Example 'x' for a field defined as 'velocity/x'";
  binning_sig["description"] = "Returns a multidimensional data binning.";

  // TODO for now this does not jit
  conduit::Node &paint_binning_sig = (*functions)["paint_binning"].append();
  paint_binning_sig["return_type"] = "field";
  paint_binning_sig["filter_name"] = "paint_binning";
  paint_binning_sig["args/binning/type"] = "binning";
  paint_binning_sig["args/binning/description"] =
      "The values in ``binning`` are used to generate the new field.";
  paint_binning_sig["args/name/type"] = "string";
  paint_binning_sig["args/name/optional"];
  paint_binning_sig["args/name/description"] =
      "The name of the new field to be generated. If not specified, a name "
      "is "
      "automatically generated and the field is treated as a temporary and "
      "removed from the dataset when the expression is done executing.";
  paint_binning_sig["args/topo/type"] = "topo";
  paint_binning_sig["args/topo/optional"];
  paint_binning_sig["args/topo/description"] =
      " The topology to paint the bin values back onto. Defaults to the "
      "topology associated with the bin axes. This topology must have "
      "all the fields used for the axes of ``binning``. It only makes sense "
      "to "
      "specify this when the only bin axes are a subset of ``x``, ``y``, "
      "``z``. Additionally, it must be specified in this case since there is "
      "not enough info to infer the topology assuming there are multiple "
      "topologies in the dataset.";
  paint_binning_sig["args/assoc/type"] = "topo";
  paint_binning_sig["args/assoc/optional"];
  paint_binning_sig["args/assoc/description"] =
      "Defaults to the association infered from the bin axes and and "
      "reduction "
      "variable. The topology to paint the bin values back onto. This "
      "topology "
      "must have all the fields used for the axes of ``binning``. It only "
      "makes sense to specify this when the only bin axes are a subset of "
      "``x``, ``y``, ``z``.";
  paint_binning_sig["args/default_value/type"] = "scalar";
  paint_binning_sig["args/default_value/optional"];
  paint_binning_sig["args/default_value/description"] =
      "Defaults to ``0``. The value given to elements which do not fall into "
      "any of the bins.";
  paint_binning_sig["description"] =
      "Paints back the bin values onto an existing mesh by binning the "
      "elements of the mesh and creating a new field there the value at each "
      "element is the value in the bin it falls into.";

  conduit::Node &binning_mesh_sig = (*functions)["binning_mesh"].append();
  binning_mesh_sig["return_type"] = "field";
  binning_mesh_sig["filter_name"] = "binning_mesh";
  binning_mesh_sig["args/binning/type"] = "binning";
  binning_mesh_sig["args/binning/description"] =
      "The values in ``binning`` are used to generate the new field.";
  binning_mesh_sig["args/name/type"] = "string";
  binning_mesh_sig["args/name/optional"];
  binning_mesh_sig["args/name/description"] =
      "The name of the new field to be generated, the corresponding topology "
      "topology and coordinate sets will be named '``name``_topo' and "
      "``name``_coords' respectively. If not specified, a name is "
      "automatically generated and the field is treated as a temporary and "
      "removed from the dataset when the expression is done executing.";
  binning_mesh_sig["description"] =
      "A binning with 3 or fewer dimensions will be output as a new element "
      "associated field on a new topology on the dataset. This is useful for "
      "directly visualizing the binning.";

  //---------------------------------------------------------------------------
  // Jitable Functions
  //---------------------------------------------------------------------------
  // Functions below this line call JitFilter
  // All jitable functions need to have ["jitable"] in order for this to happen
  // filter_name is passed to JitFilter so that it can determine which function
  // to execute

  conduit::Node &field_scalar_max_sig = (*functions)["max"].append();
  field_scalar_max_sig["return_type"] = "jitable";
  field_scalar_max_sig["filter_name"] = "field_field_max";
  field_scalar_max_sig["args/arg1/type"] = "field";
  field_scalar_max_sig["args/arg2/type"] = "scalar";
  field_scalar_max_sig["jitable"];
  field_scalar_max_sig["description"] =
      "Return a derived field that is the max of two fields.";

  //---------------------------------------------------------------------------

  conduit::Node &field_field_min_sig = (*functions)["min"].append();
  field_field_min_sig["return_type"] = "jitable";
  field_field_min_sig["filter_name"] = "field_field_min";
  field_field_min_sig["args/arg1/type"] = "field";
  field_field_min_sig["args/arg2/type"] = "field";
  field_field_min_sig["jitable"];
  field_field_min_sig["description"] =
      "Return a derived field that is the min of two fields.";

  //---------------------------------------------------------------------------

  conduit::Node &field_sin_sig = (*functions)["sin"].append();
  field_sin_sig["return_type"] = "jitable";
  field_sin_sig["filter_name"] = "field_sin";
  field_sin_sig["args/arg1/type"] = "field";
  field_sin_sig["jitable"];
  field_sin_sig["description"] =
      "Return a derived field that is the sin of a field.";

  //---------------------------------------------------------------------------

  conduit::Node &field_abs_sig = (*functions)["abs"].append();
  field_abs_sig["return_type"] = "jitable";
  field_abs_sig["filter_name"] = "field_abs";
  field_abs_sig["args/arg1/type"] = "field";
  field_abs_sig["jitable"];
  field_abs_sig["description"] =
      "Return a derived field that is the absolute value of a field.";

  //---------------------------------------------------------------------------

  conduit::Node &field_sqrt_sig = (*functions)["sqrt"].append();
  field_sqrt_sig["return_type"] = "jitable";
  field_sqrt_sig["filter_name"] = "field_sqrt";
  field_sqrt_sig["args/arg1/type"] = "field";
  field_sqrt_sig["jitable"];
  field_sqrt_sig["description"] =
      "Return a derived field that is the square root value of a field.";

  //---------------------------------------------------------------------------

  conduit::Node &field_gradient_sig = (*functions)["gradient"].append();
  field_gradient_sig["return_type"] = "jitable";
  field_gradient_sig["filter_name"] = "gradient";
  field_gradient_sig["args/field/type"] = "field";
  field_gradient_sig["jitable"];
  field_gradient_sig["description"] =
      "Return a derived field that is the gradient of a field.";

  //---------------------------------------------------------------------------

  conduit::Node &field_vorticity_sig = (*functions)["vorticity"].append();
  field_vorticity_sig["return_type"] = "jitable";
  field_vorticity_sig["filter_name"] = "vorticity";
  field_vorticity_sig["args/field/type"] = "field";
  field_vorticity_sig["jitable"];
  field_vorticity_sig["description"] =
      "Return a derived field that is the vorticity of a vector field.";

  //---------------------------------------------------------------------------

  conduit::Node &field_magnitude_sig = (*functions)["magnitude"].append();
  field_magnitude_sig["return_type"] = "jitable";
  field_magnitude_sig["filter_name"] = "magnitude";
  field_magnitude_sig["args/vector/type"] = "field";
  field_magnitude_sig["jitable"];
  field_magnitude_sig["description"] =
      "Return a derived field that is the magnitude of a vector field.";

  //---------------------------------------------------------------------------

  conduit::Node &field_vector = (*functions)["vector"].append();
  field_vector["return_type"] = "jitable";
  field_vector["filter_name"] = "vector";
  field_vector["args/arg1/type"] = "field";
  field_vector["args/arg2/type"] = "field";
  field_vector["args/arg3/type"] = "field";
  field_vector["jitable"];
  field_vector["description"] = "Return a vector field on the mesh.";

  //---------------------------------------------------------------------------

  conduit::Node &derived_field = (*functions)["derived_field"].append();
  derived_field["return_type"] = "jitable";
  derived_field["filter_name"] = "derived_field";
  derived_field["args/arg1/type"] = "scalar";
  derived_field["args/arg1/description"] =
      "The scalar to be cast to a derived field.";
  derived_field["args/topology/type"] = "string";
  derived_field["args/topology/optional"];
  derived_field["args/topology/description"] =
      "The topology to put the derived field onto. The language tries to infer "
      "this if not specified.";
  derived_field["args/association/type"] = "string";
  derived_field["args/association/optional"];
  derived_field["args/association/description"] =
      "The association of the derived field. The language will try to infer "
      "this if not specified.";
  derived_field["jitable"];
  derived_field["description"] =
      "Cast a scalar to a derived field (type `jitable`).";

  //---------------------------------------------------------------------------

  conduit::Node &derived_field2 = (*functions)["derived_field"].append();
  derived_field2["return_type"] = "jitable";
  derived_field2["filter_name"] = "derived_field";
  derived_field2["args/arg1/type"] = "field";
  derived_field2["args/arg1/description"] =
      "The scalar to be cast to a derived field.";
  derived_field2["args/topology/type"] = "string";
  derived_field2["args/topology/optional"];
  derived_field2["args/topology/description"] =
      "The topology to put the derived field onto. The language tries to infer "
      "this if not specified.";
  derived_field2["args/association/type"] = "string";
  derived_field2["args/association/optional"];
  derived_field2["args/association/description"] =
      "The association of the derived field. The language will try to infer "
      "this if not specified.";
  derived_field2["jitable"];
  derived_field2["description"] =
      "Used to explicitly specfiy the topology and association of a derived "
      "field (e.g. in case it cannot be inferred).";

  count_params();
  // functions->save("functions.json", "json");
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

  conduit::Node &topo = (*objects)["topo/attrs"];
  topo["cell/type"] = "cell";
  topo["vertex/type"] = "vertex";

  conduit::Node &cell = (*objects)["cell/attrs"];
  (*objects)["cell/jitable"];
  cell["x/type"] = "jitable";
  cell["y/type"] = "jitable";
  cell["z/type"] = "jitable";
  cell["dx/type"] = "jitable";
  cell["dy/type"] = "jitable";
  cell["dz/type"] = "jitable";
  cell["id/type"] = "jitable";
  cell["volume/type"] = "jitable";
  cell["area/type"] = "jitable";

  conduit::Node &vertex = (*objects)["vertex/attrs"];
  (*objects)["vertex/jitable"];
  vertex["x/type"] = "jitable";
  vertex["y/type"] = "jitable";
  vertex["z/type"] = "jitable";
  vertex["id/type"] = "jitable";

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

  // objects->save("objects.json", "json");
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

  w.registry().add<conduit::Node>("dataset", m_data, -1);
  w.registry().add<conduit::Node>("cache", &m_cache.m_data, -1);
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
    flow::Timer build_graph_timer;
    BuildGraphVisitor build_graph(w, false);
    expression->accept(&build_graph);
    root = build_graph.get_output();
    // if root is a derived field add a JitFilter to execute it
    if(root["type"].as_string() == "jitable")
    {
      conduit::Node params;
      params["func"] = "execute";
      params["filter_name"] = "jit_execute";
      params["execute"] = true;
      params["field_name"] = expr_name;
      conduit::Node &inp = params["inputs/jitable"];
      inp = root;
      inp["port"] = 0;
      w.graph().add_filter(register_jit_filter(w, 1), "jit_execute", params);
      // src, dest, port
      w.graph().connect(root["filter_name"].as_string(), "jit_execute", 0);
      root["filter_name"] = "jit_execute";
      root["type"] = "field";
    }
    // w.graph().save_dot_html("ascent_expressions_graph.html");
    ASCENT_DATA_ADD("build_graph time", build_graph_timer.elapsed());
    flow::Timer execute_timer;
    w.execute();
    ASCENT_DATA_ADD("execute time", execute_timer.elapsed());
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

  // remove temporary fields, topologies, and coordsets from the dataset
  const int num_domains = m_data->number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = m_data->child(i);
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

  // add the sim time
  conduit::Node n_time = get_state_var(*m_data, "time");
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

  if(first_execute && !m_cache.filtered() &&
     time <= m_cache.last_known_time() && valid_time)
  {
    // remove all cache entries that occur in the future
    m_cache.filter_time(time);
  }
  first_execute = false;

  m_cache.last_known_time(time);

  std::stringstream cache_entry;
  cache_entry << expr_name << "/" << cycle;

  m_cache.m_data[cache_entry.str()] = return_val;

  delete expression;
  w.reset();
  ASCENT_DATA_ADD("Device high water mark", ArrayRegistry::high_water_mark());
  ASCENT_DATA_ADD("Current Device usage ", ArrayRegistry::device_usage());
  ASCENT_DATA_ADD("Current host usage ", ArrayRegistry::host_usage());
  ArrayRegistry::reset_high_water_mark();
  ASCENT_DATA_CLOSE();
  return return_val;
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
ExpressionEval::get_last(conduit::Node &data)
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
