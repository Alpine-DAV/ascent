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
/// file: ascent_runtime_expression_eval.hpp
///
//-----------------------------------------------------------------------------

#include "ascent_expression_eval.hpp"
#include "expressions/ascent_blueprint_architect.hpp"
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

extern ASTExpression *expression;

conduit::Node ExpressionEval::m_cache;
conduit::Node g_function_table;

void register_builtin()
{
  flow::Workspace::register_filter_type<expressions::NullArg>();
  flow::Workspace::register_filter_type<expressions::Double>();
  flow::Workspace::register_filter_type<expressions::Integer>();
  flow::Workspace::register_filter_type<expressions::Identifier>();
  flow::Workspace::register_filter_type<expressions::BinaryOp>();
  flow::Workspace::register_filter_type<expressions::MeshVar>();
  flow::Workspace::register_filter_type<expressions::ScalarMax>();
  flow::Workspace::register_filter_type<expressions::ScalarMin>();
  flow::Workspace::register_filter_type<expressions::FieldMax>();
  flow::Workspace::register_filter_type<expressions::FieldMin>();
  flow::Workspace::register_filter_type<expressions::FieldAvg>();
  flow::Workspace::register_filter_type<expressions::Position>();
  flow::Workspace::register_filter_type<expressions::Vector>();
  flow::Workspace::register_filter_type<expressions::Magnitude>();
  flow::Workspace::register_filter_type<expressions::Histogram>();
  flow::Workspace::register_filter_type<expressions::Cycle>();

  initialize_functions();
}

ExpressionEval::ExpressionEval(conduit::Node *data)
  : m_data(data)
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
            ASCENT_ERROR("Function: optional parameters must be after require params");
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
  conduit::Node* functions = &g_function_table;

  // -------------------------------------------------------------

  conduit::Node &field_avg_sig = (*functions)["avg"].append();
  field_avg_sig["return_type"] = "scalar";
  field_avg_sig["filter_name"] = "field_avg";
  field_avg_sig["args/arg1/type"] = "meshvar"; // arg names match input port names
  // -------------------------------------------------------------

  conduit::Node &scalar_max_sig = (*functions)["max"].append();
  scalar_max_sig["return_type"] = "scalar";
  scalar_max_sig["filter_name"] = "scalar_max";
  scalar_max_sig["args/arg1/type"] = "scalar"; // arg names match input port names
  scalar_max_sig["args/arg2/type"] = "scalar";

  // -------------------------------------------------------------

  conduit::Node &field_max_sig = (*functions)["max"].append();
  field_max_sig["return_type"] = "scalar";
  field_max_sig["filter_name"] = "field_max";
  field_max_sig["args/arg1/type"] = "meshvar"; // arg names match input port names

  // -------------------------------------------------------------

  conduit::Node &field_min_sig = (*functions)["min"].append();
  field_min_sig["return_type"] = "scalar";
  field_min_sig["filter_name"] = "field_min";
  field_min_sig["args/arg1/type"] = "meshvar"; // arg names match input port names

  // -------------------------------------------------------------

  conduit::Node &scalar_min_sig = (*functions)["min"].append();
  scalar_min_sig["return_type"] = "scalar";
  scalar_min_sig["filter_name"] = "scalar_min";
  scalar_min_sig["args/arg1/type"] = "scalar"; // arg names match input port names
  scalar_min_sig["args/arg2/type"] = "scalar";

  // -------------------------------------------------------------

  conduit::Node &pos_sig = (*functions)["position"].append();
  pos_sig["return_type"] = "vector";
  pos_sig["filter_name"] = "expr_position";
  pos_sig["args/arg1/type"] = "scalar"; // Should query result be differnt type?
  // -------------------------------------------------------------

  conduit::Node &cycle_sig = (*functions)["cycle"].append();
  cycle_sig["return_type"] = "scalar";
  cycle_sig["filter_name"] = "cycle";
  cycle_sig["args"] = conduit::DataType::empty();

  // -------------------------------------------------------------
  conduit::Node &vector = (*functions)["vector"].append();
  vector["return_type"] = "vector";
  vector["filter_name"] = "vector";
  vector["args/arg1/type"] = "scalar"; // arg names match input port names
  vector["args/arg2/type"] = "scalar";
  vector["args/arg3/type"] = "scalar";

  // -------------------------------------------------------------
  conduit::Node &mag_sig= (*functions)["magnitude"].append();
  mag_sig["return_type"] = "scalar";
  mag_sig["filter_name"] = "magnitude";
  mag_sig["args/arg1/type"] = "vector";

  // -------------------------------------------------------------
  conduit::Node &hist_sig= (*functions)["histogram"].append();
  hist_sig["return_type"] = "histogram";
  hist_sig["filter_name"] = "histogram";
  hist_sig["args/arg1/type"] = "meshvar";
  // In a flow filter, these become parameters
  hist_sig["args/num_bins/type"] = "scalar";
  hist_sig["args/num_bins/optional"];
  hist_sig["args/min_val/type"] = "scalar";
  hist_sig["args/min_val/optional"];
  hist_sig["args/max_val/type"] = "scalar";
  hist_sig["args/max_val/optional"];

  count_params();
  // TODO: validate thar there are no ambiguities
}

conduit::Node
ExpressionEval::evaluate(const std::string expr, std::string expr_name)
{

  if(expr_name == "")
  {
    expr_name = expr;
  }

  w.registry().add<conduit::Node>("dataset", m_data, -1);
  w.registry().add<conduit::Node>("cache", &m_cache, -1);
  w.registry().add<conduit::Node>("function_table", &g_function_table, -1);
  int cycle = get_state_var(*m_data, "cycle").to_int32();
  w.registry().add<int>("cycle", &cycle, -1);

  try
  {
    scan_string(expr.c_str());
  }
  catch(const char* msg)
  {
    w.reset();
    ASCENT_ERROR("Expression parsing error: "<<msg<<" in '"<<expr<<"'");
  }

  ASTExpression *expression = get_result();

  expression->access();
  conduit::Node root = expression->build_graph(w);

  //std::cout<<w.graph().to_dot()<<"\n";
  try
  {
    w.execute();
  }
  catch(std::exception &e)
  {
    w.reset();
    ASCENT_ERROR("Error while executing expression '"<<expr<<"': "<<e.what());
  }
  conduit::Node *n_res = w.registry().fetch<conduit::Node>(root["filter_name"].as_string());
  conduit::Node return_val = *n_res;
  delete expression;

  std::stringstream cache_entry;
  cache_entry<<expr_name<<"/"<<cycle;
  m_cache[cache_entry.str()] = *n_res;

  w.reset();
  return return_val;
}

const conduit::Node&
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
