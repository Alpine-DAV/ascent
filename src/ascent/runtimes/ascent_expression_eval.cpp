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
#include "flow_workspace.hpp"
#include "expressions/ascent_expression_filters.hpp"
#include "expressions/ast.hpp"
#include "expressions/parser.hpp"
#include "expressions/tokens.hpp"

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
extern ASTExpression *expression;
ExpressionEval::ExpressionEval(conduit::Node *data)
  : m_data(data)
{

}

conduit::Node
ExpressionEval::evaluate(const std::string expr)
{
  flow::Workspace w;

  flow::Workspace::register_filter_type<expressions::Double>();
  flow::Workspace::register_filter_type<expressions::Integer>();
  flow::Workspace::register_filter_type<expressions::BinaryOp>();
  flow::Workspace::register_filter_type<expressions::MeshVar>();
  flow::Workspace::register_filter_type<expressions::ScalarMax>();
  flow::Workspace::register_filter_type<expressions::FieldMax>();

  conduit::Node* functions = new conduit::Node();;

  conduit::Node &scalar_max_sig = (*functions)["max"].append();
  scalar_max_sig["return_type"] = "scalar";
  scalar_max_sig["filter_name"] = "scalar_max";
  scalar_max_sig["args/arg1/type"] = "scalar"; // arg names match input port names
  scalar_max_sig["args/arg2/type"] = "scalar";

  conduit::Node &field_max_sig = (*functions)["max"].append();
  field_max_sig["return_type"] = "scalar";
  field_max_sig["filter_name"] = "field_max";
  field_max_sig["args/arg1/type"] = "meshvar"; // arg names match input port names

  w.registry().add<conduit::Node>("function_table", functions, 1);

  w.registry().add<conduit::Node>("dataset", m_data, -1);



  scan_string(expr.c_str());
  ASTExpression *expression = get_result();

  std::cout<<"Expresion "<<expression<<"\n";
  expression->access();
  conduit::Node root = expression->build_graph(w);
  std::cout<<w.graph().to_dot()<<"\n";
  w.execute();
  std::cout<<"root node \n";
  root.print();
  conduit::Node *n_res = w.registry().fetch<conduit::Node>(root["filter_name"].as_string());

  if(n_res->dtype().is_floating_point())
  {
    std::cout<<"Result "<<n_res->as_float64()<<"\n";
  }
  else
  {
    std::cout<<"Result "<<n_res->as_int32()<<"\n";
  }

  conduit::Node res;
  delete expression;
  return res;
}

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
