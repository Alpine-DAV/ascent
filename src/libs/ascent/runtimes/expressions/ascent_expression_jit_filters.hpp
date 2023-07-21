//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_EXPRESSION_JIT_FILTERS
#define ASCENT_EXPRESSION_JIT_FILTERS

#include "ascent_derived_jit.hpp"
#include <flow_workspace.hpp>
#include <flow_filter.hpp>


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

//-----------------------------------------------------------------------------
class ExprJitFilter : public ::flow::Filter
{
public:
  ExprJitFilter(const int num_inputs,
            const std::shared_ptr<const JitExecutionPolicy> exec_policy);
  ~ExprJitFilter();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();

private:
  int num_inputs;
  const std::shared_ptr<const JitExecutionPolicy> exec_policy;
};

//-----------------------------------------------------------------------------
// TODO: IS THIS JUST FOR JIT,  OR GENERAL SUPPORT?
//-----------------------------------------------------------------------------
class ExprExpressionList : public ::flow::Filter
{
public:
  ExprExpressionList();
  ExprExpressionList(int num_inputs);
  ~ExprExpressionList();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();

protected:
  int m_num_inputs;

};

//-----------------------------------------------------------------------------
// register a JitFilter with the correct number of inputs and execution policy
// or return its type_name if it exists
std::string register_jit_filter(
    flow::Workspace &w,
    const int num_inputs,
    const std::shared_ptr<const JitExecutionPolicy> exec_policy);

//-----------------------------------------------------------------------------
std::string register_expression_list_filter(flow::Workspace &w,
                                            const int num_inputs);
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

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
