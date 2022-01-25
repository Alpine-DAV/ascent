//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_jit_fusion.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_JIT_FUSION_HPP
#define ASCENT_JIT_FUSION_HPP

#include <unordered_map>
#include <string>
#include <conduit.hpp>
#include "ascent_derived_jit.hpp"

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

// handles kernel fusion for various functions
// calls TopologyCode, FieldCode, etc.
// i.e., this class knows how to combine kernels and generate jitable functions
class JitableFusion
{
public:
  JitableFusion(const conduit::Node &params,
                   const std::vector<const Jitable *> &input_jitables,
                   const std::vector<const Kernel *> &input_kernels,
                   const std::string &filter_name,
                   const conduit::Node &dataset,
                   const int dom_idx,
                   const bool not_fused,
                   Jitable &out_jitable,
                   Kernel &out_kernel);

  void binary_op();
  void builtin_functions(const std::string &function_name);
  void expr_dot();
  void expr_if();
  void derived_field();
  void gradient();
  void curl();
  void recenter();
  void magnitude();
  void vector();
  void binning_value(const conduit::Node &binning);
  void rand();

private:
  void topo_attrs(const conduit::Node &obj, const std::string &name);
  void gradient(const int field_port, const int component);
  void temporary_field(const Kernel &field_kernel,
                       const std::string &field_name);
  std::string possible_temporary(const int field_port);

  const conduit::Node &params;
  const std::vector<const Jitable *> &input_jitables;
  const std::vector<const Kernel *> &input_kernels;
  const std::string &filter_name;
  const conduit::Node &dataset;
  const int dom_idx;
  const bool not_fused;
  Jitable &out_jitable;
  Kernel &out_kernel;
  const conduit::Node &inputs;
  const conduit::Node &domain;
};

//-----------------------------------------------------------------------------
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
