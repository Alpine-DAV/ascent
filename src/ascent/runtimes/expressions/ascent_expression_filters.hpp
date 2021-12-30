//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_expression_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_EXPRESSION_FILTERS
#define ASCENT_EXPRESSION_FILTERS

#include <ascent.hpp>
#include <flow_filter.hpp>
#include <flow_graph.hpp>

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

// for multi line statements, we have intermediate symbols:
// bananas = max(field('pressure'))
// bananas < 0
// We want to keep the intermediate results around (e.g., the value of
// bananas) to put in the cache and for debugging,
// but we don't know what the value is until we execute the graph.
// Each filter will call this function to update the return values.
void resolve_symbol_result(flow::Graph &graph,
                           const conduit::Node *output,
                           const std::string filter_name);

// Need to validate the binning input in several places
// so consolidate this call
void binning_interface(const std::string &reduction_var,
                       const std::string &reduction_op,
                       const conduit::Node &n_empty_bin_val,
                       const conduit::Node &n_component,
                       const conduit::Node &n_axis_list,
                       conduit::Node &dataset,
                       conduit::Node &n_binning,
                       conduit::Node &n_output_axes);

//-----------------------------------------------------------------------------
///
/// Filters for expressions
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class NullArg : public ::flow::Filter
{
public:
  NullArg();
  ~NullArg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Identifier : public ::flow::Filter
{
public:
  Identifier();
  ~Identifier();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class History : public ::flow::Filter
{
public:
  History();
  ~History();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class HistoryRange : public ::flow::Filter
{
public:
  HistoryRange();
  ~HistoryRange();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Boolean : public ::flow::Filter
{
public:
  Boolean();
  ~Boolean();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Integer : public ::flow::Filter
{
public:
  Integer();
  ~Integer();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Double : public ::flow::Filter
{
public:
  Double();
  ~Double();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class String : public ::flow::Filter
{
public:
  String();
  ~String();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class BinaryOp : public ::flow::Filter
{
public:
  BinaryOp();
  ~BinaryOp();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Cycle : public ::flow::Filter
{
public:
  Cycle();
  ~Cycle();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Time : public ::flow::Filter
{
public:
  Time();
  ~Time();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ScalarMax : public ::flow::Filter
{
public:
  ScalarMax();
  ~ScalarMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ScalarMin : public ::flow::Filter
{
public:
  ScalarMin();
  ~ScalarMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayMin : public ::flow::Filter
{
public:
  ArrayMin();
  ~ArrayMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldMax : public ::flow::Filter
{
public:
  FieldMax();
  ~FieldMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayMax : public ::flow::Filter
{
public:
  ArrayMax();
  ~ArrayMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldMin : public ::flow::Filter
{
public:
  FieldMin();
  ~FieldMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldAvg : public ::flow::Filter
{
public:
  FieldAvg();
  ~FieldAvg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayAvg : public ::flow::Filter
{
public:
  ArrayAvg();
  ~ArrayAvg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ScalarGradient : public ::flow::Filter
{
public:
  ScalarGradient();
  ~ScalarGradient();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayGradient : public ::flow::Filter
{
public:
  ArrayGradient();
  ~ArrayGradient();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldSum : public ::flow::Filter
{
public:
  FieldSum();
  ~FieldSum();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArraySum : public ::flow::Filter
{
public:
  ArraySum();
  ~ArraySum();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldNanCount : public ::flow::Filter
{
public:
  FieldNanCount();
  ~FieldNanCount();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class FieldInfCount : public ::flow::Filter
{
public:
  FieldInfCount();
  ~FieldInfCount();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Vector : public ::flow::Filter
{
public:
  Vector();
  ~Vector();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Magnitude : public ::flow::Filter
{
public:
  Magnitude();
  ~Magnitude();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Abs : public ::flow::Filter
{
public:
  Abs();
  ~Abs();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};


class Exp : public ::flow::Filter
{
public:
  Exp();
  ~Exp();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Log : public ::flow::Filter
{
public:
  Log();
  ~Log();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Pow : public ::flow::Filter
{
public:
  Pow();
  ~Pow();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Axis : public ::flow::Filter
{
public:
  Axis();
  ~Axis();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Histogram : public ::flow::Filter
{
public:
  Histogram();
  ~Histogram();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Binning : public ::flow::Filter
{
public:
  Binning();
  ~Binning();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Entropy : public ::flow::Filter
{
public:
  Entropy();
  ~Entropy();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Pdf : public ::flow::Filter
{
public:
  Pdf();
  ~Pdf();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Cdf : public ::flow::Filter
{
public:
  Cdf();
  ~Cdf();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class IfExpr : public ::flow::Filter
{
public:
  IfExpr();
  ~IfExpr();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Field : public ::flow::Filter
{
public:
  Field();
  ~Field();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Topo : public ::flow::Filter
{
public:
  Topo();
  ~Topo();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class BinByIndex : public ::flow::Filter
{
public:
  BinByIndex();
  ~BinByIndex();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class BinByValue : public ::flow::Filter
{
public:
  BinByValue();
  ~BinByValue();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class ArrayAccess : public ::flow::Filter
{
public:
  ArrayAccess();
  ~ArrayAccess();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class DotAccess : public ::flow::Filter
{
public:
  DotAccess();
  ~DotAccess();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};


class Quantile : public ::flow::Filter
{
public:
  Quantile();
  ~Quantile();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class PointAndAxis : public ::flow::Filter
{
public:
  PointAndAxis();
  ~PointAndAxis();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class MaxFromPoint : public ::flow::Filter
{
public:
  MaxFromPoint();
  ~MaxFromPoint();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Bin : public ::flow::Filter
{
public:
  Bin();
  ~Bin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Bounds : public ::flow::Filter
{
public:
  Bounds();
  ~Bounds();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Lineout : public ::flow::Filter
{
public:
  Lineout();
  ~Lineout();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Nan : public ::flow::Filter
{
public:
  Nan();
  ~Nan();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

class Replace : public ::flow::Filter
{
public:
  Replace();
  ~Replace();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

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
