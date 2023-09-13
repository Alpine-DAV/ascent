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
// Base Language Components
// boolean, integer, double, string, nan, null, identifier, dot access, 
// if (conditional), binary operations (math, logical, etc)
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprBoolean : public ::flow::Filter
{
public:
  ExprBoolean();
  ~ExprBoolean();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprInteger : public ::flow::Filter
{
public:
  ExprInteger();
  ~ExprInteger();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprDouble : public ::flow::Filter
{
public:
  ExprDouble();
  ~ExprDouble();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprString : public ::flow::Filter
{
public:
  ExprString();
  ~ExprString();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprNan : public ::flow::Filter
{
public:
  ExprNan();
  ~ExprNan();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprNull : public ::flow::Filter
{
public:
  ExprNull();
  ~ExprNull();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprIdentifier : public ::flow::Filter
{
public:
  ExprIdentifier();
  ~ExprIdentifier();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprObjectDotAccess : public ::flow::Filter
{
public:
  ExprObjectDotAccess();
  ~ExprObjectDotAccess();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprIf : public ::flow::Filter
{
public:
  ExprIf();
  ~ExprIf();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprBinaryOp : public ::flow::Filter
{
public:
  ExprBinaryOp();
  ~ExprBinaryOp();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//---------------------------------------------------------------------------//
// Scalar Operations
//  min, max, abs, exp, log, pow
//---------------------------------------------------------------------------//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprScalarMin : public ::flow::Filter
{
public:
  ExprScalarMin();
  ~ExprScalarMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprScalarMax : public ::flow::Filter
{
public:
  ExprScalarMax();
  ~ExprScalarMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprScalarAbs : public ::flow::Filter
{
public:
  ExprScalarAbs();
  ~ExprScalarAbs();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprScalarExp : public ::flow::Filter
{
public:
  ExprScalarExp();
  ~ExprScalarExp();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprScalarLog : public ::flow::Filter
{
public:
  ExprScalarLog();
  ~ExprScalarLog();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprScalarPow : public ::flow::Filter
{
public:
  ExprScalarPow();
  ~ExprScalarPow();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Vector Operations
//   vector, vector_magnitude
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprVector : public ::flow::Filter
{
public:
  ExprVector();
  ~ExprVector();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprVectorMagnitude : public ::flow::Filter
{
public:
  ExprVectorMagnitude();
  ~ExprVectorMagnitude();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Array Operations
//  access, replace, max, min, avg, sum
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprArrayAccess : public ::flow::Filter
{
public:
  ExprArrayAccess();
  ~ExprArrayAccess();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprArrayReplace : public ::flow::Filter
{
public:
  ExprArrayReplace();
  ~ExprArrayReplace();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprArrayReductionMin : public ::flow::Filter
{
public:
  ExprArrayReductionMin();
  ~ExprArrayReductionMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprArrayReductionMax : public ::flow::Filter
{
public:
  ExprArrayReductionMax();
  ~ExprArrayReductionMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprArrayReductionAvg : public ::flow::Filter
{
public:
  ExprArrayReductionAvg();
  ~ExprArrayReductionAvg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprArrayReductionSum : public ::flow::Filter
{
public:
  ExprArrayReductionSum();
  ~ExprArrayReductionSum();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// History Operations
//  history, history_range, history_gradient, history_range_gradient
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprHistory: public ::flow::Filter
{
public:
  ExprHistory();
  ~ExprHistory();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistoryRange : public ::flow::Filter
{
public:
  ExprHistoryRange();
  ~ExprHistoryRange();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistoryGradient : public ::flow::Filter
{
public:
  ExprHistoryGradient();
  ~ExprHistoryGradient();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistoryGradientRange : public ::flow::Filter
{
public:
  ExprHistoryGradientRange();
  ~ExprHistoryGradientRange();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Histogram Operations
// histogram, entropy, pdf, cdf, quantile, bin_by_value, bin_by_index
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprHistogram : public ::flow::Filter
{
public:
  ExprHistogram();
  ~ExprHistogram();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistogramEntropy : public ::flow::Filter
{
public:
  ExprHistogramEntropy();
  ~ExprHistogramEntropy();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistogramPDF : public ::flow::Filter
{
public:
  ExprHistogramPDF();
  ~ExprHistogramPDF();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistogramCDF : public ::flow::Filter
{
public:
  ExprHistogramCDF();
  ~ExprHistogramCDF();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class  ExprHistogramCDFQuantile : public ::flow::Filter
{
public:
  ExprHistogramCDFQuantile();
  ~ExprHistogramCDFQuantile();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};


//-----------------------------------------------------------------------------
class ExprHistogramBinByIndex : public ::flow::Filter
{
public:
  ExprHistogramBinByIndex();
  ~ExprHistogramBinByIndex();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprHistogramBinByValue : public ::flow::Filter
{
public:
  ExprHistogramBinByValue();
  ~ExprHistogramBinByValue();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Mesh Operations
// cycle, time, field, topology, bounds, lineout
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprMeshCycle : public ::flow::Filter
{
public:
  ExprMeshCycle();
  ~ExprMeshCycle();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshTime : public ::flow::Filter
{
public:
  ExprMeshTime();
  ~ExprMeshTime();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshField : public ::flow::Filter
{
public:
  ExprMeshField();
  ~ExprMeshField();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshTopology : public ::flow::Filter
{
public:
  ExprMeshTopology();
  ~ExprMeshTopology();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshBounds : public ::flow::Filter
{
public:
  ExprMeshBounds();
  ~ExprMeshBounds();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshLineout : public ::flow::Filter
{
public:
  ExprMeshLineout();
  ~ExprMeshLineout();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Mesh Field Operations
//  min reduce, max reduce, avg reduce, sum reduce, nan count, inf count
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprMeshFieldReductionMin : public ::flow::Filter
{
public:
  ExprMeshFieldReductionMin();
  ~ExprMeshFieldReductionMin();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshFieldReductionMax : public ::flow::Filter
{
public:
  ExprMeshFieldReductionMax();
  ~ExprMeshFieldReductionMax();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshFieldReductionAvg : public ::flow::Filter
{
public:
  ExprMeshFieldReductionAvg();
  ~ExprMeshFieldReductionAvg();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshFieldReductionSum : public ::flow::Filter
{
public:
  ExprMeshFieldReductionSum();
  ~ExprMeshFieldReductionSum();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class  ExprMeshFieldReductionNanCount : public ::flow::Filter
{
public:
  ExprMeshFieldReductionNanCount();
  ~ExprMeshFieldReductionNanCount();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshFieldReductionInfCount : public ::flow::Filter
{
public:
  ExprMeshFieldReductionInfCount();
  ~ExprMeshFieldReductionInfCount();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Binning Operations
//  binning, binning_axis, bin_by_index, point_and_axis, max_from_point
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ExprMeshBinning : public ::flow::Filter
{
public:
  ExprMeshBinning();
  ~ExprMeshBinning();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshBinningAxis : public ::flow::Filter
{
public:
  ExprMeshBinningAxis();
  ~ExprMeshBinningAxis();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshBinningBinByIndex: public ::flow::Filter
{
public:
  ExprMeshBinningBinByIndex();
  ~ExprMeshBinningBinByIndex();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshBinningPointAndAxis : public ::flow::Filter
{
public:
  ExprMeshBinningPointAndAxis();
  ~ExprMeshBinningPointAndAxis();

  virtual void declare_interface(conduit::Node &i);
  virtual bool verify_params(const conduit::Node &params, conduit::Node &info);
  virtual void execute();
};

//-----------------------------------------------------------------------------
class ExprMeshBinningMaxFromPoint : public ::flow::Filter
{
public:
  ExprMeshBinningMaxFromPoint();
  ~ExprMeshBinningMaxFromPoint();

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
