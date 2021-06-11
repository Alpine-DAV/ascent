#ifndef ASCENT_DATA_BINNING_HPP
#define ASCENT_DATA_BINNING_HPP

#include <conduit.hpp>
#include <ascent_exports.h>

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

// bin_axes lists of axes are in the form:
//  axis_name:
//    num_bins : int (required if explicit bins are not given
//    min_val : double (optional used with num_bins)
//    max_val : double (optional used with num_bins)
//    clamp : int (optional: 0 = false 1 = true)
//    bins : [double, ...] (required if not using automatic bins)
//
//  params:
//    reduction_var: field on the mesh to bin
//    reduction_op: min, max, ave, sum, pdf, rms, var, std
//    component: if the variable is a vector, which component

ASCENT_API
conduit::Node data_binning(conduit::Node &dataset,
                           conduit::Node &bin_axes,
                           const std::string &reduction_var,
                           const std::string &reduction_op,
                           const double empty_bin_val,
                           const std::string &component);

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
