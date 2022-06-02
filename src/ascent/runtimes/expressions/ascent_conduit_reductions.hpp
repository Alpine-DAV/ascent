//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_conduit_reductions.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_CONDUIT_REDUCTIONS
#define ASCENT_CONDUIT_REDUCTIONS

#include <ascent.hpp>
#include <conduit.hpp>


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

<<<<<<< HEAD
conduit::Node field_reduction_max(const conduit::Node &field, std::string component = "");
=======
conduit::Node array_gradient(const conduit::Node &y_values, const conduit::Node &dx_values, bool is_list = false);

conduit::Node array_max(const conduit::Node &values);
>>>>>>> develop

conduit::Node field_reduction_min(const conduit::Node &field, std::string component = "");

conduit::Node field_reduction_sum(const conduit::Node &field, std::string component = "");

conduit::Node field_reduction_nan_count(const conduit::Node &field, std::string component = "");

// count of all inf or -inf
conduit::Node field_reduction_inf_count(const conduit::Node &field, std::string component = "");

conduit::Node field_reduction_histogram(const conduit::Node &field,
                                        const double &min_value,
                                        const double &max_value,
                                        const int &num_bins,
                                        std::string component = "");

conduit::Node array_max(const conduit::Node &array,
                        const std::string exec_loc,
                        std::string component = "");
conduit::Node array_min(const conduit::Node &array,
                        const std::string exec_loc,
                        std::string component = "");
conduit::Node array_sum(const conduit::Node &array,
                        const std::string exec_loc,
                        std::string component = "");
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

