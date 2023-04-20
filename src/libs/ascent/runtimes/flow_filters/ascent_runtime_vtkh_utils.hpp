//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_VTKH_UTILS_HPP
#define ASCENT_RUNTIME_VTKH_UTILS_HPP

#include <ascent_data_object.hpp>
#include <ascent_vtkh_collection.hpp>
#include <string>
#include <vector>

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
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

namespace detail
{


// call an error due to an known field and build
// up a list of altenative field names
void field_error(const std::string field_name,
                 const std::string filter_name,
                 std::shared_ptr<VTKHCollection> collection,
                 bool error = true);

// build a list of possible topologies in this collection
std::string possible_topologies(std::shared_ptr<VTKHCollection> collection);

// resolve the name of the topology and throw errors if the
// name cannot be deduced or found
std::string resolve_topology(const conduit::Node &params,
                             const std::string filter_name,
                             std::shared_ptr<VTKHCollection> collection,
                             bool error = true);

} // namespace detail
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
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
