//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_transmogrifier.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_TRANSMOGRIGIFIER_HPP
#define ASCENT_TRANSMOGRIGIFIER_HPP

#include <flow_filter.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

// The Transmogrifier is an invention that would one thing into another.
class Transmogrifier
{
public:
// refinement level for high order data
static int m_refinement_level;

static conduit::Node* low_order(conduit::Node &dataset);

static bool is_high_order(const conduit::Node &doms);

static bool is_poly(const conduit::Node &doms);

static void to_poly(conduit::Node &doms, conduit::Node &to_vtkh);

};

//-----------------------------------------------------------------------------
};
#endif
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
