//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_python_script_filter.hpp
///
//-----------------------------------------------------------------------------



/// This support enables running python-based filter scripts
/// in the case that the host code does not have python.
/// if the host code is python, we don't need to bring our own
/// python interpreter


#ifndef ASCENT_PYTHON_SCRIPT_FILTER_HPP
#define ASCENT_PYTHON_SCRIPT_FILTER_HPP

#include <ascent_exports.h>
#include <ascent_data_object.hpp>

#include <flow_python_script_filter.hpp>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime--
//-----------------------------------------------------------------------------
namespace runtime
{
//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters--
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
///
/// PythonScript runs a given python source.
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ASCENT_API AscentPythonScript : public ::flow::filters::PythonScript
{
public:
    AscentPythonScript();
   ~AscentPythonScript();

    virtual void   declare_interface(conduit::Node &i) override;
    virtual void   execute() override;
};


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters--
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime--
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


