//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_python_script_filter.hpp
///
//-----------------------------------------------------------------------------



/// This support enables running python-based filter scripts
/// in the case that the host code does not have python.
/// if the host code is python, we don't need to bring our own
/// python interpreter


#ifndef FLOW_PYTHON_SCRIPT_FILTER_HPP
#define FLOW_PYTHON_SCRIPT_FILTER_HPP

#include <flow_exports.h>
#include <flow_config.h>

#include <flow_filter.hpp>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

class PythonInterpreter;

//-----------------------------------------------------------------------------
// -- begin flow::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
///
/// PythonScript runs a given python source.
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class FLOW_API PythonScript : public ::flow::Filter
{
public:
    PythonScript();
   ~PythonScript();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();

protected:
    void execute_python(conduit::Node *n);
private:
    static flow::PythonInterpreter *interpreter();
    static flow::PythonInterpreter *m_interp;
};


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


