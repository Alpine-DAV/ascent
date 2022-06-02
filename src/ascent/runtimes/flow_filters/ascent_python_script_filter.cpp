//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_python_script_filter.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_python_script_filter.hpp"


//-----------------------------------------------------------------------------
// flow includes
//-----------------------------------------------------------------------------
#include <flow_workspace.hpp>

using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

namespace runtime
{

namespace filters
{
//-----------------------------------------------------------------------------
AscentPythonScript::AscentPythonScript()
: PythonScript()
{
// empty
}

//-----------------------------------------------------------------------------
AscentPythonScript::~AscentPythonScript()
{
// empty
}

//-----------------------------------------------------------------------------
void
AscentPythonScript::declare_interface(Node &i)
{
    i["type_name"] = "ascent_python_script";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
AscentPythonScript::execute()
{
    // make sure we have our interpreter setup b/c
    // we need the python env ready
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("AscentPythonScript input must be a DataObject");
    }

    DataObject *data_object = input<DataObject>(0);

    conduit::Node *n_input = data_object->as_node().get();

    execute_python(n_input);
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::fitlers--
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
