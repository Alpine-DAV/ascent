//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_hola_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_hola_filters.hpp"

#include "ascent_hola_mpi.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_relay_mpi.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_data_object.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

using namespace conduit;
using namespace std;

using namespace flow;

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


//-----------------------------------------------------------------------------
HolaMPIExtract::HolaMPIExtract()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
HolaMPIExtract::~HolaMPIExtract()
{
// empty
}

//-----------------------------------------------------------------------------
void
HolaMPIExtract::declare_interface(Node &i)
{
    i["type_name"]   = "hola_mpi";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
HolaMPIExtract::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("mpi_comm") ||
       ! params["mpi_comm"].dtype().is_integer() )
    {
        info["errors"].append() = "Missing required integer parameter 'mpi_comm'";
    }

    if(! params.has_child("rank_split") ||
       ! params["rank_split"].dtype().is_integer() )
    {
        info["errors"].append() = "Missing required integer parameter 'rank_split'";
    }

    return res;
}


//-----------------------------------------------------------------------------
void
HolaMPIExtract::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("hola_mpi input must be a DataObject");
    }

    DataObject * data_object = input<DataObject>(0);
    Node *n_input = data_object->as_node().get();
    // assumes multi domain input

    hola_mpi(params(),*n_input);

}



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





