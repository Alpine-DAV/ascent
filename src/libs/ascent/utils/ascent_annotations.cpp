// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: ascent_annotations.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_annotations.hpp"
#include "ascent_logging.hpp"

//-----------------------------------------------------------------------------
// -- standard lib includes --
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#if defined(ASCENT_CALIPER_ENABLED)
#include "caliper/cali-manager.h"
#endif

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::annotations --
//-----------------------------------------------------------------------------
namespace annotations
{

#if defined(ASCENT_CALIPER_ENABLED)
static cali::ConfigManager *cali_cfg_manager = NULL;
#endif


//-----------------------------------------------------------------------------
bool
supported()
{
#if defined(ASCENT_CALIPER_ENABLED)
    return true;
#else
    return false;
#endif
}

//-----------------------------------------------------------------------------
void
initialize()
{
#if defined(ASCENT_CALIPER_ENABLED)
    conduit::Node opts;
    initialize(opts);
#endif
}

//-----------------------------------------------------------------------------
void
initialize(const conduit::Node &opts)
{
#if defined(ASCENT_CALIPER_ENABLED)

    if(cali_cfg_manager != NULL)
    {
        ASCENT_ERROR("Ascent Caliper Config Manager already initialized.")
    }

    // check opts
    std::string config;
    std::string services;
    std::string ofname;

    if(opts.has_child("config") && opts["config"].dtype().is_string())
    {
        config = opts["config"].as_string();
    }

    if(opts.has_child("services") && opts["services"].dtype().is_string())
    {
        services = opts["services"].as_string();
    }

    if(opts.has_child("output_file") && opts["output_file"].dtype().is_string())
    {
        ofname = opts["output_file"].as_string();
    }

    cali_cfg_manager = new cali::ConfigManager();
    if(!ofname.empty())
    {
        cali_cfg_manager->set_default_parameter("output", ofname.c_str());
    }

    if(!config.empty())
    {
        cali_cfg_manager->add(config.c_str());
    }

    if(!services.empty())
    {
        cali_config_preset("CALI_SERVICES_ENABLE", services.c_str());
    }
    cali_cfg_manager->start();

#endif
}


//-----------------------------------------------------------------------------
void
flush()
{
#if defined(ASCENT_CALIPER_ENABLED)
    if(cali_cfg_manager != NULL)
    {
        cali_cfg_manager->flush();
    }
#endif
}


//-----------------------------------------------------------------------------
void
finalize()
{
#if defined(ASCENT_CALIPER_ENABLED)
    flush();
    if(cali_cfg_manager != NULL)
    {
        delete cali_cfg_manager;
        cali_cfg_manager = NULL;
    }
#endif
}

}
//-----------------------------------------------------------------------------
// -- end ascent::annotations --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


