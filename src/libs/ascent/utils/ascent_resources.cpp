//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_resources.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_resources.hpp"

#include <ascent.hpp>
#include <ascent_config.h>
#include <ascent_logging.hpp>
#include <ascent_resources_cinema_web.hpp>
#include <ascent_resources_ascent_web.hpp>

// conduit includes
#include <conduit_relay.hpp>

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::resources:: --
//-----------------------------------------------------------------------------
namespace resources
{


//-----------------------------------------------------------------------------
void
load_compiled_resource_tree(const std::string &resource_name,
                            conduit::Node &res)
{
    res.reset();
    if(resource_name == "cinema_web")
    {
        res.parse(RC_CINEMA_WEB,"conduit_base64_json");
    }
    else if(resource_name == "ascent_web")
    {
        res.parse(RC_ASCENT_WEB,"conduit_base64_json");
    }
}

//-----------------------------------------------------------------------------
void
expand_resource_tree_to_file_system(const conduit::Node &resource_tree,
                                    const std::string &path)
{
    NodeConstIterator itr = resource_tree.children();
    while(itr.has_next())
    {
        const Node &curr = itr.next();
        const std::string name = itr.name();

        if(curr.dtype().is_object())
        {
            // object will become a dir
            std::string child_dir = conduit::utils::join_file_path(path,
                                                                   name);

            // create a folder if it doesn't exist
            if(!conduit::utils::is_directory(child_dir))
            {
                conduit::utils::create_directory(child_dir);
            }

            expand_resource_tree_to_file_system(curr,child_dir);
        }
        else if( curr.dtype().is_string() )
        {
            std::string child_file = conduit::utils::join_file_path(path,
                                                                    name);
            std::ofstream ofs;
            ofs.open(child_file.c_str());
            if(!ofs.is_open())
            {
                ASCENT_ERROR("expand_to_file_system failed to open file: "
                             << "\"" << child_file << "\"");
            }
            ofs << curr.as_string();
        }
        else
        {
            ASCENT_ERROR("expand_to_file_system only supports text files.");
        }
    }
}



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::resources:: --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



