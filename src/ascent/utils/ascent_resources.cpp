//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2021, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_resources.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_web_interface.hpp"

#include <ascent.hpp>
#include <ascent_config.h>
#include <ascent_file_system.hpp>
#include <ascent_logging.hpp>
#include <ascent_resources_cinema_web.hpp>

// thirdparty includes
#include <lodepng.h>

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
            if(!directory_exists(child_dir))
            {
                create_directory(child_dir);
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



