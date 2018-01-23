//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
/// file: ascent_file_system.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_file_system.hpp"

#include "ascent_logging.hpp"

// standard includes
#include <stdlib.h>
// unix only
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


//-----------------------------------------------------------------------------
bool
directory_exists(const std::string &path)
{
    return conduit::utils::is_directory(path);
}

//-----------------------------------------------------------------------------
bool
create_directory(const std::string &path)
{
    return  conduit::utils::create_directory(path);
}

//-----------------------------------------------------------------------------
bool
copy_file(const std::string &src_path,
          const std::string &dest_path)
{
    std::ifstream ifile(src_path, std::ios::binary);
    std::ofstream ofile(dest_path, std::ios::binary);

    ofile << ifile.rdbuf();
    
    return true;
}


//-----------------------------------------------------------------------------
bool
copy_directory(const std::string &src_path,
               const std::string &dest_path)
{
    bool res = true;

    if(!directory_exists(dest_path))
    {
        create_directory(dest_path);
    }

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (src_path.c_str())) != NULL)
    {

        while ( res == true &&  ( (ent = readdir (dir)) != NULL))
        { 
            // filter pwd and parent
            if (ent->d_name != std::string(".") && ent->d_name != std::string("..") )
            {
                // check if we have a file or a dir
                
                std::string src_child_path = conduit::utils::join_path(src_path,
                                                                       std::string(ent->d_name));
                std::string dest_child_path = conduit::utils::join_path(dest_path,
                                                                        std::string(ent->d_name));

                if(directory_exists(src_child_path))
                {
                    res = res && copy_directory(src_child_path, dest_child_path);
                }
                else
                {
                    res = res && copy_file(src_child_path, dest_child_path);
                }
            }
        }
        
        closedir (dir);
    }
    else
    {
        res = false;
    }
    return res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



