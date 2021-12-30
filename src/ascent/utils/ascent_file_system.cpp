//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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



