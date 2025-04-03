//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_utils.hpp"
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_metadata.hpp>

#include <algorithm>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#endif

using namespace conduit;

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

std::string output_dir(const std::string file_name)
{
  std::string output_path;

  std::string file, base_path;
  conduit::utils::rsplit_file_path(file_name, file, base_path);
  if(base_path == "")
  {
    std::string dir = default_dir();
    output_path = conduit::utils::join_file_path(dir, file);
  }
  else
  {
    output_path = file_name;
  }
  return output_path;
}

//-----------------------------------------------------------------------------

bool check_dir_path_exists(const std::string file_path, int mpi_comms_id, conduit::Node &err_msg)
{
  int rank = 0;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comms_id);
  MPI_Comm_rank(mpi_comm, &rank);
#endif

  int res = true;
  std::string curr, next;
  conduit::utils::rsplit_file_path(file_path, curr, next);
  
  // If no directory is given or if the given directory does not exist log an error
  // Only check on rank 0
  if(rank == 0 && next.length() != 0 && !conduit::utils::is_directory(next))
  {
    err_msg.set("Error: The specified directory '" + next + 
                "' does not exist. Please check the path and try again.");
    res = false;
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Bcast(&res, 1, MPI_INT, 0, mpi_comm);
  conduit::relay::mpi::broadcast_using_schema(err_msg, 0, mpi_comm);
#endif

  return res;
}

//-----------------------------------------------------------------------------

std::string default_dir()
{
  if(Metadata::n_metadata.has_path("default_dir"))
  {
    return Metadata::n_metadata["default_dir"].as_string();
  }
  else return ".";
}

//-----------------------------------------------------------------------------

std::string filter_to_path(const std::string filter_name)
{
  std::string res;
  std::vector<std::string> path = split(filter_name, '_');
  for(size_t i = 0; i < path.size(); ++i)
  {
    res += path[i];
    if(i != path.size() - 1)
    {
      res += "/";
    }
  }
  return res;
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





