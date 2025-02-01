//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_hola.cpp
///
//-----------------------------------------------------------------------------

#include <ascent_hola.hpp>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include <conduit_relay_io.hpp>
#include <conduit_relay_io_handle.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay_io_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_logging_old.hpp>

#include <fstream>

#if defined(ASCENT_MPI_ENABLED)
    #include "ascent_hola_mpi.hpp"
    #include <conduit_relay_mpi.hpp>
    #include <conduit_relay_mpi_io_blueprint.hpp>
#endif

using namespace conduit;
using namespace std;
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
void hola(const std::string &source,
          const Node &options,
          Node &data)
{
    data.reset();
    if(source == "relay/blueprint/mesh")
    {
	std::string root_file = options["root_file"].as_string();
#if defined(ASCENT_MPI_ENABLED)
	MPI_Comm comm  = MPI_Comm_f2c(options["mpi_comm"].to_int());
	conduit::relay::mpi::io::blueprint::load_mesh(root_file,data,comm);
#else
	conduit::relay::io::blueprint::load_mesh(root_file,data);
#endif
    }
    else if(source == "hola_mpi")
    {
#if defined(ASCENT_MPI_ENABLED)
        hola_mpi(options,data);
#else
        ASCENT_ERROR("mpi disabled: 'hola_mpi' can only be used in ascent_mpi" );
#endif
    }
    else
    {
        ASCENT_ERROR("Unknown hola source: " << source);
    }

}

//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------


