//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_conduit_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_blueprint.hpp"


//-----------------------------------------------------------------------------
TEST(conduit_smoke, conduit_about)
{
    conduit::Node about;
    conduit::about(about["conduit"]);
    conduit::relay::about(about["conduit/relay"]);
    conduit::relay::io::about(about["conduit/relay/io"]);
    conduit::blueprint::about(about["conduit/blueprint"]);

    std::cout << about.to_yaml() << std::endl;
}

