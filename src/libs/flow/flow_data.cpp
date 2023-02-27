//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_data.cpp
///
//-----------------------------------------------------------------------------

#include "flow_data.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
Data::Data(void *data)
:m_data_ptr(data)
{
        // empty
}

//-----------------------------------------------------------------------------
Data::~Data()
{
        // empty
}

//-----------------------------------------------------------------------------
void *
Data::data_ptr()
{
    return m_data_ptr;
}

//-----------------------------------------------------------------------------
void
Data::set_data_ptr(void *data_ptr)
{
    m_data_ptr = data_ptr;
}

//-----------------------------------------------------------------------------
const void *
Data::data_ptr() const
{
    return m_data_ptr;
}
//-----------------------------------------------------------------------------
void
Data::info(Node &out) const
{
    out.reset();
    ostringstream oss;
    oss << m_data_ptr;
    out["data_ptr"] = oss.str();
}



//-----------------------------------------------------------------------------
std::string
Data::to_json() const
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Data::print() const
{
    CONDUIT_INFO(to_json());
}


// export compiled instance of Node wrapper
template class FLOW_API DataWrapper<conduit::Node>;


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------



