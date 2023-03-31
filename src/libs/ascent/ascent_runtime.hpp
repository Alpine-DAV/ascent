//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_HPP
#define ASCENT_RUNTIME_HPP

#include <ascent.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

// Pipeline Interface Class

class ASCENT_API Runtime
{
public:
    Runtime();
    virtual ~Runtime();

    virtual void  Initialize(const conduit::Node &options)=0;

    virtual void  Publish(const conduit::Node &data)=0;
    virtual void  Execute(const conduit::Node &actions)=0;

    virtual void  Info(conduit::Node &info_out)=0;

    virtual void  Cleanup()=0;

    virtual void  DisplayError(const std::string &msg);
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


