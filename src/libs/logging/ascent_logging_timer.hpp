//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_logging_timer.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_LOGGING_TIMER_HPP
#define ASCENT_LOGGING_TIMER_HPP

#include <ascent_logging_exports.h>
#include <chrono>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


//-----------------------------------------------------------------------------
class ASCENT_API Timer
{

public:

    explicit Timer();
            ~Timer();
    void     reset();
    float    elapsed() const;

private:
    std::chrono::high_resolution_clock::time_point m_start;
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


