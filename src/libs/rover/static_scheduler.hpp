//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_static_scheduler_h
#define rover_static_scheduler_h

#include <scheduler.hpp>

namespace rover
{

#if 0 // removing volume renderer
// static scedular handles the case where all ranks get all rays
// and takes care of the compositing.
class StaticSchedular : public Schedular
{
public:
protected:
};
#endif

} // namespace rover
#endif
