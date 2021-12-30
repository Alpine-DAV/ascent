//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_static_schedular_h
#define rover_static_schedular_h

#include <scheduler.h>

namespace rover {
// static scedular handles the case where all ranks get all rays
// and takes care of the compositing.
class StaticSchedular : public Schedular
{
public:
protected:
};
} // namespace rover
#endif
