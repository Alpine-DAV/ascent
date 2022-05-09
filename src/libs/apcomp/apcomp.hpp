//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_HPP
#define APCOMP_HPP

#include <apcomp/apcomp_exports.h>
#include <string>

namespace apcomp
{

  APCOMP_API bool mpi_enabled();
  APCOMP_API bool openmp_enabled();
  APCOMP_API int  mpi_rank();
  APCOMP_API int  mpi_size();

  APCOMP_API void mpi_comm(int mpi_comm_id);
  APCOMP_API int  mpi_comm();

  APCOMP_API std::string about();
}
#endif
