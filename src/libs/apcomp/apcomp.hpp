#ifndef APCOMP_H_HPP
#define APCOMP_H_HPP

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
