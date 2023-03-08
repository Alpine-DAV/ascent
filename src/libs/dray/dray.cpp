// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/array_registry.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/dray_config.h>
#include <dray/dray_exports.h>
#include <dray/error.hpp>
#include <iostream>
#include <apcomp/apcomp.hpp>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

static int g_mpi_comm_id = -1;

void dray::set_host_allocator_id(int id)
{
  ArrayRegistry::set_host_allocator_id(id);
}

void dray::set_device_allocator_id(int id)
{
  ArrayRegistry::set_device_allocator_id(id);
}


void
check_comm_handle()
{
  if(g_mpi_comm_id == -1)
  {
    std::stringstream msg;
    msg<<"Devil Ray internal error. There is no valid MPI comm available. ";
    msg<<"It is likely that dray.mpi_comm(int) was not called.";
    DRAY_ERROR(msg.str());
  }
}

void dray::mpi_comm(int mpi_comm_id)
{
#ifdef DRAY_MPI_ENABLED
  g_mpi_comm_id = mpi_comm_id;
  apcomp::mpi_comm(mpi_comm_id);
  DataLogger::get_instance()->set_rank(dray::mpi_rank());
#else
  (void) mpi_comm_id;
  DRAY_ERROR("Cannot set mpi comm handle in non mpi version");
#endif
}

int dray::mpi_comm()
{
#ifdef DRAY_MPI_ENABLED
  check_comm_handle();
#else
  DRAY_ERROR("Cannot get mpi comm handle in non mpi version");
#endif
  return g_mpi_comm_id;
}

bool dray::mpi_enabled()
{
#ifdef DRAY_MPI_ENABLED
  return true;
#else
  return false;
#endif
}

int dray::mpi_size()
{
#ifdef DRAY_MPI_ENABLED
  int size;
  MPI_Comm comm = MPI_Comm_f2c(mpi_comm());
  MPI_Comm_size(comm, &size);
  return size;
#else
  return 1;
#endif
}

int dray::mpi_rank()
{
#ifdef DRAY_MPI_ENABLED
   int rank;
  MPI_Comm comm = MPI_Comm_f2c(mpi_comm());
  MPI_Comm_rank(comm, &rank);
  return rank;
#else
  return 0;
#endif
}

int dray::m_face_subdivisions = 1;
int dray::m_zone_subdivisions = 1;
bool dray::m_prefer_native_order_mesh = true;
bool dray::m_prefer_native_order_field = true;

void dray::set_face_subdivisions (int num_subdivisions)
{
  m_face_subdivisions = num_subdivisions;
}

void dray::set_zone_subdivisions (int num_subdivisions)
{
  m_zone_subdivisions = num_subdivisions;
}

int dray::get_zone_subdivisions ()
{
  return m_zone_subdivisions;
}

int dray::get_face_subdivisions ()
{
  return m_zone_subdivisions;
}

void dray::prefer_native_order_mesh(bool on)
{
  m_prefer_native_order_mesh = on;
}

bool dray::prefer_native_order_mesh()
{
  return m_prefer_native_order_mesh;
}

void dray::prefer_native_order_field(bool on)
{
  m_prefer_native_order_field = on;
}

bool dray::prefer_native_order_field()
{
  return m_prefer_native_order_field;
}

void dray::init ()
{
}

void dray::finalize ()
{
}

bool dray::device_enabled ()
{
#ifdef DRAY_DEVICE_ENABLED
  return true;
#else
  return false;
#endif
}

bool dray::cuda_enabled ()
{
#ifdef DRAY_CUDA_ENABLED
  return true;
#else
  return false;
#endif
}

bool dray::hip_enabled ()
{
#ifdef DRAY_HIP_ENABLED
  return true;
#else
  return false;
#endif
}


void dray::about ()
{
  std::cout << "                                          v0.0.1               "
               "                       \n\n\n";
  std::cout << "                                       @          &,           "
               "                           \n";
  std::cout << "                                      @&          .@           "
               "                           \n";
  std::cout << "                                      @%          .@*          "
               "                           \n";
  std::cout << "                                    &@@@@@@@@@@@@@@@@@/        "
               "                           \n";
  std::cout << "                                   @@@@@@@@@@@@@@@@@@@@.       "
               "                           \n";
  std::cout << "                                   @@@@@@@@@@@@@@@@@@@@,       "
               "                           \n";
  std::cout << "                                 /@@@@@@@@@@@@@@@@@@@@@@@      "
               "                           \n";
  std::cout << "                               &@@@@@@@@@@@@@@@@@@@@@@@@@@@    "
               "                           \n";
  std::cout
  << "                         ,%&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*        "
     "                 \n";
  std::cout
  << "                   (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&  "
     "                 \n";
  std::cout << "              "
               ",@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@# "
               "             \n";
  std::cout << "           "
               "(@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@.          \n";
  std::cout << "        "
               "/@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@&        \n";
  std::cout << "      "
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@@@@@.     \n";
  std::cout << "   "
               ".@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@@@@@@@@@@%   \n";
  std::cout << " .@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@@@@@@@@@@@@@@@# \n";
  std::cout
  << "@#                 /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&  "
     "               *@\n";
  std::cout
  << "                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*      "
     "                 \n";
  std::cout
  << "                           ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          "
     "                 \n";
  std::cout << "                              &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   "
               "                           \n";
  std::cout << "                                 &@@@@@@@@@@@@@@@@@@@@@@@      "
               "                           \n";
  std::cout << "                                    #@@@@@@@@@@@@@@@@@.        "
               "                           \n";
  std::cout << "                                       .@@@@@@@@@@#            "
               "                           \n";
  std::cout << "                                           @@@@.               "
               "                           \n";
  std::cout << "                                            &@%                "
               "                           \n";
  std::cout << "                                            ,@(                "
               "                           \n";
  std::cout << "                                             @,                "
               "                           \n";
  std::cout << "                                             @                 "
               "                           \n";
  std::cout << "                                           ,,@,,               "
               "                           \n";
  std::cout << "                                      /@&.*  @  *,@@*          "
               "                           \n";
  std::cout << "                                    %@       @       &&        "
               "                           \n";
  std::cout << "                                   *&        @        %.       "
               "                           \n";
  std::cout << "                                    @        @        @        "
               "                           \n";
  std::cout << "                                     @       @       @.        "
               "                           \n";
  std::cout << "                                      @      @      @.         "
               "                           \n";
  std::cout << "                                      @      @      @          "
               "                           \n";
  std::cout << "                                      @      @      @          "
               "                           \n";
  std::cout << "                                   &@@       @       @@/       "
               "                           \n";
  std::cout << "                                   @@.       @        @&       "
               "                           \n";
  std::cout << "                                           (@@@&               "
               "                           \n";
  std::cout << "                                            *@%                "
               "                           \n";

  std::cout << "== Precision...: ";
#ifdef DRAY_DOUBLE_PRECISION
  std::cout << "Double\n";
#else
  std::cout << "Single\n";
#endif

  std::cout << "== Logging.....: ";
#ifdef DRAY_ENABLE_LOGGING
  std::cout << "Enabled\n";
#else
  std::cout << "Disabled\n";
#endif

  std::cout << "== Stats.......: ";
#ifdef DRAY_STATS
  std::cout << "Enabled\n";
#else
  std::cout << "Disabled\n";
#endif

  std::cout << "== OpenMP......: ";
#ifdef DRAY_OPENMP_ENABLED
  std::cout << "Enabled\n";
#else
  std::cout << "Disabled\n";
#endif

  std::cout << "== CUDA........: ";
#ifdef DRAY_CUDA_ENABLED
  std::cout << "Enabled\n";
#else
  std::cout << "Disabled\n";
#endif

  std::cout << "== HIP.......: ";
#ifdef DRAY_HIP_ENABLED
  std::cout << "Enabled\n";
#else
  std::cout << "Disabled\n";
#endif

  std::cout << "== MPI.........: ";
  if(mpi_enabled())
  {
    std::cout << "Enabled\n";
  }
  else
  {
    std::cout << "Disabled\n";
  }
}

} // namespace dray
