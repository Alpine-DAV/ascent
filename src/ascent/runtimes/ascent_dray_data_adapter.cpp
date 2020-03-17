//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_dray_data_adapter.cpp
///
//-----------------------------------------------------------------------------
#include <ascent_dray_data_adapter.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <ascent_logging.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

DRayCollection::DRayCollection()
  : m_mpi_comm_id(-1)
{
}

void
DRayCollection::mpi_comm(int comm)
{
  m_mpi_comm_id = comm;
}

dray::Range
DRayCollection::get_global_range(const std::string field_name)
{
  dray::Range res;

  for(dray::DataSet &dom : m_domains)
  {
    res.include(dom.field(field_name)->range()[0]);
  }

#ifdef ASCENT_MPI_ENABLED
  if(m_mpi_comm_id == -1)
  {
    ASCENT_ERROR("DRayCollection: mpi_comm never set");
  }

  MPI_Comm mpi_comm = MPI_Comm_f2c(m_mpi_comm_id);
  float local_min = res.min();
  float local_max = res.max();
  float global_min = 0;
  float global_max = 0;

  MPI_Allreduce((void *)(&local_min),
                (void *)(&global_min),
                1,
                MPI_FLOAT,
                MPI_MIN,
                mpi_comm);

  MPI_Allreduce((void *)(&local_max),
                (void *)(&global_max),
                1,
                MPI_FLOAT,
                MPI_MAX,
                mpi_comm);
  res.reset();
  res.include(global_min);
  res.include(global_max);
#endif
    return res;
  }

dray::AABB<3>
DRayCollection::get_global_bounds()
{
  dray::AABB<3> res;

  for(dray::DataSet &dom : m_domains)
  {
    res.include(dom.topology()->bounds());
  }
#ifdef ASCENT_MPI_ENABLED
  if(m_mpi_comm_id == -1)
  {
    ASCENT_ERROR("DRayCollection: mpi_comm never set");
  }

  MPI_Comm mpi_comm = MPI_Comm_f2c(m_mpi_comm_id);
  dray::AABB<3> global_bounds;
  for(int i = 0; i < 3; ++i)
  {

    float local_min = res.m_ranges[i].min();
    float local_max = res.m_ranges[i].max();
    float global_min = 0;
    float global_max = 0;

    MPI_Allreduce((void *)(&local_min),
                  (void *)(&global_min),
                  1,
                  MPI_FLOAT,
                  MPI_MIN,
                  mpi_comm);

    MPI_Allreduce((void *)(&local_max),
                  (void *)(&global_max),
                  1,
                  MPI_FLOAT,
                  MPI_MAX,
                  mpi_comm);

    global_bounds.m_ranges[i].include(global_min);
    global_bounds.m_ranges[i].include(global_max);
  }
  res.include(global_bounds);
#endif
  return res;
}

DRayCollection DRayCollection::boundary()
{
    DRayCollection dcol;
    const int num_domains = m_domains.size();
    for(int i = 0; i < num_domains; ++i)
    {
      // we only want the external faces
      dray::MeshBoundary boundary;
      dcol.m_domains.push_back(boundary.execute(m_domains[i]));
    }
    return dcol;
}

DRayCollection*
DRayCollection::blueprint_to_dray(const conduit::Node &node)
{
  DRayCollection *dcol = new DRayCollection();
  int num_domains = node.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = node.child(i);
    dray::DataSet dataset = dray::BlueprintReader::blueprint_to_dray(dom);
    dcol->m_domains.push_back(dataset);
  }
  return dcol;
}

}
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

