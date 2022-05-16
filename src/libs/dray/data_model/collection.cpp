// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/collection.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/utils/mpi_utils.hpp>
#include <set>
#include <sstream>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

Collection::Collection()
{

}
void
Collection::add_domain(const DataSet &domain)
{
  m_domains.push_back(domain);
}

bool
Collection::local_has_field(const std::string field_name)
{
  bool res = true;

  for(DataSet &dom : m_domains)
  {
    res &= dom.has_field(field_name);
  }

  return res;
}

bool
Collection::has_field(const std::string field_name)
{
  bool exists = local_has_field(field_name);
#ifdef DRAY_MPI_ENABLED
  int local_boolean = exists ? 1 : 0;
  int global_boolean;

  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);


  if(global_boolean == dray::mpi_size())
  {
    exists = true;
  }
  else
  {
    // this is technically not needed but added for clarity
    exists = false;
  }
#endif
  return exists;
}

Range Collection::local_range(const std::string field_name)
{
  Range res;

  for(DataSet &dom : m_domains)
  {
    res.include(dom.field(field_name)->range()[0]);
  }
  return res;
}

Range
Collection::range(const std::string field_name)
{
  Range res = local_range(field_name);
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());

  float64 local_min = (float64) res.min();
  float64 local_max = (float64) res.max();
  float64 global_min = 0;
  float64 global_max = 0;

  MPI_Allreduce((void *)(&local_min),
                (void *)(&global_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                mpi_comm);

  MPI_Allreduce((void *)(&local_max),
                (void *)(&global_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                mpi_comm);
  res.reset();
  res.include(Float(global_min));
  res.include(Float(global_max));
#endif
  return res;
}

AABB<3>
Collection::local_bounds()
{
  AABB<3> res;

  for(DataSet &dom : m_domains)
  {
    res.include(dom.mesh()->bounds());
  }

  return res;
}

AABB<3>
Collection::bounds()
{
  AABB<3> res = local_bounds();
  if(!m_bounds.is_empty())
  {
    return m_bounds;
  }

#ifdef DRAY_MPI_ENABLED

  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  AABB<3> global_bounds;
  for(int i = 0; i < 3; ++i)
  {

    float64 local_min = (float64)res.m_ranges[i].min();
    float64 local_max = (float64)res.m_ranges[i].max();
    float64 global_min = 0;
    float64 global_max = 0;

    MPI_Allreduce((void *)(&local_min),
                  (void *)(&global_min),
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_comm);

    MPI_Allreduce((void *)(&local_max),
                  (void *)(&global_max),
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_comm);

    global_bounds.m_ranges[i].include(Float(global_min));
    global_bounds.m_ranges[i].include(Float(global_max));
  }
  res.include(global_bounds);
#endif
  m_bounds = res;
  return res;
}

int32
Collection::topo_dims()
{
  int dims = 0;
  for(DataSet &dom : m_domains)
  {
    dims = std::max(dom.mesh()->dims(), dims);
  }
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());

  int global_dims;
  MPI_Allreduce((void *)(&dims),
                (void *)(&global_dims),
                1,
                MPI_INT,
                MPI_MAX,
                mpi_comm);

  dims = global_dims;
#endif
  return dims;
}

int32
Collection::size()
{
  int32 size = local_size();

  int32 global_size = size;
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());

  MPI_Allreduce((void *)(&size),
                (void *)(&global_size),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

#endif
  return global_size;
}

int32
Collection::local_size()
{
  return (int32)m_domains.size();
}

DataSet
Collection::domain(int32 index)
{
  if(index < 0 || index  > local_size() - 1)
  {
    DRAY_ERROR("Invalid domain index");
  }
  return m_domains[index];
}

std::string Collection::field_list()
{
  std::set<std::string> fields;
  for(int32 i = 0; i < m_domains.size(); ++i)
  {
    std::vector<std::string> dom_fields = m_domains[i].fields();
    for(auto &f : dom_fields)
    {
      fields.insert(f);
    }
  }
  gather_strings(fields);

  std::stringstream ss;
  ss<<"[";
  for(auto &f : fields)
  {
    ss<<" "<<f;
  }
  ss<<"]";
  return ss.str();
}

} // namespace dray
