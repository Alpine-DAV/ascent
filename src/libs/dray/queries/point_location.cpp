#include <dray/queries/point_location.hpp>

#include <dray/error.hpp>
#include <dray/warning.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/dray.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

namespace detail
{


bool has_data(const Array<Location> &locs)
{
  const int32 size = locs.size ();

  const Location *locs_ptr = locs.get_device_ptr_const();
  RAJA::ReduceMax<reduce_policy, int32> max_value (-1);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 el_id = locs_ptr[i].m_cell_id;
    max_value.max(el_id);
  });
  DRAY_ERROR_CHECK();

  // if a cell was located then there is valid data here
  return max_value.get() != -1;
}

#ifdef DRAY_MPI_ENABLED
// TODO: just put these functions into a mpi_utils class
void mpi_send(const float32 *data, int32 count, int32 dest, int32 tag, MPI_Comm comm)
{
  MPI_Send(data, count, MPI_FLOAT, dest, tag, comm);
}

void mpi_send(const float64 *data, int32 count, int32 dest, int32 tag, MPI_Comm comm)
{
  MPI_Send(data, count, MPI_DOUBLE, dest, tag, comm);
}

void mpi_recv(float32 *data, int32 count, int32 src, int32 tag, MPI_Comm comm)
{
  MPI_Recv(data, count, MPI_FLOAT, src, tag, comm, MPI_STATUS_IGNORE);
}

void mpi_recv(float64 *data, int32 count, int32 src, int32 tag, MPI_Comm comm)
{
  MPI_Recv(data, count, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
}

void mpi_bcast(float64 *data, int32 count, int32 root, MPI_Comm comm)
{
  MPI_Bcast(data, count, MPI_DOUBLE, root, comm);
}

void mpi_bcast(float32 *data, int32 count, int32 root, MPI_Comm comm)
{
  MPI_Bcast(data, count, MPI_FLOAT, root, comm);
}
#endif

void merge_values(const Float *src, Float *dst, const int32 size, const Float empty_value)
{
  //  we are doing this on the host because we don't want to pay the mem transfer
  //  cost and we are just going to turn around and broadcast the data back to
  //  all ranks
  for(int32 i = 0; i < size; ++i)
  {
    Float value = src[i];
    if(value != empty_value)
    {
      dst[i] = value;
    }
  }
}

void gather_data(std::vector<Array<Float>> &values, bool has_data, const Float empty_value)
{
#ifdef DRAY_MPI_ENABLED
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  int32 has = has_data ? 1 : 0;
  int32 mpi_size = dray::mpi_size();
  int32 mpi_rank = dray::mpi_rank();

  int32 *ranks_data = new int32[mpi_size];

  MPI_Allgather(&has, 1, MPI_INT, ranks_data, 1, MPI_INT, comm);

  // we know we have at least one variable
  const int32 array_size = values[0].size();
  // we also know that we are only doing scalars at the moment
  Float *temp = new Float[array_size];
  const int32 num_vars = values.size();

  // loop through the ranks that actually have line data
  // and gather the data to rank 0
  for(int32 rank = 1; rank < mpi_size; ++rank)
  {
    if(ranks_data[rank] == 1)
    {
      for(int32 i = 0; i < num_vars; ++i)
      {
        if(mpi_rank == 0)
        {
          mpi_recv(temp, array_size, rank, 0, comm);
          merge_values(temp, values[i].get_host_ptr(), array_size, empty_value);
        }
        else if(rank == mpi_rank)
        {
          mpi_send(values[i].get_host_ptr_const(), array_size, 0, 0, comm);
        }

      }
    }
  }
  // now turn around and broadcast the data back to all ranks
  // NOTE: we could make this a parameter, but we might use this
  // within ascent's expressions so all ranks would need the data
  for(int32 i = 0; i < num_vars; ++i)
  {
    mpi_bcast(values[i].get_host_ptr(), array_size, 0, comm);
  }

  delete[] temp;
  delete[] ranks_data;

#endif
}

struct PointLocationLocateFunctor
{
  Array<Vec<Float,3>> m_points;
  Array<Vec<Float,3>> m_ref_points;
  PointLocationLocateFunctor(const Array<Vec<Float,3>> &points)
    : m_points(points)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    //m_res = detail::reflect_execute(topo.mesh(), m_point, m_normal);
  }
};

}//namespace detail

PointLocation::PointLocation()
  : m_empty_val(0)
{
}

void
PointLocation::add_var(const std::string var)
{
  m_vars.push_back(var);
}

void
PointLocation::empty_val(const Float val)
{
  m_empty_val = val;
}

PointLocation::Result
PointLocation::execute(Collection &collection, Array<Vec<Float,3>> &points)
{
  const int32 vars_size = m_vars.size();
  if(vars_size == 0)
  {
    DRAY_ERROR("PointLocation: must specify at least 1 variables.");
  }
  const int32 points_size = points.size();
  if(points_size == 0)
  {
    DRAY_ERROR("PointLocation: must specify at least 1 point.");
  }

  std::vector<std::string> missing_vars;
  std::vector<std::string> valid_vars;

  for(int32 i = 0; i < vars_size; ++i)
  {
    if(!collection.has_field(m_vars[i]))
    {
      missing_vars.push_back(m_vars[i]);
    }
    else
    {
      valid_vars.push_back(m_vars[i]);
    }
  }

  if(missing_vars.size() == vars_size)
  {
    std::stringstream msg;
    msg<<"PointLocation: must specify at least one valid field.";
    msg<<" Invalid fields [";
    for(int32 i = 0; i < missing_vars.size(); ++i)
    {
      msg<<" "<<missing_vars[i];
    }
    msg<<" ] ";
    msg<<"Known fields "<<collection.field_list();
    DRAY_ERROR(msg.str());
  }

  // warn the user but continue
  if(missing_vars.size() > 0)
  {
    std::stringstream msg;
    msg<<"PointLocation: some fields provided are invalid.";
    msg<<" Invalid fields [";
    for(int32 i = 0; i < missing_vars.size(); ++i)
    {
      msg<<" "<<missing_vars[i];
    }
    msg<<" ] ";
    msg<<"Known fields "<<collection.field_list();
    DRAY_WARNING(msg.str());
  }

  AABB<3> bounds = collection.bounds();
  int32 topo_dims = collection.topo_dims();
  if(topo_dims == 2)
  {
    Range z_range = bounds.m_ranges[2];
    if(z_range.length() != 0)
    {
      DRAY_ERROR("Cannot perform lineout on 2d data where z != 0, "<<
                 "i.e., all data must be on a plane.");
    }
  }

  const int32 valid_size = valid_vars.size();
  std::vector<Array<Float>> values;
  values.resize(valid_size);
  for(int32 i = 0; i < valid_size; ++i)
  {
    values[i].resize(points_size);
    array_memset(values[i], m_empty_val);
  }

  bool has_data = false;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    // we are looping over all the points in each domain.
    // if the points are not found, the values won't be updated,
    // so at the end, we will should have all the field values
    DataSet data_set = collection.domain(i);
    Array<Location> locs = data_set.mesh()->locate(points);
    bool domain_has_data = detail::has_data(locs);
    if(domain_has_data)
    {
      has_data = true;
      for(int32 f = 0; f < valid_size; ++f)
      {
        // TODO: one day we might need to check if this
        // particular data has each field
        data_set.field(valid_vars[f])->eval(locs, values[f]);
      }
    }
  }

  detail::gather_data(values, has_data, m_empty_val);

  Result res;
  res.m_points = points;
  res.m_vars = m_vars;
  res.m_values = values;
  res.m_empty_val = m_empty_val;

  return res;
}


}//namespace dray
