#include <dray/queries/lineout.hpp>
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


Lineout::Lineout()
  : m_samples(100),
    m_empty_val(0)
{
}

int32
Lineout::samples() const
{
  return m_samples;
}

void
Lineout::samples(int32 samples)
{
  if(samples < 1)
  {
    DRAY_ERROR("Number of samples must be positive");
  }
  m_samples = samples;
}

void
Lineout::add_line(const Vec<Float,3> start, const Vec<Float,3> end)
{
  m_starts.push_back(start);
  m_ends.push_back(end);
}

void
Lineout::add_var(const std::string var)
{
  m_vars.push_back(var);
}

void
Lineout::empty_val(const Float val)
{
  m_empty_val = val;
}

Array<Vec<Float,3>>
Lineout::create_points()
{
  const int32 lines_size = m_starts.size();
  // number of samples will be at the beginning + end of
  // the line plus how ever many samples the user asks for
  // m_samples must be > 0
  const int32 samples = m_samples + 2;
  const int32 total_points = lines_size * samples;

  Array<Vec<Float,3>> starts;
  Array<Vec<Float,3>> ends;
  starts.resize(lines_size);
  ends.resize(lines_size);
  // pack the points
  {
    Vec<Float,3> *starts_ptr = starts.get_host_ptr();
    Vec<Float,3> *ends_ptr = ends.get_host_ptr();
    for(int32 i = 0; i < lines_size; ++i)
    {
      starts_ptr[i] = m_starts[i];
      ends_ptr[i] = m_ends[i];
    }
  }

  Array<Vec<Float,3>> points;
  points.resize(total_points);
  Vec<Float,3> *points_ptr = points.get_device_ptr();

  const Vec<Float,3> *starts_ptr = starts.get_device_ptr_const();
  const Vec<Float,3> *ends_ptr = ends.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_points), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 line_id = i / samples;
    const int32 sample_id = i % samples;
    const Vec<Float,3> start = starts_ptr[line_id];
    const Vec<Float,3> end = ends_ptr[line_id];
    Vec<Float,3> dir = end - start;
    const Float step = dir.magnitude() / Float(samples - 1);
    dir.normalize();
    Vec<Float,3> point = start + (step * Float(sample_id)) * dir;

    points_ptr[i] = point;
  });
  DRAY_ERROR_CHECK();

  return points;
}

Lineout::Result
Lineout::execute(Collection &collection)
{
  const int32 vars_size = m_vars.size();
  if(vars_size == 0)
  {
    DRAY_ERROR("Lineout: must specify at least 1 variables:");
  }
  const int32 lines_size = m_starts.size();
  if(lines_size == 0)
  {
    DRAY_ERROR("Lineout: must specify at least 1 line:");
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
    msg<<"Lineout: must specify at least one valid field.";
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
    msg<<"Lineout: some fields provided are invalid.";
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

  // set up the location
  PointLocation locator;
  for(int32 i = 0; i < valid_vars.size(); ++i)
  {
    locator.add_var(valid_vars[i]);
  }

  locator.empty_val(m_empty_val);

  Array<Vec<Float,3>> points = create_points();
  PointLocation::Result lres = locator.execute(collection, points);


  Result res;
  res.m_points = points;
  // start + end + samples
  res.m_points_per_line = m_samples + 2;
  res.m_vars = m_vars;
  res.m_values = lres.m_values;
  res.m_empty_val = m_empty_val;

  return res;
}


}//namespace dray
