#include <dray/utils/appstats.hpp>
#include <dray/dray.hpp>
#include <dray/array.hpp>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

namespace dray
{

namespace stats
{

namespace detail
{
  
//---------------------------------------------------------------------------//
void write_ray_data(const int32 width,
                    const int32 height,
                    std::vector<std::pair<int32,Stats>> &ray_data,
                    const std::string &file_name)
{
#ifdef DRAY_STATS
  // create a blank field we can fill ine
  const int32 image_size = width * height;

  std::vector<float32> c_field;
  c_field.resize(image_size);
  std::vector<float32> n_field;
  n_field.resize(image_size);
  std::vector<float32> f_field;
  f_field.resize(image_size);

  std::fill(c_field.begin(), c_field.end(), 0.f);
  std::fill(n_field.begin(), n_field.end(), 0.f);
  std::fill(f_field.begin(), f_field.end(), 0.f);

  for(int i = 0; i < ray_data.size(); ++i)
  {
    auto p = ray_data[i];
    c_field[p.first] = p.second.m_candidates;
    n_field[p.first] = p.second.m_newton_iters;
    f_field[p.first] = p.second.m_found;
  }

  std::ofstream file;
  file.open (file_name + ".vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"ray data\n";
  file<<"ASCII\n";
  file<<"DATASET STRUCTURED_POINTS\n";
  file<<"DIMENSIONS "<<width + 1<<" "<<height + 1<<" 1\n";

  file<<"CELL_DATA "<<width * height<<"\n";

  file<<"SCALARS candidates float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < image_size; ++i)
  {
    file<<c_field[i]<<"\n";
  }

  file<<"SCALARS newton_iters float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < image_size; ++i)
  {
    file<<n_field[i]<<"\n";
  }

  file<<"SCALARS found float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < image_size; ++i)
  {
    file<<f_field[i]<<"\n";
  }

  file.close();
#else
  (void) width;
  (void) height;
  (void) ray_data;
  (void) file_name;
#endif
}

//---------------------------------------------------------------------------//
template<typename T>
void add_point_stats_impl(Array<Vec<T,3>> &points,
                        Array<Stats> &stats,
                        std::vector<std::vector<std::pair<Vec<float32,3>,Stats>>> &point_stats)
{
#ifdef DRAY_STATS
  const int32 size = points.size();
  std::vector<std::pair<Vec<float32,3>,Stats>> point_data;
  point_data.resize(size);
  Vec<T,3> *point_ptr = points.get_host_ptr();
  Stats *stat_ptr = stats.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {
    Vec<T,3> point_t = point_ptr[i];
    Vec<float32,3> point_f;

    point_f[0] = static_cast<float32>(point_t[0]);
    point_f[1] = static_cast<float32>(point_t[1]);
    point_f[2] = static_cast<float32>(point_t[2]);

    Stats mstat = stat_ptr[i];
    point_data[i] = std::make_pair(point_f, mstat);
  }

  point_stats.push_back(std::move(point_data));
#else
  (void) points;
  (void) stats;
  (void) point_stats;
#endif
}

} // namespace detail

//---------------------------------------------------------------------------//
// static
//---------------------------------------------------------------------------//
#ifdef DRAY_STATS
bool StatStore::m_stats_supported = true;
#else
bool StatStore::m_stats_supported = false;
#endif

bool StatStore::m_stats_enabled = false;

std::vector<std::vector<std::pair<int32,Stats>>> StatStore::m_ray_stats;
std::vector<std::vector<std::pair<Vec<float32,3>,Stats>>> StatStore::m_point_stats;

//---------------------------------------------------------------------------//
bool
StatStore::stats_supported()
{ 
  return m_stats_supported;
}

//---------------------------------------------------------------------------//
bool
StatStore::stats_enabled()
{ 
  return m_stats_enabled;
}

//---------------------------------------------------------------------------//
void
StatStore::enable_stats()
{ 
  if(!m_stats_supported)
  {
    DRAY_ERROR (
    "StatStore::enable_stats() -- Cannot enable stats, "
    "Dray was compiled without stats support (DRAY_ENABLE_STATS=OFF)");
  }
  m_stats_enabled = true;
}

//---------------------------------------------------------------------------//
void
StatStore::disable_stats()
{ 
  m_stats_enabled = false;
}


//---------------------------------------------------------------------------//
void
StatStore::write_point_stats(const std::string &ofile_base)
{
  if(!m_stats_enabled)
  {
    return;
  }
#ifdef DRAY_STATS
  const int32 num_layers = m_point_stats.size();
  int32 tot_size = 0;
  for(int32 l = 0; l < num_layers; ++l)
  {
    tot_size += m_point_stats[l].size();
  }

  std::stringstream file_name;
  file_name << ofile_base << "_point_stats_" << dray::mpi_rank() << ".vtk";
  std::ofstream file;
  file.open (file_name.str());
  file<<"# vtk DataFile Version 3.0\n";
  file<<"particles\n";
  file<<"ASCII\n";
  file<<"DATASET UNSTRUCTURED_GRID\n";
  file<<"POINTS "<<tot_size<<" double\n";

  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.first[0]<<" ";
      file<<p.first[1]<<" ";
      file<<p.first[2]<<"\n";
    }
  }

  file<<"CELLS "<<tot_size<<" "<<tot_size* 2<<"\n";
  for(int i = 0; i < tot_size; ++i)
  {
    file<<"1 "<<i<<"\n";
  }

  file<<"CELL_TYPES "<<tot_size<<"\n";
  for(int i = 0; i < tot_size; ++i)
  {
    file<<"1\n";
  }

  file<<"POINT_DATA "<<tot_size<<"\n";
  file<<"SCALARS candidates float\n";
  file<<"LOOKUP_TABLE default\n";

  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.second.m_candidates<<"\n";
    }
  }

  file<<"SCALARS newton_iters float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.second.m_newton_iters<<"\n";
    }
  }

  file<<"SCALARS found float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.second.m_found<<"\n";
    }
  }

  file.close();
  m_point_stats.clear();
#else
  (void) name;
#endif
}

//---------------------------------------------------------------------------//
void
StatStore::clear()
{
  m_point_stats.clear();
  m_ray_stats.clear();
}

//---------------------------------------------------------------------------//
void
StatStore::add_ray_stats(const Array<Ray> &rays, Array<Stats> &stats)
{
  if(!m_stats_enabled)
  {
    return;
  }
#ifdef DRAY_STATS
  const int32 size = rays.size();
  std::vector<std::pair<int32,Stats>> ray_data;
  ray_data.resize(size);
  const Ray *ray_ptr = rays.get_host_ptr_const();
  Stats *stat_ptr = stats.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {
    const Ray ray = ray_ptr[i];
    Stats mstat = stat_ptr[i];
    ray_data[i] = std::make_pair(ray.m_pixel_id, mstat);
  }

 m_ray_stats.push_back(std::move(ray_data));
#else
 (void) rays;
 (void) stats;
#endif
}

//---------------------------------------------------------------------------//
void
StatStore::add_point_stats(Array<Vec<Float,3>> &points, Array<Stats> &stats)
{
  if(!m_stats_enabled)
  {
    return;
  }
#ifdef DRAY_STATS
  const int32 size = points.size();
  std::vector<std::pair<Vec<float32,3>,Stats>> point_data;
  point_data.resize(size);
  Vec<Float,3> *point_ptr = points.get_host_ptr();
  Stats *stat_ptr = stats.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {
    Vec<Float,3> point_t = point_ptr[i];
    Vec<float32,3> point_f;

    point_f[0] = static_cast<float32>(point_t[0]);
    point_f[1] = static_cast<float32>(point_t[1]);
    point_f[2] = static_cast<float32>(point_t[2]);

    Stats mstat = stat_ptr[i];
    point_data[i] = std::make_pair(point_f, mstat);
  }

  m_point_stats.push_back(std::move(point_data));
#else
  (void) points;
  (void) stats;
#endif
}

//---------------------------------------------------------------------------//
void
StatStore::write_ray_stats(const std::string &ofile_base,
                           const int32 width,const int32 height)
{
  if(!m_stats_enabled)
  {
    return;
  }
#ifdef DRAY_STATS
  const int num_images = m_ray_stats.size();
  for(int i = 0; i < num_images; ++i)
  {
    std::stringstream ss;
    ss<< ofile_base << "_ray_stats_" << i << "_r_" << dray::mpi_rank();
    detail::write_ray_data(width,
                           height,
                           m_ray_stats[i],
                           ss.str());
  }

  m_ray_stats.clear();
#else
  (void) width;
  (void) height;
#endif
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const Stats &stats)
{
  if(!StatStore::stats_enabled())
  {
    return os;
  }
#ifdef DRAY_STATS
  os << "[" << stats.m_newton_iters <<", "<<stats.m_candidates<<"]";
#else
  (void) stats;
#endif
  return os;
}

} // namespace stats

} // namespace dray
