// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/volume.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/rendering/volume_shader.hpp>

#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>
#include <dray/error_check.hpp>
#include <dray/device_color_map.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/device_field.hpp>

namespace dray
{

namespace detail
{
void init_partials(Array<VolumePartial> &partials)
{
  const int32 size = partials.size();
  VolumePartial *partial_ptr = partials.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    VolumePartial p;
    p.m_pixel_id = -1;
    p.m_depth = infinity32();
    p.m_color = {0.f, 0.f, 0.f, 0.f};
    partial_ptr[i] = p;
  });
  DRAY_ERROR_CHECK();
}

Array<VolumePartial> compact_partials(const Array<VolumePartial> &partials)
{
  const int32 size = partials.size();

  if(size == 0)
  {
    return partials;
  }

  Array<int32> valid_segments;
  valid_segments.resize(size);
  int32 * valid_ptr = valid_segments.get_device_ptr();

  const VolumePartial *partial_ptr = partials.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const float32 depth = partial_ptr[i].m_depth;
    valid_ptr[i] = (depth == infinity32()) ? 0 : 1;
  });
  DRAY_ERROR_CHECK();

  Array<int32> compact_idxs = index_flags(valid_segments);

  return gather(partials, compact_idxs);
}

template<typename MeshElement, typename FieldElement>
Array<VolumePartial>
integrate_partials(UnstructuredMesh<MeshElement> &mesh,
                   UnstructuredField<FieldElement> &field,
                   Array<Ray> &rays,
                   Array<PointLight> &lights,
                   const int32 samples,
                   const AABB<3> bounds,
                   ColorMap &color_map,
                   bool use_lighting)
{
  DRAY_LOG_OPEN("volume");
  constexpr float32 correction_scalar = 10.f;
  float32 ratio = correction_scalar / samples;

  ColorMap corrected;
  corrected.scalar_range(color_map.scalar_range());
  corrected.log_scale(color_map.log_scale());
  corrected.color_table(color_map.color_table().correct_opacity(ratio));

  AABB<> sample_bounds = bounds;
  float32 mag = (sample_bounds.max() - sample_bounds.min()).magnitude();
  const float32 sample_dist = mag / float32(samples);

  const int32 num_elems = mesh.cells();

  DRAY_LOG_ENTRY("samples", samples);
  DRAY_LOG_ENTRY("sample_distance", sample_dist);
  DRAY_LOG_ENTRY("cells", num_elems);
  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  // this no longer alerters the incoming rays
  Array<Ray> active_rays = remove_missed_rays(rays, mesh.bounds());
  DRAY_LOG_ENTRY("active_rays", active_rays.size());

  const int32 ray_size = active_rays.size();
  const Ray *rays_ptr = active_rays.get_device_ptr_const();

  constexpr int32 max_segments = 5;
  Array<VolumePartial> partials;
  partials.resize(ray_size * max_segments);
  init_partials(partials);
  VolumePartial *partials_ptr = partials.get_device_ptr();


  // complicated device stuff
  DeviceMesh<MeshElement> device_mesh(mesh);

  DeviceColorMap d_color_map(corrected);


  VolumeShader<MeshElement, FieldElement> shader(mesh,
                                                 field,
                                                 corrected,
                                                 lights);

  Array<stats::Stats> mstats;
  mstats.resize(ray_size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  // TODO: somehow load balance based on far - near
  Timer timer;
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 i)
  {
    const Ray ray = rays_ptr[i];
    // advance the ray one step
    Float distance = ray.m_near + sample_dist;
    constexpr Vec4f clear = {0.f, 0.f, 0.f, 0.f};
    const int32 partial_offset = max_segments * i;
    int32 segment = 0;

    VolumePartial partial;

    partial.m_color = {{0.f, 0.f, 0.f, 0.f}};
    partial.m_pixel_id = ray.m_pixel_id;

    stats::Stats mstat;
    mstat.construct();

    for(int s = 0; s < max_segments; ++s)
    {
      bool found = false;
      // find next segment
      Location loc;
      while(distance < ray.m_far && !found)
      {
        Vec<Float,3> point = ray.m_orig + distance * ray.m_dir;
        loc = device_mesh.locate(point);
        if(loc.m_cell_id != -1)
        {
          found = true;
        }
        else
        {
          distance += sample_dist;
        }
      }

      if(distance >= ray.m_far)
      {
        // we are done
        break;
      }

      partial.m_depth = distance;
      partial.m_color = clear;

      int count = 0;
      mstat.acc_candidates(1);
      do
      {
        // we know we have a valid location

        Vec<float32, 4> sample_color;
        // shade
        if(use_lighting)
        {
          sample_color = shader.shaded_color(loc, ray);
        }
        else
        {
          sample_color = shader.color(loc);
        }

        blend(partial.m_color, sample_color);
        count++;

        distance += sample_dist;
        Vec<Float,3> point = ray.m_orig + distance * ray.m_dir;
        loc = device_mesh.locate(point);
        found = loc.m_cell_id != -1;
      }
      while(distance < ray.m_far && found && partial.m_color[3] < 0.95f);

      partials_ptr[partial_offset + segment] = partial;
      segment++;

      if(distance >= ray.m_far || partial.m_color[3] > 0.95f)
      {
        // we are done
        break;
      }

    } // for segments
    mstats_ptr[i] = mstat;
  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_ENTRY("integrate_partials",timer.elapsed());
  stats::StatStore::add_ray_stats(active_rays, mstats);

  timer.reset();
  partials = detail::compact_partials(partials);
  DRAY_LOG_ENTRY("compact",timer.elapsed());

  DRAY_LOG_CLOSE();
  return partials;
}

// ------------------------------------------------------------------------
struct IntegratePartialsFunctor
{
  Array<Ray> *m_rays;
  Array<PointLight> m_lights;
  ColorMap &m_color_map;
  Float m_samples;
  AABB<3> m_bounds;
  bool m_use_lighting;
  Array<VolumePartial> m_partials;
  IntegratePartialsFunctor(Array<Ray> *rays,
                           Array<PointLight> &lights,
                           ColorMap &color_map,
                           Float samples,
                           AABB<3> bounds,
                           bool use_lighting)
    :
      m_rays(rays),
      m_lights(lights),
      m_color_map(color_map),
      m_samples(samples),
      m_bounds(bounds),
      m_use_lighting(use_lighting)

  {
  }

  template<typename MeshType, typename FieldType>
  void operator()(MeshType &mesh, FieldType &field)
  {
    m_partials = detail::integrate_partials(mesh,
                                            field,
                                            *m_rays,
                                            m_lights,
                                            m_samples,
                                            m_bounds,
                                            m_color_map,
                                            m_use_lighting);
  }
};

} // namespace detail

// ------------------------------------------------------------------------
Volume::Volume(Collection &collection)
  : m_samples(100),
    m_collection(collection),
    m_use_lighting(true),
    m_active_domain(0)
{
  // add some default alpha
  ColorTable table = m_color_map.color_table();
  table.add_alpha(0.1000, .0f);
  table.add_alpha(1.0000, .7f);
  m_color_map.color_table(table);
  m_bounds = m_collection.bounds();
}

// ------------------------------------------------------------------------
Volume::~Volume()
{
}

// ------------------------------------------------------------------------
void
Volume::input(Collection &collection)
{
  m_collection = collection;
  m_active_domain = 0;
  m_bounds = m_collection.bounds();
}

// ------------------------------------------------------------------------

void
Volume::field(const std::string field)
{
  m_field = field;
  m_field_range = m_collection.range(m_field);
}

std::string
Volume::field() const
{
  return m_field;
}

ColorMap& Volume::color_map()
{
  return m_color_map;
}
// ------------------------------------------------------------------------

Array<VolumePartial>
Volume::integrate(Array<Ray> &rays, Array<PointLight> &lights)
{
  Collection collection = m_collection;


  DataSet data_set = m_collection.domain(m_active_domain);

  if(m_field == "")
  {
    DRAY_ERROR("Field never set");
  }
  if(!m_color_map.range_set())
  {
    m_color_map.scalar_range(m_field_range);
  }

  Mesh *mesh = data_set.mesh();
  Field *field = data_set.field(m_field);

  detail::IntegratePartialsFunctor func(&rays,
                                        lights,
                                        m_color_map,
                                        m_samples,
                                        m_bounds,
                                        m_use_lighting);
  dispatch_3d(mesh, field, func);
  return func.m_partials;
}
// ------------------------------------------------------------------------

void Volume::samples(int32 num_samples)
{
  m_samples = num_samples;
}

// ------------------------------------------------------------------------

void Volume::use_lighting(bool do_it)
{
  m_use_lighting = do_it;
}


// ------------------------------------------------------------------------

void Volume::save(const std::string name,
                  Array<VolumePartial> partials,
                  const int32 width,
                  const int32 height)
{
  Framebuffer fb(width, height);
  fb.clear();
  Array<Vec<float32,4>> colors = fb.colors();
  Vec<float32,4> *color_ptr = colors.get_host_ptr();
  const int size = partials.size();

  VolumePartial *partials_ptr = partials.get_host_ptr();

  if(size > 0)
  {
    VolumePartial p = partials_ptr[0];
    Vec<float32,4> color = p.m_color;
    int32 current_pixel = -1;

    for(int i = 0; i < size; ++i)
    {
      VolumePartial p = partials_ptr[i];
      if(current_pixel != p.m_pixel_id)
      {
        if(current_pixel != -1)
        {
          color_ptr[current_pixel] = color;
        }

        color = p.m_color;
        current_pixel = p.m_pixel_id;
      }
      else
      {
        pre_mult_alpha_blend_host(color, p.m_color);
      }

      if(i == size - 1)
      {
        color_ptr[current_pixel] = color;
      }

    }
  }
  fb.composite_background();
  fb.save(name);

}

void Volume::active_domain(int32 domain_index)
{
  m_active_domain = domain_index;
}

int32 Volume::active_domain()
{
  return m_active_domain;
}

int32 Volume::num_domains()
{
  return m_collection.local_size();
}

// ------------------------------------------------------------------------
} // namespace dray
