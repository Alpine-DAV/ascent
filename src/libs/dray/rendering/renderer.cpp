// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/rendering/screen_annotator.hpp>
#include <dray/rendering/world_annotator.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>

#include <apcomp/compositor.hpp>
#include <apcomp/partial_compositor.hpp>

#include <memory>
#include <vector>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

namespace detail
{

// one ranks says yes
bool someone_agrees(bool vote)
{
  bool agreement = vote;
#ifdef DRAY_MPI_ENABLED
  int local_boolean = vote ? 1 : 0;
  int global_boolean;

  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean > 0)
  {
    agreement = true;
  }
#endif
  return agreement;
}

void convert_partials(std::vector<Array<VolumePartial>> &input,
                      std::vector<std::vector<apcomp::VolumePartial<float>>> &output)
{
  size_t total_size = 0;
  const int in_size = input.size();
  std::vector<size_t> offsets;
  offsets.resize(in_size);
  output.resize(1);

  for(size_t i = 0; i< in_size; ++i)
  {
    offsets[i] = total_size;
    total_size += input[i].size();
  }

  output[0].resize(total_size);

  for(size_t a = 0; a < in_size; ++a)
  {
    const VolumePartial *partial_ptr = input[a].get_host_ptr_const();
    const size_t offset = offsets[a];
    const size_t size = input[a].size();
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      const size_t index = offset + i;
      const VolumePartial p = partial_ptr[i];
      output[0][index].m_pixel_id = p.m_pixel_id;
      output[0][index].m_depth = p.m_depth;

      output[0][index].m_pixel[0] = p.m_color[0];
      output[0][index].m_pixel[1] = p.m_color[1];
      output[0][index].m_pixel[2] = p.m_color[2];

      output[0][index].m_alpha = p.m_color[3];
    }
  }

}

void
partials_to_framebuffer(const std::vector<apcomp::VolumePartial<float>> &input,
                        Framebuffer &fb,
                        bool blend)
{
  const int32 size = input.size();
  Vec<float32,4> *colors = fb.colors().get_host_ptr();
  float32 *depths = fb.depths().get_host_ptr();

  if(blend)
  {
    // framebuffer is the surface and is behind the volume
#ifdef DRAY_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      const int id = input[i].m_pixel_id;
      float32 opacity = 1.f - input[i].m_alpha;
      colors[id][0] = input[i].m_pixel[0] + opacity * colors[id][0];
      colors[id][1] = input[i].m_pixel[1] + opacity * colors[id][1];
      colors[id][2] = input[i].m_pixel[2] + opacity * colors[id][2];
      colors[id][3] = input[i].m_alpha + opacity * colors[id][3];
      depths[id] = input[i].m_depth;
    }
  }
  else
  {
#ifdef DRAY_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      const int id = input[i].m_pixel_id;
      colors[id][0] = input[i].m_pixel[0];
      colors[id][1] = input[i].m_pixel[1];
      colors[id][2] = input[i].m_pixel[2];
      colors[id][3] = input[i].m_alpha;
      depths[id] = input[i].m_depth;
    }
  }
}

PointLight default_light(Camera &camera)
{
  Vec<float32,3> look_at = camera.get_look_at();
  Vec<float32,3> pos = camera.get_pos();
  Vec<float32,3> up = camera.get_up();
  up.normalize();
  Vec<float32,3> look = look_at - pos;
  float32 mag = look.magnitude();
  Vec<float32,3> right = cross (look, up);
  right.normalize();

  Vec<float32, 3> miner_up = cross (right, look);
  miner_up.normalize();

  Vec<float32, 3> light_pos = pos + .1f * mag * miner_up;
  PointLight light;
  light.m_pos = light_pos;
  return light;
}

} // namespace detail

Renderer::Renderer()
  : m_volume(nullptr),
    m_use_lighting(true),
    m_world_annotations(false),
    m_color_bar(true),
    m_triad(false),
    m_max_color_bars(2)
{
}

void Renderer::clear()
{
  m_traceables.clear();
}

void Renderer::world_annotations(bool on)
{
  m_world_annotations = on;
}

void Renderer::color_bar(bool on)
{
  m_color_bar = on;
}

void Renderer::triad(bool on)
{
  m_triad = on;
}

void Renderer::clear_lights()
{
  m_lights.clear();
}

void Renderer::add_light(const PointLight &light)
{
  m_lights.push_back(light);
}

void Renderer::use_lighting(bool use_it)
{
  m_use_lighting = use_it;
}

void Renderer::add(std::shared_ptr<Traceable> traceable)
{
  m_traceables.push_back(traceable);
}


void Renderer::volume(std::shared_ptr<Volume> volume)
{
  m_volume = volume;
}

AABB<3> Renderer::bounds()
{
  AABB<3> scene_bounds;
  for(auto &tracable :  m_traceables)
  {
    scene_bounds.include(tracable->collection().bounds());
  }
  return scene_bounds;
}

Framebuffer Renderer::render(Camera &camera)
{
  DRAY_LOG_OPEN("render");
  Array<Ray> rays;
  camera.create_rays (rays);

  std::vector<std::string> field_names;
  std::vector<ColorMap> color_maps;

  Framebuffer framebuffer (camera.get_width(), camera.get_height());
  framebuffer.clear ();

  Array<PointLight> lights;
  if(m_lights.size() > 0)
  {
    lights.resize(m_lights.size());
    PointLight* light_ptr = lights.get_host_ptr();
    for(int i = 0; i < m_lights.size(); ++i)
    {
      light_ptr[i] = m_lights[i];
    }
  }
  else
  {
    lights.resize(1);
    PointLight light = detail::default_light(camera);
    PointLight* light_ptr = lights.get_host_ptr();
    light_ptr[0] = light;
  }

  const int32 size = m_traceables.size();

  bool need_composite = false;
  for(int i = 0; i < size; ++i)
  {
    const int domains = m_traceables[i]->num_domains();
    for(int d = 0; d < domains; ++d)
    {
      m_traceables[i]->active_domain(d);
      Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
      Array<Fragment> fragments = m_traceables[i]->fragments(hits);
      if(m_use_lighting)
      {
        m_traceables[i]->shade(rays, hits, fragments, lights, framebuffer);
      }
      else
      {
        m_traceables[i]->shade(rays, hits, fragments, framebuffer);
      }

      ray_max(rays, hits);
    }
    // we just did some rendering so we need to composite
    need_composite = true;

    // get stuff for annotations
    field_names.push_back(m_traceables[i]->field());
    color_maps.push_back(m_traceables[i]->color_map());
  }

  // Do world objects if any
  if(m_world_annotations)
  {
      WorldAnnotator world_annotator(this->bounds());
      world_annotator.render(framebuffer, rays, camera);
  }

  // we only need to synch depths if we are going to
  // perform volume rendering
  bool synch_depths = m_volume != nullptr;

  // not all ranks might have data so we need to
  // all agree to do things that might involve mpi
  if(detail::someone_agrees(need_composite))
  {
    composite(rays, camera, framebuffer, synch_depths);
  }

  if(m_volume != nullptr)
  {
    Timer timer;
    const int domains = m_volume->num_domains();
    std::vector<Array<VolumePartial>> domain_partials;
    for(int d = 0; d < domains; ++d)
    {
      m_volume->active_domain(d);
      Array<VolumePartial> partials = m_volume->integrate(rays, lights);
      domain_partials.push_back(partials);
    }

    DRAY_LOG_ENTRY("volume_total",timer.elapsed());
    field_names.push_back(m_volume->field());
    color_maps.push_back(m_volume->color_map());

    std::vector<std::vector<apcomp::VolumePartial<float>>> c_partials;
    detail::convert_partials(domain_partials, c_partials);

    std::vector<apcomp::VolumePartial<float>> result;
    apcomp::PartialCompositor<apcomp::VolumePartial<float>> compositor;
    compositor.composite(c_partials, result);
    if(dray::mpi_rank() == 0)
    {

      detail::partials_to_framebuffer(result,
                                      framebuffer,
                                      need_composite);
    }

  }

  if (dray::mpi_rank() == 0)
  {
    Timer timer;
    ScreenAnnotator annot;
    if (m_color_bar)
    {
      annot.max_color_bars(m_max_color_bars);
      annot.draw_color_bars(framebuffer, field_names, color_maps);
    }
    if (m_triad)
    {
      // we want it to be in the bottom left corner
      // so 1/10th of the width and height gets converted into
      // screen space coords from -1 to 1
      Vec<float32, 2> SS_triad_pos = {{0.1 * 2.0 - 1.0, 0.1 * 2.0 - 1.0}};
      float32 distance_from_triad = 15.f;
      annot.draw_triad(framebuffer, SS_triad_pos, distance_from_triad, camera);
    }
    DRAY_LOG_ENTRY("screen_annotations",timer.elapsed());
  }

  DRAY_LOG_CLOSE();

  return framebuffer;
}

void Renderer::composite(Array<Ray> &rays,
                         Camera &camera,
                         Framebuffer &framebuffer,
                         bool synch_depths) const
{
#ifdef DRAY_MPI_ENABLED
  apcomp::Compositor comp;
  comp.SetCompositeMode(apcomp::Compositor::CompositeMode::Z_BUFFER_SURFACE_WORLD);

  const float32 *cbuffer =
    reinterpret_cast<const float*>(framebuffer.colors().get_host_ptr());
  comp.AddImage(cbuffer,
                framebuffer.depths().get_host_ptr_const(),
                camera.get_width(),
                camera.get_height());

  // valid only on rank 0
  apcomp::Image result = comp.Composite();
  float32 * depth_ptr = framebuffer.depths().get_host_ptr();
  const int image_size = camera.get_width() * camera.get_height();
  // copy the result back to the framebuffer
  const size_t bytes = image_size * sizeof(float32);
  if(dray::mpi_rank() == 0)
  {
    // write back the result depth buffer
    memcpy ( depth_ptr, &result.m_depths[0], bytes);
    Vec<float32, 4> *color_ptr = framebuffer.colors().get_host_ptr();
    // copy the result colors back to the framebuffer
    RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, image_size), [=] DRAY_CPU_LAMBDA (int32 i)
    {
      const int32 offset = i * 4;
      Vec<float32, 4> color;
      for(int c = 0; c < 4; ++c)
      {
        color[c] = float32(result.m_pixels[offset + c]) / 255.f;
      }
      color_ptr[i] = color;
    });
  }

  if(synch_depths)
  {
    // tell everyone else about the current depth
    // this will write direcly back to the framebuffer
    MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());

    MPI_Bcast( depth_ptr, image_size, MPI_FLOAT, 0, mpi_comm);

    // everyone now needs to update the rays max depth to the updated
    // depth values
    const int32 size = rays.size();
    Ray *ray_ptr = rays.get_device_ptr();
    float32 *d_depth_ptr = framebuffer.depths().get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
    {
      ray_ptr[i].m_far = d_depth_ptr[ray_ptr[i].m_pixel_id];
    });

  }
#else
  // nothing to do. We have already composited via ray_max
#endif
}

void Renderer::max_color_bars(const int32 max_bars)
{
  // limits will be enforced in the annotator
  m_max_color_bars = max_bars;
}

} // namespace dray
