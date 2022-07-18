// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_renderer.hpp>

#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/filters/vector_component.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/policies.hpp>
#include <dray/utils/string_utils.hpp>
#include <dray/utils/mpi_utils.hpp>

#include <memory>
#include <vector>
#include <set>

#include <apcomp/scalar_compositor.hpp>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

namespace
{

void copy_float32_depths(const float32 *in_depths, float32 *out_depths, const int32 size)
{
  memcpy(out_depths, in_depths, sizeof(float32) * size);
}

void copy_float32_depths(const float32 *in_depths, float64 *out_depths, const int32 size)
{
  // in depths is always on the cpu, so we can't use raja here
  for(int32 i = 0; i < size; ++i)
  {
    out_depths[i] = static_cast<float32>(in_depths[i]);
  }
}

Array<float32> get_float32_depths(Array<float32> depths)
{
  return depths;
}

Array<float32> get_float32_depths(Array<float64> depths)
{
  Array<float32> fdepths;
  fdepths.resize(depths.size());

  const float64 *d_ptr = depths.get_device_ptr_const();
  float32 *f_ptr = fdepths.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, depths.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    f_ptr[ii] = static_cast<float32>(d_ptr[ii]);
  });
  return fdepths;
}

void calc_offsets(Collection &collection, std::vector<int32> &offsets)
{
  const int size = collection.local_size();
  offsets.resize(size);
  int total_cells = 0;
  for(int i = 0; i < size; ++i)
  {
    offsets[i] = total_cells;

    total_cells += collection.domain(i).mesh()->cells();
  }

#ifdef DRAY_MPI_ENABLED
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  int *global_counts = new int[dray::mpi_size()];
  MPI_Allgather(&total_cells, 1, MPI_INT,global_counts, 1, MPI_INT, comm);
  int rank = dray::mpi_rank();
  int global_offset = 0;

  if(global_counts[dray::mpi_size() - 1] < 0)
  {
    DRAY_ERROR("Zone count overflow: blame matt");
  }

  for(int i = 0; i < rank; ++i)
  {
    global_offset += global_counts[i];
  }

  delete[] global_counts;

  for(int i = 0; i < size; ++i)
  {
    offsets[i] += global_offset;
  }
#endif
}

ScalarBuffer convert(apcomp::ScalarImage &image, std::vector<std::string> &names)
{
  const int num_fields = names.size();

  const int dx  = image.m_bounds.m_max_x - image.m_bounds.m_min_x + 1;
  const int dy  = image.m_bounds.m_max_y - image.m_bounds.m_min_y + 1;
  const int size = dx * dy;

  ScalarBuffer result(dx, dy, nan<Float>());

  std::vector<Float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    result.add_field(names[i]);
    Float * buffer = result.m_scalars[names[i]].get_host_ptr();
    buffers.push_back(buffer);
  }

  int32 *zone_id_ptr = result.m_zone_ids.get_host_ptr();

  // this is fields + zone_id for each pixel
  const unsigned char *loads = &image.m_payloads[0];
  const size_t payload_size = image.m_payload_bytes;

#ifdef DRAY_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(Float);
      memcpy(&buffers[i][x], loads + offset, sizeof(Float));
    }

    const size_t zone_offset = x * payload_size + num_fields * sizeof(Float);
    memcpy(zone_id_ptr+x, loads + zone_offset, sizeof(int32));
  }

  result.m_depths.resize(size);
  Float* dbuffer = result.m_depths.get_host_ptr();
  copy_float32_depths(&image.m_depths[0], dbuffer, size);

  return result;
}

apcomp::ScalarImage *
convert(ScalarBuffer &result, const std::vector<std::string> &names)
{
  const int num_fields = names.size();
  // payload is fields + zone id
  const int payload_size = num_fields * sizeof(Float) + sizeof(int32);

  apcomp::Bounds bounds;
  bounds.m_min_x = 1;
  bounds.m_min_y = 1;
  bounds.m_max_x = result.m_width;
  bounds.m_max_y = result.m_height;
  const size_t size = result.m_width * result.m_height;

  apcomp::ScalarImage *image = new apcomp::ScalarImage(bounds, payload_size);
  unsigned char *loads = &image->m_payloads[0];

  Array<float32> f_depths = get_float32_depths(result.m_depths);
  const float32 * dbuffer = f_depths.get_host_ptr();
  memcpy(&image->m_depths[0], dbuffer, sizeof(float32) * size);

  // copy scalars into payload
  std::vector<Float*> buffers;
  for(auto field : names)
  {
    Float* buffer = result.m_scalars[field].get_host_ptr();
    buffers.push_back(buffer);
  }

  const int32 *zone_id_ptr = result.m_zone_ids.get_host_ptr_const();

#ifdef DRAY_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(Float);
      memcpy(loads + offset, &buffers[i][x], sizeof(Float));
    }
    const size_t zone_id_offset = x * payload_size + num_fields * sizeof(Float);
    memcpy(loads + zone_id_offset, zone_id_ptr+x, sizeof(int32));
  }
  return image;
}


} // namespace


ScalarRenderer::ScalarRenderer()
  : m_traceable(nullptr)
{
}

ScalarRenderer::ScalarRenderer(std::shared_ptr<Traceable> traceable)
{
  set(traceable);
}

void ScalarRenderer::set(std::shared_ptr<Traceable> traceable)
{
  m_traceable = traceable;
  calc_offsets(m_traceable->collection(), m_offsets);
}

void ScalarRenderer::field_names(const std::vector<std::string> &field_names)
{
  m_field_names = field_names;
}

void
ScalarRenderer::decompose_vectors()
{
  Collection input = m_traceable->collection();
  std::set<std::string> vector_fields;

  std::set<std::string> names;
  // figure out the vector fields in the list
  for(int32 i = 0; i < input.local_size(); ++i)
  {
    DataSet data_set = input.domain(i);
    for(auto &name : m_field_names)
    {
      if(data_set.has_field(name))
      {
        if(data_set.field(name)->components() == 2 || data_set.field(name)->components() == 3)
        {
          vector_fields.insert(name);
        }
        else
        {
          // this is a scalar field, so
          // just add it to the list of names
          names.insert(name);
        }
      }
    }
  }

  // make sure everyone has the same list
  gather_strings(vector_fields);
  gather_strings(names);

  if(vector_fields.size() > 0)
  {
    std::vector<std::string> suffix({"_x", "_y", "_z"});
    // decompose the vectors and add the names
    for(auto &vector : vector_fields)
    {
      input = VectorComponent::decompose_field(input, vector);

      // find which were added
      for(int32 d = 0; d < input.local_size(); ++d)
      {
        DataSet data_set = input.domain(d);
        // look for the new names in the decomped result
        for(int32 i = 0; i < suffix.size(); ++i)
        {
          if(data_set.has_field(vector+suffix[i]))
          {
            names.insert(vector+suffix[i]);
          }
        }
      }
    }

    // make sure everyone has the same names.
    gather_strings(names);

    // set the traceable collection to the one with the decomposed fields
    m_traceable->input(input);
    m_actual_field_names.resize(names.size());
    std::copy(names.begin(), names.end(), m_actual_field_names.begin());
  }
  else
  {
    m_actual_field_names = m_field_names;
  }
}


void
ScalarRenderer::render(Array<Ray> &rays, ScalarBuffer &scalar_buffer)
{
  // we only handle scalars so if they asked for a vector field
  // then we have to decompose them
  decompose_vectors();

  const int32 buffer_size = scalar_buffer.size();
  const int32 field_size = m_actual_field_names.size();

  //scalar_buffer.m_depths.resize(buffer_size);

  const int domains = m_traceable->num_domains();

  for(int d = 0; d < domains; ++d)
  {
    DRAY_INFO("Tracing scalar domain "<<d);
    m_traceable->active_domain(d);
    const int domain_offset = m_offsets[d];
    
  
    Array<RayHit> hits = m_traceable->nearest_hit(rays);
    Float *depth_ptr = scalar_buffer.m_depths.get_device_ptr();
    int32 *zone_id_ptr = scalar_buffer.m_zone_ids.get_device_ptr();

    const Ray *ray_ptr = rays.get_device_ptr_const ();
    const RayHit *hit_ptr = hits.get_device_ptr_const ();

    RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
    {
      const RayHit &hit = hit_ptr[ii];
      const Ray &ray = ray_ptr[ii];
      if (hit.m_hit_idx > -1)
      {
        const int32 pid = ray.m_pixel_id;
        depth_ptr[pid] = hit.m_dist;
        zone_id_ptr[pid] = hit.m_hit_idx + domain_offset;
      }
    });

    for(int32 i = 0; i < field_size; ++i)
    {
      std::string field = m_actual_field_names[i];
      m_traceable->field(field);
      Array<Fragment> fragments = m_traceable->fragments(hits);
      if(!scalar_buffer.has_field(field))
      {
        scalar_buffer.add_field(field);
      }
      Array<Float> buffer = scalar_buffer.m_scalars[field];;
      Float *buffer_ptr = buffer.get_device_ptr();

      const Fragment *frag_ptr = fragments.get_device_ptr_const ();
      RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
      {
        const RayHit &hit = hit_ptr[ii];
        const Fragment &frag = frag_ptr[ii];
        const Ray &ray = ray_ptr[ii];

        if (hit.m_hit_idx > -1)
        {
          const int32 pid = ray.m_pixel_id;
          buffer_ptr[pid] = frag.m_scalar;
        }

      });

    }

    ray_max(rays, hits);
  }


#ifdef DRAY_MPI_ENABLED
  apcomp::PayloadCompositor compositor;
  apcomp::ScalarImage *pimage = convert(scalar_buffer, m_actual_field_names);
  compositor.AddImage(*pimage);
  delete pimage;

  ScalarBuffer final_result;
  // only valid on rank 0
  apcomp::ScalarImage final_image = compositor.Composite();
  if(dray::mpi_rank() == 0)
  {
    final_result = convert(final_image, m_actual_field_names);
  }
  scalar_buffer = final_result;
#else
  // we have composited locally so there is nothing to do
#endif
}

ScalarBuffer
ScalarRenderer::render(PlaneDetector &detector)
{
  const int32 p_width = detector.m_x_res;
  const int32 p_height = detector.m_y_res;
  const Float width = detector.m_plane_width;
  const Float height = detector.m_plane_height;

  const Float dx = width / Float(p_width);
  const Float dy = height / Float(p_height);


  // TODO: Float
  ScalarBuffer scalar_buffer(p_width,
                             p_height,
                             nan<Float>());

  Vec<Float, 3> view = detector.m_view;
  Vec<Float, 3> up = detector.m_up;

  view.normalize();
  up.normalize();

  // create the orthogal basis vectors
  Vec<Float, 3> rx = cross(view, up);
  Vec<Float, 3> ry = cross (rx, view);

  const Vec<Float, 3> center = detector.m_center;
  // bottom left pixel origin
  const Vec<Float, 3> origin
    = center - rx * (width + dx) * 0.5 - ry * (height + dy) * 0.5;

  Array<Ray> rays;
  rays.resize(p_width * p_height);

  Ray * ray_ptr = rays.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    Ray ray;
    ray.m_dir = view;
    ray.m_pixel_id = ii;

    int32 i = int32(ii) % p_width;
    int32 j = int32(ii) / p_width;

    ray.m_orig = origin + rx * Float(i) * dx + ry * Float(j) * dy;
    ray.m_near = 0.f;
    ray.m_far = infinity<Float>();

    ray_ptr[ii] = ray;
  });

  render(rays, scalar_buffer);

  return scalar_buffer;
}

ScalarBuffer
ScalarRenderer::render(Camera &camera)
{
  if(m_traceable == nullptr)
  {
    DRAY_ERROR("ScalarRenderer: traceable never set");
  }

  Array<Ray> rays;
  camera.create_rays (rays);

  ScalarBuffer scalar_buffer(camera.get_width(),
                             camera.get_height(),
                             nan<Float>());

  render(rays, scalar_buffer);
  return scalar_buffer;
}

} // namespace dray
