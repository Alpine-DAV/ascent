#include "VolumeRenderer.hpp"

#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/compositing/Compositor.hpp>
#include <vtkh/Logger.hpp>

#include <vtkm/rendering/CanvasRayTracer.h>

#include <memory>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif


#include <vtkm/cont/ColorTable.h>
#include <vtkm/rendering/ConnectivityProxy.h>
#include <vtkh/compositing/PartialCompositor.hpp>
#include <vtkm/rendering/raytracing/VolumeRendererStructured.h>
#include <vtkm/rendering/raytracing/RayOperations.h>
#include <vtkm/rendering/raytracing/Camera.h>

#include <vtkh/compositing/VolumePartial.hpp>

#define VTKH_OPACITY_CORRECTION 10.f

namespace vtkh {

namespace detail
{

struct VisOrdering
{
  int m_rank;
  int m_domain_index;
  int m_order;
  float m_minz;
};

struct DepthOrder
{
  inline bool operator()(const VisOrdering &lhs, const VisOrdering &rhs)
  {
    return lhs.m_minz < rhs.m_minz;
  }
};

struct RankOrder
{
  inline bool operator()(const VisOrdering &lhs, const VisOrdering &rhs)
  {
    if(lhs.m_rank < rhs.m_rank)
    {
      return true;
    }
    else if(lhs.m_rank == rhs.m_rank)
    {
      return lhs.m_domain_index < rhs.m_domain_index;
    }
    return false;
  }
};

vtkm::cont::ArrayHandle<vtkm::Vec4f_32>
convert_table(const vtkm::cont::ColorTable& colorTable)
{

  constexpr vtkm::Float32 conversionToFloatSpace = (1.0f / 255.0f);

  vtkm::cont::ArrayHandle<vtkm::Vec4ui_8> temp;

  {
    vtkm::cont::ScopedRuntimeDeviceTracker tracker(vtkm::cont::DeviceAdapterTagSerial{});
    colorTable.Sample(1024, temp);
  }

  vtkm::cont::ArrayHandle<vtkm::Vec4f_32> color_map;
  color_map.Allocate(1024);
  auto portal = color_map.WritePortal();
  auto colorPortal = temp.ReadPortal();
  for (vtkm::Id i = 0; i < 1024; ++i)
  {
    auto color = colorPortal.Get(i);
    vtkm::Vec4f_32 t(color[0] * conversionToFloatSpace,
                     color[1] * conversionToFloatSpace,
                     color[2] * conversionToFloatSpace,
                     color[3] * conversionToFloatSpace);
    portal.Set(i, t);
  }
  return color_map;
}

class VolumeWrapper
{
protected:
  vtkm::cont::DataSet m_data_set;
  vtkm::Range m_scalar_range;
  std::string m_field_name;
  vtkm::Float32 m_sample_dist;
  vtkm::cont::ArrayHandle<vtkm::Vec4f_32> m_color_map;
public:
  VolumeWrapper() = delete;

  VolumeWrapper(vtkm::cont::DataSet &data_set)
   : m_data_set(data_set)
  {
  }

  virtual ~VolumeWrapper()
  {

  }

  void sample_distance(const vtkm::Float32 &distance)
  {
    m_sample_dist = distance;
  }

  void field(const std::string &field_name)
  {
    m_field_name = field_name;
  }

  void scalar_range(vtkm::Range &range)
  {
    m_scalar_range = range;
  }

  void color_map(vtkm::cont::ArrayHandle<vtkm::Vec4f_32> &color_map)
  {
    m_color_map = color_map;
  }

  virtual void
  render(const vtkm::rendering::Camera &camera,
         vtkm::rendering::CanvasRayTracer &canvas,
         std::vector<VolumePartial<float>> &partials) = 0;

};

void vtkm_to_partials(vtkm::rendering::PartialVector32 &vtkm_partials,
                      std::vector<VolumePartial<float>> &partials)
{
  const int num_vecs = vtkm_partials.size();
  std::vector<int> offsets;
  offsets.reserve(num_vecs);

  int total_size = 0;
  for(int i = 0; i < num_vecs; ++i)
  {
    const int size = vtkm_partials[i].PixelIds.GetNumberOfValues();
    offsets.push_back(total_size);
    total_size += size;
  }

  partials.resize(total_size);

  for(int i = 0; i < num_vecs; ++i)
  {
    const int size = vtkm_partials[i].PixelIds.GetNumberOfValues();
    auto pixel_ids = vtkm_partials[i].PixelIds.ReadPortal();
    auto distances = vtkm_partials[i].Distances.ReadPortal();
    auto colors = vtkm_partials[i].Buffer.Buffer.ReadPortal();

    const int offset = offsets[i];
#ifdef VTKH_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int p = 0; p < size; ++p)
    {
      VolumePartial<float> &partial = partials[offset+p];
      partial.m_pixel[0] = colors.Get(p*4 + 0);
      partial.m_pixel[1] = colors.Get(p*4 + 1);
      partial.m_pixel[2] = colors.Get(p*4 + 2);
      partial.m_alpha = colors.Get(p*4 + 3);
      partial.m_pixel_id = pixel_ids.Get(p);
      partial.m_depth = distances.Get(p);
    }
  }
}

class UnstructuredWrapper : public VolumeWrapper
{
  vtkm::rendering::ConnectivityProxy m_tracer;
public:
  UnstructuredWrapper(vtkm::cont::DataSet &data_set)
    : VolumeWrapper(data_set),
      m_tracer(data_set)
  {
  }

  virtual void
  render(const vtkm::rendering::Camera &camera,
         vtkm::rendering::CanvasRayTracer &canvas,
         std::vector<VolumePartial<float>> &partials) override
  {
    const vtkm::cont::CoordinateSystem &coords = m_data_set.GetCoordinateSystem();

    vtkm::rendering::raytracing::Camera rayCamera;
    vtkm::rendering::raytracing::Ray<vtkm::Float32> rays;
    vtkm::Int32 width = (vtkm::Int32) canvas.GetWidth();
    vtkm::Int32 height = (vtkm::Int32) canvas.GetHeight();

    rayCamera.SetParameters(camera, width, height);

    rayCamera.CreateRays(rays, coords.GetBounds());
    rays.Buffers.at(0).InitConst(0.f);
    vtkm::rendering::raytracing::RayOperations::MapCanvasToRays(rays, camera, canvas);

    m_tracer.SetSampleDistance(m_sample_dist);
    m_tracer.SetColorMap(m_color_map);
    m_tracer.SetScalarField(m_field_name);
    m_tracer.SetScalarRange(m_scalar_range);

    vtkm::rendering::PartialVector32 vtkm_partials;
    vtkm_partials = m_tracer.PartialTrace(rays);

    vtkm_to_partials(vtkm_partials, partials);
  }

};

class StructuredWrapper : public VolumeWrapper
{
public:
  StructuredWrapper(vtkm::cont::DataSet &data_set)
    : VolumeWrapper(data_set)
  {
  }
  virtual void
  render(const vtkm::rendering::Camera &camera,
         vtkm::rendering::CanvasRayTracer &canvas,
         std::vector<VolumePartial<float>> &partials) override
  {
    const vtkm::cont::DynamicCellSet &cellset = m_data_set.GetCellSet();
    const vtkm::cont::Field &field = m_data_set.GetField(m_field_name);
    const vtkm::cont::CoordinateSystem &coords = m_data_set.GetCoordinateSystem();

    vtkm::rendering::raytracing::Camera rayCamera;
    vtkm::rendering::raytracing::Ray<vtkm::Float32> rays;
    vtkm::Int32 width = (vtkm::Int32) canvas.GetWidth();
    vtkm::Int32 height = (vtkm::Int32) canvas.GetHeight();
    rayCamera.SetParameters(camera, width, height);

    rayCamera.CreateRays(rays, coords.GetBounds());
    rays.Buffers.at(0).InitConst(0.f);
    vtkm::rendering::raytracing::RayOperations::MapCanvasToRays(rays, camera, canvas);

    vtkm::rendering::raytracing::VolumeRendererStructured tracer;
    tracer.SetSampleDistance(m_sample_dist);
    tracer.SetData(coords,
                   field,
                   cellset.Cast<vtkm::cont::CellSetStructured<3>>(),
                   m_scalar_range);
    tracer.SetColorMap(m_color_map);


    tracer.Render(rays);

    // Convert the rays to partial composites
    const int ray_size = rays.NumRays;
    // partials use the max distance
    auto depths = rays.MaxDistance.ReadPortal();
    auto pixel_ids = rays.PixelIdx.ReadPortal();
    auto colors = rays.Buffers.at(0).Buffer.ReadPortal();

    // TODO: better way? we could do this in parallel if we
    // don't check the alpha
    partials.reserve(ray_size);
    for(int i = 0; i < ray_size; ++i)
    {
      const int offset = i * 4;
      float alpha = colors.Get(offset + 3);
      if(alpha < 0.001f) continue;
      VolumePartial<float> partial;
      partial.m_pixel[0] = colors.Get(offset + 0);
      partial.m_pixel[1] = colors.Get(offset + 1);
      partial.m_pixel[2] = colors.Get(offset + 2);
      partial.m_alpha = alpha;
      partial.m_pixel_id = pixel_ids.Get(i);
      partial.m_depth = depths.Get(i);
      partials.push_back(std::move(partial));
    }
  }
};

void partials_to_canvas(std::vector<VolumePartial<float>> &partials,
                        const vtkm::rendering::Camera &camera,
                        vtkm::rendering::CanvasRayTracer &canvas)
{

  // partial depths are in world space but the canvas depths
  // are in image space. We have to find the intersection
  // point to project it into image space to get the correct
  // depths for annotations
  vtkm::Id width = canvas.GetWidth();
  vtkm::Id height = canvas.GetHeight();
  vtkm::Matrix<vtkm::Float32, 4, 4> projview =
    vtkm::MatrixMultiply(camera.CreateProjectionMatrix(width, height),
                         camera.CreateViewMatrix());

  const vtkm::Vec3f_32 origin = camera.GetPosition();

  float fov_y = camera.GetFieldOfView();
  float fov_x = fov_y;
  if(width != height)
  {
    vtkm::Float32 fovyRad = fov_y * vtkm::Pi_180f();
    vtkm::Float32 verticalDistance = vtkm::Tan(0.5f * fovyRad);
    vtkm::Float32 aspectRatio = vtkm::Float32(width) / vtkm::Float32(height);
    vtkm::Float32 horizontalDistance = aspectRatio * verticalDistance;
    vtkm::Float32 fovxRad = 2.0f * vtkm::ATan(horizontalDistance);
    fov_x = fovxRad / vtkm::Pi_180f();
  }

  vtkm::Vec3f_32 look = camera.GetLookAt() - origin;
  vtkm::Normalize(look);
  vtkm::Vec3f_32 up = camera.GetViewUp();

  const vtkm::Float32 thx = tanf((fov_x * vtkm::Pi_180f()) * .5f);
  const vtkm::Float32 thy = tanf((fov_y * vtkm::Pi_180f()) * .5f);
  vtkm::Vec3f_32 ru = vtkm::Cross(look, up);

  vtkm::Normalize(ru);
  vtkm::Vec3f_32 rv = vtkm::Cross(ru, look);
  vtkm::Normalize(rv);
  vtkm::Vec3f_32 delta_x = ru * (2 * thx / (float)width);
  vtkm::Vec3f_32 delta_y = ru * (2 * thy / (float)height);

  vtkm::Float32 zoom = camera.GetZoom();
  if(zoom > 0)
  {
    delta_x[0] = delta_x[0] / zoom;
    delta_x[1] = delta_x[1] / zoom;
    delta_x[2] = delta_x[2] / zoom;
    delta_y[0] = delta_y[0] / zoom;
    delta_y[1] = delta_y[1] / zoom;
    delta_y[2] = delta_y[2] / zoom;
  }

  const int size = partials.size();
  auto colors = canvas.GetColorBuffer().WritePortal();
  auto depths = canvas.GetDepthBuffer().WritePortal();

#ifdef VTKH_OPENMP_ENABLED
  #pragma omp parallel for
#endif
  for(int p = 0; p < size; ++p)
  {
    const int pixel_id = partials[p].m_pixel_id;
    const int i = pixel_id % width;
    const int j = pixel_id / width;

    vtkm::Vec3f_32 dir;
    dir = look + delta_x * ((2.f * float(i) - float(width)) / 2.0f) +
      delta_y * ((2.f * float(j) - float(height)) / 2.0f);
    vtkm::Normalize(dir);

    const float world_depth = partials[p].m_depth;

    vtkm::Vec3f_32 pos = origin + world_depth * dir;
    vtkm::Vec4f_32 point(pos[0], pos[1], pos[2], 1.f);
    vtkm::Vec4f_32 newpoint;
    newpoint = vtkm::MatrixMultiply(projview, point);

    // don't push it all the way(.49 instead of .5) so that
    // subtle differences allow bounding box annotations don't
    // draw in front of the back
    const float image_depth = 0.5f*(newpoint[2] / newpoint[3]) + 0.49f;

    vtkm::Vec4f_32 color;
    color[0] = partials[p].m_pixel[0];
    color[1] = partials[p].m_pixel[1];
    color[2] = partials[p].m_pixel[2];
    color[3] = partials[p].m_alpha;

    vtkm::Vec4f_32 inColor = colors.Get(pixel_id);
    // We crafted the rendering so that all new colors are in front
    // of the colors that exist in the canvas
    // if transparency exists, all alphas have been pre-multiplied
    vtkm::Float32 alpha = (1.f - color[3]);
    color[0] = color[0] + inColor[0] * alpha;
    color[1] = color[1] + inColor[1] * alpha;
    color[2] = color[2] + inColor[2] * alpha;
    color[3] = inColor[3] * alpha + color[3];

    colors.Set(pixel_id, color);
    depths.Set(pixel_id, image_depth);

  }
}

} //  namespace detail

VolumeRenderer::VolumeRenderer()
{
  typedef vtkm::rendering::MapperVolume TracerType;
  m_tracer = std::make_shared<TracerType>();
  this->m_mapper = m_tracer;
  m_tracer->SetCompositeBackground(false);
  //
  // add some default opacity to the color table
  //
  m_color_table.AddPointAlpha(0.0f, .02);
  m_color_table.AddPointAlpha(.0f, .5);
  m_num_samples = 100.f;
  m_has_unstructured = false;
}

VolumeRenderer::~VolumeRenderer()
{
  ClearWrappers();
}

void
VolumeRenderer::Update()
{
  VTKH_DATA_OPEN(this->GetName());
#ifdef VTKH_ENABLE_LOGGING
  VTKH_DATA_ADD("device", GetCurrentDevice());
  long long int in_cells = this->m_input->GetNumberOfCells();
  VTKH_DATA_ADD("input_cells", in_cells);
  VTKH_DATA_ADD("input_domains", this->m_input->GetNumberOfDomains());
  int in_topo_dims;
  bool in_structured = this->m_input->IsStructured(in_topo_dims);
  if(in_structured)
  {
    VTKH_DATA_ADD("in_topology", "structured");
  }
  else
  {
    VTKH_DATA_ADD("in_topology", "unstructured");
  }
#endif

  PreExecute();
  DoExecute();
  PostExecute();

  VTKH_DATA_CLOSE();
}

void VolumeRenderer::SetColorTable(const vtkm::cont::ColorTable &color_table)
{
  m_color_table = color_table;
}

void VolumeRenderer::CorrectOpacity()
{
  const float correction_scalar = VTKH_OPACITY_CORRECTION;
  float samples = m_num_samples;

  float ratio = correction_scalar / samples;
  vtkm::cont::ColorTable corrected;
  corrected = m_color_table.MakeDeepCopy();
  int num_points = corrected.GetNumberOfPointsAlpha();
  for(int i = 0; i < num_points; i++)
  {
    vtkm::Vec<vtkm::Float64,4> point;
    corrected.GetPointAlpha(i,point);
    point[1] = 1. - vtkm::Pow((1. - point[1]), double(ratio));
    corrected.UpdatePointAlpha(i,point);
  }

  m_corrected_color_table = corrected;
}

void
VolumeRenderer::DoExecute()
{
  if(m_input->OneDomainPerRank() && !m_has_unstructured)
  {
    // Danger: this logic only works if there is exactly one per rank
    RenderOneDomainPerRank();
  }
  else
  {
    RenderMultipleDomainsPerRank();
  }
}

void
VolumeRenderer::RenderOneDomainPerRank()
{
  if(m_mapper.get() == 0)
  {
    std::string msg = "Renderer Error: no renderer was set by sub-class";
    throw Error(msg);
  }

  m_tracer->SetSampleDistance(m_sample_dist);

  int total_renders = static_cast<int>(m_renders.size());
  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  if(num_domains > 1)
  {
    throw Error("RenderOneDomainPerRank: this should never happend.");
  }
  for(int dom = 0; dom < num_domains; ++dom)
  {
    vtkm::cont::DataSet data_set;
    vtkm::Id domain_id;
    m_input->GetDomain(0, data_set, domain_id);

    if(!data_set.HasField(m_field_name))
    {
      continue;
    }

    const vtkm::cont::DynamicCellSet &cellset = data_set.GetCellSet();
    const vtkm::cont::Field &field = data_set.GetField(m_field_name);
    const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();

    if(cellset.GetNumberOfCells() == 0) continue;

    for(int i = 0; i < total_renders; ++i)
    {
      m_mapper->SetActiveColorTable(m_corrected_color_table);

      Render::vtkmCanvas &canvas = m_renders[i].GetCanvas();
      const vtkmCamera &camera = m_renders[i].GetCamera();
      m_mapper->SetCanvas(&canvas);
      m_mapper->RenderCells(cellset,
                            coords,
                            field,
                            m_corrected_color_table,
                            camera,
                            m_range);
    }
  }

  if(m_do_composite)
  {
    this->Composite(total_renders);
  }
}

void
VolumeRenderer::RenderMultipleDomainsPerRank()
{
  // We are treating this as the most general case
  // where we could have a mix of structured and
  // unstructured data sets. There are zero
  // assumptions

  // this might be smaller than the input since
  // it is possible for cell sets to be empty
  const int num_domains = m_wrappers.size();
  const int total_renders = static_cast<int>(m_renders.size());

  vtkm::cont::ArrayHandle<vtkm::Vec4f_32> color_map
    = detail::convert_table(this->m_corrected_color_table);
  vtkm::cont::ArrayHandle<vtkm::Vec4f_32> color_map2
    = detail::convert_table(this->m_color_table);

  // render/domain/result
  std::vector<std::vector<std::vector<VolumePartial<float>>>> render_partials;
  render_partials.resize(total_renders);
  for(int i = 0; i < total_renders; ++i)
  {
    render_partials[i].resize(num_domains);
  }

  for(int i = 0; i < num_domains; ++i)
  {
    detail::VolumeWrapper *wrapper = m_wrappers[i];
    wrapper->sample_distance(m_sample_dist);
    wrapper->color_map(color_map);
    wrapper->field(m_field_name);
    wrapper->scalar_range(m_range);

    for(int r = 0; r < total_renders; ++r)
    {
      Render::vtkmCanvas &canvas = m_renders[r].GetCanvas();
      const vtkmCamera &camera = m_renders[r].GetCamera();
      wrapper->render(camera, canvas, render_partials[r][i]);
    }
  }

  PartialCompositor<VolumePartial<float>> compositor;
#ifdef VTKH_PARALLEL
  compositor.set_comm_handle(GetMPICommHandle());
#endif
  // composite
  for(int r = 0; r < total_renders; ++r)
  {
    std::vector<VolumePartial<float>> res;
    compositor.composite(render_partials[r],res);
    if(vtkh::GetMPIRank() == 0)
    {
      detail::partials_to_canvas(res,
                                 m_renders[r].GetCamera(),
                                 m_renders[r].GetCanvas());
    }
  }

}

void
VolumeRenderer::PreExecute()
{
  Renderer::PreExecute();

  CorrectOpacity();

  vtkm::Vec<vtkm::Float32,3> extent;
  extent[0] = static_cast<vtkm::Float32>(this->m_bounds.X.Length());
  extent[1] = static_cast<vtkm::Float32>(this->m_bounds.Y.Length());
  extent[2] = static_cast<vtkm::Float32>(this->m_bounds.Z.Length());
  vtkm::Float32 dist = vtkm::Magnitude(extent) / m_num_samples;
  m_sample_dist = dist;
}

void
VolumeRenderer::PostExecute()
{
  // do nothing and override compositing since
  // we already did it
}

void
VolumeRenderer::SetNumberOfSamples(const int num_samples)
{
  if(num_samples < 1)
  {
    throw Error("Volume rendering samples must be greater than 0");
  }
  m_num_samples = num_samples;
}

Renderer::vtkmCanvasPtr
VolumeRenderer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

float
VolumeRenderer::FindMinDepth(const vtkm::rendering::Camera &camera,
                                 const vtkm::Bounds &bounds) const
{

  vtkm::Vec<vtkm::Float64,3> center = bounds.Center();
  vtkm::Vec<vtkm::Float64,3> fcenter;
  fcenter[0] = static_cast<vtkm::Float32>(center[0]);
  fcenter[1] = static_cast<vtkm::Float32>(center[1]);
  fcenter[2] = static_cast<vtkm::Float32>(center[2]);
  vtkm::Vec<vtkm::Float32,3> pos = camera.GetPosition();
  vtkm::Float32 dist = vtkm::Magnitude(fcenter - pos);
  return dist;
}

void
VolumeRenderer::Composite(const int &num_images)
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());

  m_compositor->SetCompositeMode(Compositor::VIS_ORDER_BLEND);

  FindVisibilityOrdering();

  for(int i = 0; i < num_images; ++i)
  {
    float* color_buffer =
      &GetVTKMPointer(m_renders[i].GetCanvas().GetColorBuffer())[0][0];
    float* depth_buffer =
      GetVTKMPointer(m_renders[i].GetCanvas().GetDepthBuffer());
    int height = m_renders[i].GetCanvas().GetHeight();
    int width = m_renders[i].GetCanvas().GetWidth();

    m_compositor->AddImage(color_buffer,
                           depth_buffer,
                           width,
                           height,
                           m_visibility_orders[i][0]);

    Image result = m_compositor->Composite();
    const std::string image_name = m_renders[i].GetImageName() + ".png";
#ifdef VTKH_PARALLEL
    if(vtkh::GetMPIRank() == 0)
    {
#endif
      ImageToCanvas(result, m_renders[i].GetCanvas(), true);
#ifdef VTKH_PARALLEL
    }
#endif
    m_compositor->ClearImages();
  } // for image
}

void
VolumeRenderer::DepthSort(int num_domains,
                          std::vector<float> &min_depths,
                          std::vector<int> &local_vis_order)
{
  if(min_depths.size() != num_domains)
  {
    throw Error("min depths size does not equal the number of domains");
  }
  if(local_vis_order.size() != num_domains)
  {
    throw Error("local vis order not equal to number of domains");
  }
#ifdef VTKH_PARALLEL
  int root = 0;
  MPI_Comm comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  int num_ranks = vtkh::GetMPISize();
  int rank = vtkh::GetMPIRank();
  int *domain_counts = NULL;
  int *domain_offsets = NULL;
  int *vis_order = NULL;
  float *depths = NULL;

  if(rank == root)
  {
    domain_counts = new int[num_ranks];
    domain_offsets = new int[num_ranks];
  }

  MPI_Gather(&num_domains,
             1,
             MPI_INT,
             domain_counts,
             1,
             MPI_INT,
            root,
             comm);

  int depths_size = 0;
  if(rank == root)
  {
    //scan for dispacements
    domain_offsets[0] = 0;
    for(int i = 1; i < num_ranks; ++i)
    {
      domain_offsets[i] = domain_offsets[i - 1] + domain_counts[i - 1];
    }

    for(int i = 0; i < num_ranks; ++i)
    {
      depths_size += domain_counts[i];
    }

    depths = new float[depths_size];

  }

  MPI_Gatherv(&min_depths[0],
              num_domains,
              MPI_FLOAT,
              depths,
              domain_counts,
              domain_offsets,
              MPI_FLOAT,
              root,
              comm);

  if(rank == root)
  {
    std::vector<detail::VisOrdering> order;
    order.resize(depths_size);

    for(int i = 0; i < num_ranks; ++i)
    {
      for(int c = 0; c < domain_counts[i]; ++c)
      {
        int index = domain_offsets[i] + c;
        order[index].m_rank = i;
        order[index].m_domain_index = c;
        order[index].m_minz = depths[index];
      }
    }

    std::sort(order.begin(), order.end(), detail::DepthOrder());

    for(int i = 0; i < depths_size; ++i)
    {
      order[i].m_order = i;
    }

    std::sort(order.begin(), order.end(), detail::RankOrder());

    vis_order = new int[depths_size];
    for(int i = 0; i < depths_size; ++i)
    {
      vis_order[i] = order[i].m_order;
    }
  }

  MPI_Scatterv(vis_order,
               domain_counts,
               domain_offsets,
               MPI_INT,
               &local_vis_order[0],
               num_domains,
               MPI_INT,
               root,
               comm);

  if(rank == root)
  {
    delete[] domain_counts;
    delete[] domain_offsets;
    delete[] vis_order;
    delete[] depths;
  }
#else

  std::vector<detail::VisOrdering> order;
  order.resize(num_domains);

  for(int i = 0; i < num_domains; ++i)
  {
      order[i].m_rank = 0;
      order[i].m_domain_index = i;
      order[i].m_minz = min_depths[i];
  }
  std::sort(order.begin(), order.end(), detail::DepthOrder());

  for(int i = 0; i < num_domains; ++i)
  {
    order[i].m_order = i;
  }

  std::sort(order.begin(), order.end(), detail::RankOrder());

  for(int i = 0; i < num_domains; ++i)
  {
    local_vis_order[i] = order[i].m_order;
  }
#endif
}

void
VolumeRenderer::FindVisibilityOrdering()
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  const int num_cameras = static_cast<int>(m_renders.size());
  m_visibility_orders.resize(num_cameras);

  for(int i = 0; i < num_cameras; ++i)
  {
    m_visibility_orders[i].resize(num_domains);
  }

  //
  // In order for parallel volume rendering to composite correctly,
  // we nee to establish a visibility ordering to pass to IceT.
  // We will transform the data extents into camera space and
  // take the minimum z value. Then sort them while keeping
  // track of rank, then pass the list in.
  //
  std::vector<float> min_depths;
  min_depths.resize(num_domains);

  for(int i = 0; i < num_cameras; ++i)
  {
    const vtkm::rendering::Camera &camera = m_renders[i].GetCamera();
    for(int dom = 0; dom < num_domains; ++dom)
    {
      vtkm::Bounds bounds = this->m_input->GetDomainBounds(dom);
      min_depths[dom] = FindMinDepth(camera, bounds);
    }

    DepthSort(num_domains, min_depths, m_visibility_orders[i]);

  } // for each camera
}
void VolumeRenderer::SetInput(DataSet *input)
{
  Filter::SetInput(input);
  ClearWrappers();

  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  m_has_unstructured = false;
  for(int dom = 0; dom < num_domains; ++dom)
  {
    vtkm::cont::DataSet data_set;
    vtkm::Id domain_id;
    m_input->GetDomain(dom, data_set, domain_id);

    const vtkm::cont::DynamicCellSet &cellset = data_set.GetCellSet();
    if(cellset.GetNumberOfCells() == 0)
    {
      continue;
    }

    const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();
    using Uniform = vtkm::cont::ArrayHandleUniformPointCoordinates;
    using DefaultHandle = vtkm::cont::ArrayHandle<vtkm::FloatDefault>;
    using Rectilinear
      = vtkm::cont::ArrayHandleCartesianProduct<DefaultHandle,
                                                DefaultHandle,
                                                DefaultHandle>;
    bool structured = coords.GetData().IsType<Uniform>() ||
                      coords.GetData().IsType<Rectilinear>();

    if(structured)
    {
      m_wrappers.push_back(new detail::StructuredWrapper(data_set));
    }
    else
    {
      m_has_unstructured = true;
      m_wrappers.push_back(new detail::UnstructuredWrapper(data_set));
    }
  }
}

void VolumeRenderer::ClearWrappers()
{
  const int num_wrappers = m_wrappers.size();
  for(int i = 0; i < num_wrappers; ++i)
  {
    delete m_wrappers[i];
  }
  m_wrappers.clear();
}

std::string
VolumeRenderer::GetName() const
{
  return "vtkh::VolumeRenderer";
}

} // namespace vtkh
