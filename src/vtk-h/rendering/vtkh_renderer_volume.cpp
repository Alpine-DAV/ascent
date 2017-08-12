#include <rendering/vtkh_renderer_volume.hpp>
#include <utils/vtkm_array_utils.hpp>

#include <vtkm/rendering/CanvasRayTracer.h>

#include <memory>

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
  m_color_table.AddAlphaControlPoint(0.0f, .1);
  m_color_table.AddAlphaControlPoint(1.0f, .1);
  m_num_samples = -1;
}

VolumeRenderer::~VolumeRenderer()
{
}

void 
VolumeRenderer::PreExecute() 
{
  Renderer::PreExecute();
  
  const float default_samples = 200.f;
  float samples = default_samples;
  if(m_num_samples != -1)
  {
    samples = m_num_samples;
    float ratio = default_samples / samples;
    vtkm::rendering::ColorTable corrected;
    corrected = this->m_color_table.CorrectOpacity(ratio);
    m_tracer->SetActiveColorTable(corrected);
  }
  vtkm::Vec<vtkm::Float32,3> extent; 
  extent[0] = static_cast<vtkm::Float32>(this->m_bounds.X.Length());
  extent[1] = static_cast<vtkm::Float32>(this->m_bounds.Y.Length());
  extent[2] = static_cast<vtkm::Float32>(this->m_bounds.Z.Length());
  vtkm::Float32 dist = vtkm::Magnitude(extent) / samples; 
  m_tracer->SetSampleDistance(dist);
}

void
VolumeRenderer::SetNumberOfSamples(const int num_samples)
{
  assert(num_samples > 0);
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
  
  vtkm::Matrix<vtkm::Float32,4,4> view_matrix = camera.CreateViewMatrix();
  
  //
  // z's should both be negative since the camera is 
  // looking down the neg z-axis
  //
  double x[2], y[2], z[2];
   
  x[0] = bounds.X.Min;
  x[1] = bounds.X.Max;
  y[0] = bounds.Y.Min;
  y[1] = bounds.Y.Max;
  z[0] = bounds.Z.Min;
  z[1] = bounds.Z.Max;
  
  float minz;
  minz = std::numeric_limits<float>::max();
  vtkm::Vec<vtkm::Float32,4> extent_point;
  
  for(int i = 0; i < 2; i++)
      for(int j = 0; j < 2; j++)
          for(int k = 0; k < 2; k++)
          {
              extent_point[0] = static_cast<vtkm::Float32>(x[i]);
              extent_point[1] = static_cast<vtkm::Float32>(y[j]);
              extent_point[2] = static_cast<vtkm::Float32>(z[k]);
              extent_point[3] = 1.f;
              extent_point = vtkm::MatrixMultiply(view_matrix, extent_point);
              // perform the perspective divide
              extent_point[2] = extent_point[2] / extent_point[3];
              minz = std::min(minz, -extent_point[2]);
          }

  return minz;
}

void 
VolumeRenderer::Composite(const int &num_images)
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());

  m_compositor->SetCompositeMode(Compositor::VIS_ORDER_BLEND);

  FindVisibilityOrdering(); 

  for(int i = 0; i < num_images; ++i)
  {
    const int num_canvases = m_renders[i].GetNumberOfCanvases();

    for(int dom = 0; dom < num_canvases; ++dom)
    {
      float* color_buffer = &GetVTKMPointer(m_renders[i].GetCanvas(dom)->GetColorBuffer())[0][0]; 
      float* depth_buffer = GetVTKMPointer(m_renders[i].GetCanvas(dom)->GetDepthBuffer()); 

      int height = m_renders[i].GetCanvas(dom)->GetHeight();
      int width = m_renders[i].GetCanvas(dom)->GetWidth();

      m_compositor->AddImage(color_buffer,
                             width,
                             height,
                             m_visibility_orders[i][dom]);
    } //for dom

    Image result = m_compositor->Composite();
    const std::string image_name = m_renders[i].GetImageName() + ".png";
#ifdef PARALLEL
    if(vtkh::GetMPIRank() == 0)
    {
      float bg_color[4];
      bg_color[0] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[0];
      bg_color[1] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[1];
      bg_color[2] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[2];
      bg_color[3] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[3];

      result.CompositeBackground(bg_color);
      result.Save(image_name);
    }
#else
      float bg_color[4];
      bg_color[0] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[0];
      bg_color[1] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[1];
      bg_color[2] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[2];
      bg_color[3] = m_renders[i].GetCanvas(0)->GetBackgroundColor().Components[3];

      result.CompositeBackground(bg_color);
    result.Save(image_name);
#endif
    m_compositor->ClearImages();
  } // for image
}

void 
VolumeRenderer::DepthSort(const int &num_domains, 
                              const std::vector<float> &min_depths,
                              std::vector<int> &local_vis_order)
{
  assert(min_depths.size() == num_domains);
  assert(local_vis_order.size() == num_domains);
#ifdef PARALLEL
  int root = 0;
  MPI_Comm comm = vtkh::GetMPIComm();
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
} // namespace vtkh
