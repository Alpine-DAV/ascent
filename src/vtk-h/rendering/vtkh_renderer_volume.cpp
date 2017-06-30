#include <rendering/vtkh_renderer_volume.hpp>
#include <utils/vtkm_array_utils.hpp>

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperVolume.h>

#include <memory>

namespace vtkh {

namespace detail
{
  struct VisOrdering
  {
    int m_rank;
    int m_domain_index;
    int order;
    float m_minz;
    bool operator<(const VisOrdering &other) const
    {
      return m_minz < other.m_minz;
    } 
  };
} //  namespace detail

vtkhVolumeRenderer::vtkhVolumeRenderer()
{
  typedef vtkm::rendering::MapperVolume TracerType;
  this->m_mapper = std::make_shared<TracerType>();
}

vtkhVolumeRenderer::~vtkhVolumeRenderer()
{
}

vtkhRenderer::vtkmCanvasPtr 
vtkhVolumeRenderer::GetNewCanvas()
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>();
}
void 
vtkhVolumeRenderer::SetupCanvases()
{
  this->CreateCanvases();
  float black[4] = {0.f, 0.f, 0.f, 0.f};
  this->SetCanvasBackgroundColor(black);
}

float 
vtkhVolumeRenderer::FindMinDepth(const vtkm::rendering::Camera &camera, 
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
vtkhVolumeRenderer::Composite(const int &num_images)
{
  const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());

  m_compositor->SetCompositeMode(Compositor::VIS_ORDER_BLEND);

  FindVisibilityOrdering(); 
  return;

  for(int i = 0; i < num_images; ++i)
  {
    for(int dom = 0; dom < num_domains; ++dom)
    {
      float* color_buffer = &GetVTKMPointer(m_canvases[dom][i]->GetColorBuffer())[0][0]; 
      float* depth_buffer = GetVTKMPointer(m_canvases[dom][i]->GetDepthBuffer()); 
      int height = m_canvases[dom][i]->GetHeight();
      int width = m_canvases[dom][i]->GetWidth();

      m_compositor->AddImage(color_buffer,
                             depth_buffer,
                             width,
                             height);
    } //for dom

    Image result = m_compositor->Composite();
    const std::string image_name = "output.png";
#ifdef PARALLEL
    if(VTKH::GetMPIRank() == 0)
    {
      result.Save(image_name);
    }
#else
    result.Save(image_name);
#endif
    m_compositor->ClearImages();
  } // for image
}

void
vtkhVolumeRenderer::FindVisibilityOrdering()
{
    const int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
    const int num_cameras = static_cast<int>(m_cameras.size());

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
      const vtkm::rendering::Camera &camera = m_cameras[i];
      for(int dom = 0; dom < num_domains; ++dom)
      {
        vtkm::Bounds bounds = this->m_input->GetDomainBounds(dom);
        min_depths[dom] = FindMinDepth(camera, bounds);
      }
     
#ifdef PARALLEL
      int root = 0;
      MPI_Comm comm = VTKH::GetMPIComm();
      int num_ranks = VTKH::GetMPISize();
      int rank = VTKH::GetMPIRank();
      int *domain_counts = NULL; 
      int *domain_offsets = NULL; 
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
        for(int i = 0; i < num_ranks; ++i)
        {
          std::cout<<"Rank"<<i<<" has "<<domain_counts[i]<<"\n";
        }

        for(int i = 0; i < depths_size; ++i)
        {
          std::cout<<"Global Domain "<<i<<" has depth "<<depths[i]<<"\n";
        }
     }

     if(rank == root)
     {
       delete[] domain_counts;
       delete[] domain_offsets;
       delete[] depths;
     }
#endif
#if 0 
      //FindMinDepth
      int data_type_size;

      MPI_Type_size(MPI_FLOAT, &data_type_size);
      void *z_array;
      
      void *vis_rank_order = malloc(m_mpi_size * sizeof(int));
      VTKMVisibility *vis_order;

      if(m_rank == 0)
      {
          // TODO CDH :: new / delete, or use conduit?
          z_array = malloc(m_mpi_size * data_type_size);
      }

      MPI_Gather(&minz, 1, MPI_FLOAT, z_array, 1, MPI_FLOAT, 0, m_mpi_comm);

      if(m_rank == 0)
      {
          vis_order = new VTKMVisibility[m_mpi_size];
          
          for(int i = 0; i < m_mpi_size; i++)
          {
              vis_order[i].m_rank = i;
              vis_order[i].m_minz = ((float*)z_array)[i];
          }

          std::qsort(vis_order,
                     m_mpi_size,
                     sizeof(VTKMVisibility),
                     VTKMCompareVisibility);

          
          for(int i = 0; i < m_mpi_size; i++)
          {
              ((int*) vis_rank_order)[i] = vis_order[i].m_rank;;
          }
          
          free(z_array);
          delete[] vis_order;
      }

      MPI_Bcast(vis_rank_order, m_mpi_size, MPI_INT, 0, m_mpi_comm);
      return (int*)vis_rank_order;
#endif
    } // for each camera
}
} // namespace vtkh
