#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

Scene::Scene()
  : m_has_volume(false),
    m_batch_size(10)
{

}

Scene::~Scene()
{

}

void
Scene::SetRenderBatchSize(int batch_size)
{
  if(batch_size < 1)
  {
    throw Error("Render batch size must be greater than 0");
  }
  m_batch_size = batch_size;
}

int
Scene::GetRenderBatchSize() const
{
  return m_batch_size;
}

void
Scene::AddRender(vtkh::Render &render)
{
  m_renders.push_back(render);
}

void
Scene::SetRenders(const std::vector<vtkh::Render> &renders)
{
  m_renders = renders;
}

bool
Scene::IsMesh(vtkh::Renderer *renderer)
{
  bool is_mesh = false;

  if(dynamic_cast<vtkh::MeshRenderer*>(renderer) != nullptr)
  {
    is_mesh = true;
  }
  return is_mesh;
}

bool
Scene::IsVolume(vtkh::Renderer *renderer)
{
  bool is_volume = false;

  if(dynamic_cast<vtkh::VolumeRenderer*>(renderer) != nullptr)
  {
    is_volume = true;
  }
  return is_volume;
}

void
Scene::AddRenderer(vtkh::Renderer *renderer)
{
  bool is_volume = IsVolume(renderer);
  bool is_mesh = IsMesh(renderer);

  if(is_volume)
  {
    if(m_has_volume)
    {
      throw Error("Scenes only support a single volume plot");
    }

    m_has_volume = true;
    // make sure that the volume render is last
    m_renderers.push_back(renderer);
  }
  else if(is_mesh)
  {
    // make sure that the mesh plot is last
    // and before the volume pl0t
    if(m_has_volume)
    {
      if(m_renderers.size() == 1)
      {
        m_renderers.push_front(renderer);
      }
      else
      {
        auto it = m_renderers.end();
        it--;
        it--;
        m_renderers.insert(it,renderer);
      }
    }
    else
    {
      m_renderers.push_back(renderer);
    }
  }
  else
  {
    m_renderers.push_front(renderer);
  }
}

void
Scene::Render()
{

  std::vector<vtkm::Range> ranges;
  std::vector<std::string> field_names;
  std::vector<vtkm::cont::ColorTable> color_tables;
  bool do_once = true;

  //
  // We are going to render images in batches. With databases
  // like Cinema, we could be rendering hundres of images. Keeping
  // all the canvases around can hog memory so we will conserve it.
  // For example, if we rendered 360 images at 1024^2, all the canvases
  // would consume 7GB of space. Not good on the GPU, where resources
  // are limited.
  //
  const int render_size = m_renders.size();
  int batch_start = 0;
  while(batch_start < render_size)
  {
    int batch_end = std::min(m_batch_size + batch_start, render_size);
    auto begin = m_renders.begin() + batch_start;
    auto end = m_renders.begin() + batch_end;

    std::vector<vtkh::Render> current_batch(begin, end);

    for(auto  render : current_batch)
    {
      render.GetCanvas().Clear();
    }

    const int plot_size = m_renderers.size();
    auto renderer = m_renderers.begin();

    // render order is enforced inside add
    // Order is:
    // 1) surfaces
    // 2) meshes
    // 3) volume

    // if we have both surfaces/mesh and volumes
    // we need to synchronize depths so that volume
    // only render to the max depth
    bool synch_depths = false;

    int opaque_plots = plot_size;
    if(m_has_volume)
    {
      opaque_plots -= 1;
    }

    //
    // pass 1: opaque geometry
    //
    for(int i = 0; i < opaque_plots; ++i)
    {
      if(i == opaque_plots - 1)
      {
        (*renderer)->SetDoComposite(true);
      }
      else
      {
        (*renderer)->SetDoComposite(false);
      }

      (*renderer)->SetRenders(current_batch);
      (*renderer)->Update();

      (*renderer)->ClearRenders();

      synch_depths = true;
      renderer++;
    }

    //
    // pass 2: volume
    //
    if(m_has_volume)
    {
      if(synch_depths)
      {
        SynchDepths(current_batch);
      }
      (*renderer)->SetDoComposite(true);
      (*renderer)->SetRenders(current_batch);
      (*renderer)->Update();

      current_batch  = (*renderer)->GetRenders();
      (*renderer)->ClearRenders();
    }

    if(do_once)
    {
      // gather color tables and other information for
      // annotations
      for(auto plot : m_renderers)
      {
        if((*plot).GetHasColorTable())
        {
          ranges.push_back((*plot).GetRange());
          field_names.push_back((*plot).GetFieldName());
          color_tables.push_back((*plot).GetColorTable());
        }
      }
      do_once = false;
    }

    // render screen annotations last and save
    for(int i = 0; i < current_batch.size(); ++i)
    {
      current_batch[i].RenderWorldAnnotations();
      current_batch[i].RenderScreenAnnotations(field_names, ranges, color_tables);
      current_batch[i].RenderBackground();
      current_batch[i].Save();
    }

    batch_start = batch_end;
  } // while
}

void Scene::SynchDepths(std::vector<vtkh::Render> &renders)
{
#ifdef VTKH_PARALLEL
  int root = 0; // full images in rank 0
  MPI_Comm comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  int num_ranks = vtkh::GetMPISize();
  int rank = vtkh::GetMPIRank();
  for(auto render : renders)
  {
    vtkm::rendering::Canvas &canvas = render.GetCanvas();
    const int image_size = canvas.GetWidth() * canvas.GetHeight();
    float *depth_ptr = GetVTKMPointer(canvas.GetDepthBuffer());
    MPI_Bcast( depth_ptr, image_size, MPI_FLOAT, 0, comm);
  }
#endif
}

void
Scene::Save()
{

}

} // namespace vtkh
