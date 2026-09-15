#include "ScalarRenderer.hpp"
#include <vtkh/compositing/PayloadCompositor.hpp>

#include <vtkh/vtkh.hpp>

#include <vtkh/Logger.hpp>
#include <vtkh/utils/viskores_array_utils.hpp>
#include <vtkh/utils/viskores_dataset_info.hpp>
#include <viskores/cont/Algorithm.h>
#include <viskores/rendering/raytracing/Logger.h>
#include <viskores/rendering/raytracing/RayOperations.h>
#include <viskores/rendering/Camera.h>
#include <viskores/rendering/raytracing/Camera.h>


#include <viskores/rendering/ScalarRenderer.h>
#include <conduit/conduit.hpp>
#include <conduit/conduit_relay.hpp>
#include <conduit/conduit_blueprint.hpp>
#include <conduit_fmt/conduit_fmt.h>

#ifdef VTKH_PARALLEL
  #include <mpi.h>
#endif
#include <assert.h>
#include <string.h>
#include <algorithm>

using namespace std;
using namespace conduit;

namespace vtkh
{

namespace detail
{
viskores::cont::DataSet
filter_scalar_fields(viskores::cont::DataSet &dataset,
                     const std::vector<std::string> &field_names)
{
  // we will also screen field names if passed vector is non empty
  bool skip_field_names = field_names.empty();
  viskores::cont::DataSet res;
  const viskores::Id num_coords = dataset.GetNumberOfCoordinateSystems();
  for(viskores::Id i = 0; i < num_coords; ++i)
  {
    res.AddCoordinateSystem(dataset.GetCoordinateSystem(i));
  }
  res.SetCellSet(dataset.GetCellSet());

  const viskores::Id num_fields = dataset.GetNumberOfFields();
  for(viskores::Id i = 0; i < num_fields; ++i)
  {
    viskores::cont::Field field = dataset.GetField(i);
    if(field.GetData().GetNumberOfComponentsFlat() == 1)
    {
      if(skip_field_names || 
         std::find(field_names.begin(), field_names.end(), field.GetName()) != field_names.end() )
      {
          if(field.GetData().IsValueType<viskores::Float32>() ||
             field.GetData().IsValueType<viskores::Float64>())
          {
            res.AddField(field);
          }
      }
    }
  }


  return res;
}

} // namespace detail

ScalarRenderer::ScalarRenderer()
  : m_width(1024),
    m_height(1024),
    m_mode("camera")
{
}

ScalarRenderer::~ScalarRenderer()
{
}

std::string
ScalarRenderer::GetName() const
{
  return "vtkh::ScalarRenderer";
}

void
ScalarRenderer::SetCamera(viskoresCamera &camera)
{
  m_camera = camera;
  m_mode = "camera";
}


void
ScalarRenderer::SetFields(const std::vector<std::string> &field_names)
{
  m_field_names = field_names;
}

void
ScalarRenderer::PreExecute()
{
}

void
ScalarRenderer::Update()
{
  VTKH_DATA_OPEN(this->GetName());
#ifdef VTKH_ENABLE_LOGGING
  long long int in_cells = this->m_input->GetNumberOfCells();
  VTKH_DATA_ADD("input_cells", in_cells);
#endif
  PreExecute();
  DoExecute();
  PostExecute();
  VTKH_DATA_CLOSE();
}

void
ScalarRenderer::PostExecute()
{
  Filter::PostExecute();
}

namespace detail
{

template <typename Precision>
void
CreateRaysMesh(const ScalarRenderer::Result &srender_res,
               const viskores::rendering::raytracing::Ray<Precision> &rays,
               conduit::Node &rays_mesh)
{
    // Create a Blueprint Mesh that represents the ray trace results

    // Scalar Rendering Result Struct Details
    /*
    struct VISKORES_RENDERING_EXPORT Result
    {
      viskores::Int32 Width;
      viskores::Int32 Height;
      viskores::cont::ArrayHandle<viskores::Float32> Depths;
      std::vector<viskores::cont::ArrayHandle<viskores::Float32>> Scalars;
      std::vector<std::string> ScalarNames;
      std::map<std::string, viskores::Range> Ranges;
    */

    const int num_rays   = srender_res.Width * srender_res.Height;
    const int num_fields = srender_res.ScalarNames.size();

    const float *depth_buffer = GetVISKORESPointer(srender_res.Depths);

    rays_mesh.reset();
    rays_mesh["coordsets/rays_coords/type"] = "explicit";

    // each ray will have to verts
    index_t npts = num_rays * 2;

    rays_mesh["coordsets/rays_coords/values/x"].set(DataType::float64(npts));
    rays_mesh["coordsets/rays_coords/values/y"].set(DataType::float64(npts));
    rays_mesh["coordsets/rays_coords/values/z"].set(DataType::float64(npts));
    float64_array xs = rays_mesh["coordsets/rays_coords/values/x"].value();
    float64_array ys = rays_mesh["coordsets/rays_coords/values/y"].value();
    float64_array zs = rays_mesh["coordsets/rays_coords/values/z"].value();

    auto rays_orig_x = rays.OriginX.ReadPortal();
    auto rays_orig_y = rays.OriginY.ReadPortal();
    auto rays_orig_z = rays.OriginZ.ReadPortal();

    auto rays_dir_x = rays.DirX.ReadPortal();
    auto rays_dir_y = rays.DirY.ReadPortal();
    auto rays_dir_z = rays.DirZ.ReadPortal();
    auto rays_pixel_idx = rays.PixelIdx.ReadPortal();

    xs.fill(0);
    ys.fill(0);
    zs.fill(0);

    index_t idx = 0;
    //
    // // debug stmt to check ray buffer sizes
    // std::cout << "total : " << rays_orig_x.GetNumberOfValues() << " vs " << srender_res.Height <<
    //     " " << srender_res.Width << " " << "tot " << (srender_res.Height * srender_res.Width) << std::endl;
    //

    index_t num_active_rays = rays_orig_x.GetNumberOfValues();
    for(index_t active_ray_idx=0; active_ray_idx<num_active_rays; active_ray_idx++)
    {
        index_t pixel_idx = rays_pixel_idx.Get(active_ray_idx);
        index_t img_idx = pixel_idx *2;
        viskores::Vec<Precision,3> ray_origin(rays_orig_x.Get(active_ray_idx),
                                              rays_orig_y.Get(active_ray_idx),
                                              rays_orig_z.Get(active_ray_idx));
        if(depth_buffer[active_ray_idx] > 0)
        {
            // first point:
            //  origin
            // second point:
            //  distance * normalize(dir) + origin
            // normalize dir

            viskores::Vec<Precision,3> ray_dir(rays_dir_x.Get(active_ray_idx),
                                               rays_dir_y.Get(active_ray_idx),
                                               rays_dir_z.Get(active_ray_idx));

            Precision ray_dist = (Precision) depth_buffer[active_ray_idx];

            xs[img_idx] = ray_origin[0];
            ys[img_idx] = ray_origin[1];
            zs[img_idx] = ray_origin[2];

            viskores::Normalize(ray_dir);
            viskores::Vec<Precision,3> ray_end = (ray_dist * ray_dir) + ray_origin;
            xs[img_idx+1] = ray_end[0];
            ys[img_idx+1] = ray_end[1];
            zs[img_idx+1] = ray_end[2];

        }
        else // no hit, line
        {
            xs[img_idx] = ray_origin[0];
            ys[img_idx] = ray_origin[1];
            zs[img_idx] = ray_origin[2];

            xs[img_idx+1] = ray_origin[0];
            ys[img_idx+1] = ray_origin[1];
            zs[img_idx+1] = ray_origin[2];
        }
        idx+=2;
    }

    rays_mesh["topologies/rays/type"] = "unstructured";
    rays_mesh["topologies/rays/coordset"] = "rays_coords";
    rays_mesh["topologies/rays/elements/shape"] = "line";
    rays_mesh["topologies/rays/elements/connectivity"].set(DataType::index_t(npts));
    index_t_array ray_conn = rays_mesh["topologies/rays/elements/connectivity"].value();

    idx = 0;
    for(index_t j=0;j<srender_res.Height;j++)
    for(index_t i=0;i<srender_res.Width;i++)
    {
        ray_conn[idx]   = idx;
        ray_conn[idx+1] = idx+1;
        idx+=2;
    }

    rays_mesh["fields/depth/topology"] = "rays";
    rays_mesh["fields/depth/association"] = "element";
    rays_mesh["fields/depth/values"].set(DataType::float64(num_rays));
    float64_array depth_vals = rays_mesh["fields/depth/values"].value();

    for(index_t active_ray_idx=0; active_ray_idx<num_active_rays; active_ray_idx++)
    {
        index_t pixel_idx = rays_pixel_idx.Get(active_ray_idx);
        depth_vals[pixel_idx] = depth_buffer[active_ray_idx];
    }

    for(index_t i=0; i<srender_res.Scalars.size(); i++)
    {
        const float* scalar_buffer = GetVISKORESPointer(srender_res.Scalars[i]);
        const std::string field_path = "fields/" + srender_res.ScalarNames[i];
        rays_mesh[field_path + "/topology"] = "rays";
        rays_mesh[field_path + "/association"] = "element";
        rays_mesh[field_path + "/values"].set(DataType::float64(num_rays));
        float64_array fld_vals = rays_mesh[field_path + "/values"].value();

        for(index_t active_ray_idx=0; active_ray_idx<num_active_rays; active_ray_idx++)
        {
            index_t pixel_idx = rays_pixel_idx.Get(active_ray_idx);
            fld_vals[pixel_idx] = scalar_buffer[active_ray_idx];
        }
    }

    conduit::Node info;
    if(!conduit::blueprint::mesh::verify(rays_mesh,info))
    {
        // TODO: Error
        std::cout << info.to_yaml() << std::endl;
    }
}
};


void
ScalarRenderer::GenerateResultRaysMesh(const Result &result_image,
                                       conduit::Node &rays_mesh)
{
    if(vtkh::GetMPIRank() == 0)
    {
        // rays output
        viskores::rendering::raytracing::Ray<viskores::Float32> rays;
        if(m_mode == "camera")
        {
            GenerateCameraRays(GetCamera(),
                               GetResultBounds(),
                               m_width,
                               m_height,
                               rays);
        }
        else if(m_mode == "rays")
        {
            GenerateExplicitRays(rays);
        }
        // create a mesh that represents the rays
        detail::CreateRaysMesh(result_image, rays, rays_mesh.append());
    }
}


void
ScalarRenderer::DoExecute()
{

  int num_domains = static_cast<int>(m_input->GetNumberOfDomains());
  this->m_output = new DataSet();

  //
  // There external faces + bvh construction happens
  // when we set the input for the renderer, which
  // we don't want to repeat for every camera. Also,
  // We could be processing AMR patches, numbering
  // in the 1000s, and with 100 images * 1000s amr
  // patches we could blow memory. We will set the input
  // once and composite after every image (todo: batch images
  // in groups of X).
  //
  std::vector<viskores::rendering::ScalarRenderer> renderers;
  std::vector<viskores::Id> cell_counts;
  renderers.resize(num_domains);
  cell_counts.resize(num_domains);
  for(int dom = 0; dom < num_domains; ++dom)
  {
    viskores::cont::DataSet data_set;
    viskores::Id domain_id;
    m_input->GetDomain(dom, data_set, domain_id);
    viskores::cont::DataSet filtered = detail::filter_scalar_fields(data_set,
                                                                m_field_names);
    renderers[dom].SetInput(filtered);
    renderers[dom].SetWidth(m_width);
    renderers[dom].SetHeight(m_height);

    // all the data sets better be the same
    cell_counts.push_back(data_set.GetCellSet().GetNumberOfCells());
  }

  // basic sanity checking
  int min_p = std::numeric_limits<int>::max();
  int max_p = std::numeric_limits<int>::min();
  bool do_once = true;

  std::vector<std::string> field_names;
  PayloadCompositor compositor;

  int num_cells = 0;

  // make no assumptions
  bool no_data = num_cells == 0;

  //Bounds needed for parallel execution
  float bounds[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};;
  for(int dom = 0; dom < num_domains; ++dom)
  {
    viskores::cont::DataSet data_set;
    viskores::Id domain_id;
    m_input->GetDomain(dom, data_set, domain_id);
    num_cells = data_set.GetCellSet().GetNumberOfCells();

    if(data_set.GetCellSet().GetNumberOfCells())
    {
      no_data = num_cells == 0;


      Result res;
      if(m_mode == "camera")
      {
          res = renderers[dom].Render(m_camera);
      }
      else if(m_mode == "rays")
      {
          // TODO Float32 vs 64
          viskores::rendering::raytracing::Ray<viskores::Float32> rays;
          GenerateExplicitRays(rays);
          res = renderers[dom].Render(rays);
      }

      field_names = res.ScalarNames;
      PayloadImage *pimage = Convert(res);
      min_p = std::min(min_p, pimage->m_payload_bytes);
      max_p = std::max(max_p, pimage->m_payload_bytes);
      compositor.AddImage(*pimage);
      bounds[0] = pimage->m_bounds.X.Min;
      bounds[1] = pimage->m_bounds.X.Max;
      bounds[2] = pimage->m_bounds.Y.Min;
      bounds[3] = pimage->m_bounds.Y.Max;
      bounds[4] = pimage->m_bounds.Z.Min;
      bounds[5] = pimage->m_bounds.Z.Max;
      delete pimage;
    }
  }

#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

  int comm_size = GetMPISize();
  int rank = GetMPIRank();
  std::vector<int> votes;

  if(!no_data && num_cells == 0)
    num_cells = 1;
  int vote = num_cells > 0 ? 1 : 0;
  votes.resize(comm_size);

  MPI_Allgather(&vote, 1, MPI_INT, &votes[0], 1, MPI_INT, mpi_comm);
  int winner = -1;
  for(int i = 0; i < comm_size; ++i)
  {
    if(votes[i] == 1)
    {
      winner = i;
      break;
    }
  }
  if(winner != -1)
  {
    MPI_Bcast(bounds, 6, MPI_FLOAT, winner, mpi_comm);
    MPI_Bcast(&max_p, 1, MPI_INT, winner, mpi_comm);
    MPI_Bcast(&min_p, 1, MPI_INT, winner, mpi_comm);
    no_data = false;
  }

  if(winner > 0)
  {
    if(vtkh::GetMPIRank() == 0 && num_cells == 0)
    {
      MPI_Status status;
      int num_fields = 0;
      MPI_Recv(&num_fields, 1, MPI_INT, winner, 0, mpi_comm, &status);
      for(int i = 0; i < num_fields; i++)
      {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, winner, 0, mpi_comm, &status);
        char * array = new char[len];
        MPI_Recv(array, len, MPI_CHAR, winner, 0, mpi_comm, &status);
        std::string name;
        name.assign(array,len);
        field_names.push_back(name);
        memset(array, 0, sizeof(*array));
        delete[] array;
      }
    }
    if(vtkh::GetMPIRank() == winner)
    {
      int num_fields = field_names.size();
      MPI_Send(&num_fields, 1, MPI_INT, 0, 0, mpi_comm); 
      for(int i = 0; i < num_fields; i++)
      {
        int len = strlen(field_names[i].c_str());
        MPI_Send(&len, 1, MPI_INT, 0, 0, mpi_comm);
        MPI_Send(const_cast<char*>(field_names[i].c_str()),
                 strlen(field_names[i].c_str()),
                 MPI_CHAR, 0, 0,mpi_comm);
      }
    }
  }
#endif

  if(!no_data)
  {
    if(num_cells == 0)
    {
      viskores::Bounds b(bounds);
      PayloadImage p(b, max_p);
      int size = p.m_depths.size();
      std::vector<float> depths(size);
      for(int i = 0; i < size; i++)
        depths[i] = std::numeric_limits<int>::max();
      std::copy(&depths[0], &depths[0] + size, &p.m_depths[0]);
      compositor.AddImage(p);
    }

    // keep a copy of the bounds
    m_bounds = viskores::Bounds(bounds);

    if(min_p != max_p)
    {
      throw Error("Scalar Renderer: mismatch in payload bytes");
    }

    PayloadImage final_image = compositor.Composite();
    if(vtkh::GetMPIRank() == 0)
    {
      m_result_image = Convert(final_image, field_names);
      if(m_result_image.Scalars.size() != 0)
      {
        viskores::cont::DataSet dset = m_result_image.ToDataSet();
        const int domain_id = 0;
        this->m_output->AddDomain(dset, domain_id);
      }
    }
  }
}

template <typename Precision>
void
ScalarRenderer::GenerateCameraRays(const viskoresCamera &camera,
                                   const viskores::Bounds &bounds,
                                   int width, int height,
                                   viskores::rendering::raytracing::Ray<Precision> &rays)
{
    viskores::Bounds cam_bounds(bounds);
    viskores::rendering::raytracing::Camera ray_cam = camera.CreateRaytracingCamera((viskores::Int32)width,
                                                                                    (viskores::Int32)height);
    ray_cam.CreateRays(rays, bounds);
    rays.Buffers.at(0).InitConst(0.f);
}

template <typename Precision>
void
ScalarRenderer::GenerateExplicitRays(viskores::rendering::raytracing::Ray<Precision> &rays)
{
        viskores::rendering::raytracing::RayOperations::Resize(rays, m_rays_pts_xs.GetNumberOfValues());

        Precision infinity;
        viskores::rendering::raytracing::GetInfinity(infinity);

        viskores::cont::ArrayHandleConstant<Precision> inf(infinity, rays.NumRays);
        viskores::cont::Algorithm::Copy(inf, rays.MaxDistance);

        viskores::cont::ArrayHandleConstant<Precision> zero(0, rays.NumRays);
        viskores::cont::Algorithm::Copy(zero, rays.MinDistance);
        viskores::cont::Algorithm::Copy(zero, rays.Distance);

        viskores::cont::ArrayHandleConstant<viskores::Id> initHit(-2, rays.NumRays);
        viskores::cont::Algorithm::Copy(initHit, rays.HitIdx);


        viskores::cont::Algorithm::Copy(m_rays_pts_xs, rays.OriginX);
        viskores::cont::Algorithm::Copy(m_rays_pts_ys, rays.OriginY);
        viskores::cont::Algorithm::Copy(m_rays_pts_zs, rays.OriginZ);

        viskores::cont::Algorithm::Copy(m_rays_dirs_xs, rays.DirX);
        viskores::cont::Algorithm::Copy(m_rays_dirs_ys, rays.DirY);
        viskores::cont::Algorithm::Copy(m_rays_dirs_zs, rays.DirZ);

        for(int i=0;i<rays.NumRays;i++)
        {
            rays.PixelIdx.WritePortal().Set(i,i);
        }

        rays.EnableIntersectionData();
        rays.Buffers.at(0).InitConst(0.f);

};


ScalarRenderer::Result
ScalarRenderer::Convert(PayloadImage &image, std::vector<std::string> &names)
{
  Result result;
  result.ScalarNames = names;
  const int num_fields = names.size();

  const int dx  = image.m_bounds.X.Max - image.m_bounds.X.Min + 1;
  const int dy  = image.m_bounds.Y.Max - image.m_bounds.Y.Min + 1;
  const int size = dx * dy;

  result.Width = dx;
  result.Height = dy;

  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    viskores::cont::ArrayHandle<viskores::Float32> array;
    array.Allocate(size);
    result.Scalars.push_back(array);
    float* buffer = GetVISKORESPointer(result.Scalars[i]);
    buffers.push_back(buffer);
  }

  const unsigned char *loads = &image.m_payloads[0];
  const size_t payload_size = image.m_payload_bytes;

  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(float);
      memcpy(&buffers[i][x], loads + offset, sizeof(float));
    }
  }

  //
  result.Depths.Allocate(size);
  float* dbuffer = GetVISKORESPointer(result.Depths);
  memcpy(dbuffer, &image.m_depths[0], sizeof(float) * size);

  return result;
}

PayloadImage * ScalarRenderer::Convert(Result &result)
{
  const int num_fields = result.Scalars.size();
  const int payload_size = num_fields * sizeof(float);
  viskores::Bounds bounds;
  bounds.X.Min = 1;
  bounds.Y.Min = 1;
  bounds.X.Max = result.Width;
  bounds.Y.Max = result.Height;

  const size_t size = result.Width * result.Height;

  PayloadImage *image = new PayloadImage(bounds, payload_size);
  unsigned char *loads = &image->m_payloads[0];

  float* dbuffer = GetVISKORESPointer(result.Depths);
  memcpy(&image->m_depths[0], dbuffer, sizeof(float) * size);
  // copy scalars into payload
  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    viskores::cont::ArrayHandle<viskores::Float32> scalar = result.Scalars[i];
    float* buffer = GetVISKORESPointer(scalar);
    buffers.push_back(buffer);
  }
#ifdef VTKH_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(float);
      memcpy(loads + offset, &buffers[i][x], sizeof(float));
    }
  }
  return image;
}

void
ScalarRenderer::SetHeight(const int height)
{
  m_height = height;
}

void
ScalarRenderer::SetWidth(const int width)
{
  m_width = width;
}

void
ScalarRenderer::SetRays(viskores::cont::ArrayHandle<viskores::Float64> pts_xs,
                        viskores::cont::ArrayHandle<viskores::Float64> pts_ys,
                        viskores::cont::ArrayHandle<viskores::Float64> pts_zs,
                        viskores::cont::ArrayHandle<viskores::Float64> dirs_xs,
                        viskores::cont::ArrayHandle<viskores::Float64> dirs_ys,
                        viskores::cont::ArrayHandle<viskores::Float64> dirs_zs)
{
    // result is a 1D image
    SetWidth(pts_xs.GetNumberOfValues());
    SetHeight(1);

    m_rays_pts_xs = pts_xs;
    m_rays_pts_ys = pts_ys;
    m_rays_pts_zs = pts_zs;

    m_rays_dirs_xs = dirs_xs;
    m_rays_dirs_ys = dirs_ys;
    m_rays_dirs_zs = dirs_zs;

    m_mode = "rays";
}


vtkh::DataSet *
ScalarRenderer::GetInput()
{
  return m_input;
}

} // namespace vtkh
