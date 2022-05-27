#include "ScalarRenderer.hpp"
#include <vtkh/compositing/PayloadCompositor.hpp>

#include <vtkh/vtkh.hpp>

#include <vtkh/Logger.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>
#include <vtkm/rendering/raytracing/Logger.h>
#include <vtkm/rendering/ScalarRenderer.h>

#ifdef VTKH_PARALLEL
  #include <mpi.h>
#endif
#include <assert.h>
#include <string.h>

using namespace std;

namespace vtkh
{

namespace detail
{
vtkm::cont::DataSet
filter_scalar_fields(vtkm::cont::DataSet &dataset)
{
  vtkm::cont::DataSet res;
  const vtkm::Id num_coords = dataset.GetNumberOfCoordinateSystems();
  for(vtkm::Id i = 0; i < num_coords; ++i)
  {
    res.AddCoordinateSystem(dataset.GetCoordinateSystem(i));
  }
  res.SetCellSet(dataset.GetCellSet());

  const vtkm::Id num_fields = dataset.GetNumberOfFields();
  for(vtkm::Id i = 0; i < num_fields; ++i)
  {
    vtkm::cont::Field field = dataset.GetField(i);
    if(field.GetData().GetNumberOfComponentsFlat() == 1)
    {
      if(field.GetData().IsValueType<vtkm::Float32>() ||
         field.GetData().IsValueType<vtkm::Float64>())
      {
        res.AddField(field);
      }
    }
  }


  return res;
}

} // namespace detail

ScalarRenderer::ScalarRenderer()
  : m_width(1024),
    m_height(1024)
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
ScalarRenderer::SetCamera(vtkmCamera &camera)
{
  m_camera = camera;
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
  std::vector<vtkm::rendering::ScalarRenderer> renderers;
  std::vector<vtkm::Id> cell_counts;
  renderers.resize(num_domains);
  cell_counts.resize(num_domains);
  for(int dom = 0; dom < num_domains; ++dom)
  {
    vtkm::cont::DataSet data_set;
    vtkm::Id domain_id;
    m_input->GetDomain(dom, data_set, domain_id);
    vtkm::cont::DataSet filtered = detail::filter_scalar_fields(data_set);
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
    vtkm::cont::DataSet data_set;
    vtkm::Id domain_id;
    m_input->GetDomain(dom, data_set, domain_id);
    num_cells = data_set.GetCellSet().GetNumberOfCells();

    if(data_set.GetCellSet().GetNumberOfCells())
    {
      no_data = num_cells == 0;

      Result res = renderers[dom].Render(m_camera);

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
        delete array;
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
        MPI_Send(field_names[i].c_str(),strlen(field_names[i].c_str()),MPI_CHAR, 0, 0,mpi_comm);
      }
    }
  }
#endif

  if(!no_data)
  {
    if(num_cells == 0)
    {
      vtkm::Bounds b(bounds);
      PayloadImage p(b, max_p);
      int size = p.m_depths.size();
      std::vector<float> depths(size);
      for(int i = 0; i < size; i++)
        depths[i] = std::numeric_limits<int>::max();
      std::copy(&depths[0], &depths[0] + size, &p.m_depths[0]);
      compositor.AddImage(p);
    }

    if(min_p != max_p)
    {
      throw Error("Scalar Renderer: mismatch in payload bytes");
    }

    PayloadImage final_image = compositor.Composite();
    if(vtkh::GetMPIRank() == 0)
    {
      Result final_result = Convert(final_image, field_names);
      if(final_result.Scalars.size() != 0)
      {
        vtkm::cont::DataSet dset = final_result.ToDataSet();
        const int domain_id = 0;
        this->m_output->AddDomain(dset, domain_id);
      }
    }
  }

}

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
    vtkm::cont::ArrayHandle<vtkm::Float32> array;
    array.Allocate(size);
    result.Scalars.push_back(array);
    float* buffer = GetVTKMPointer(result.Scalars[i]);
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
  float* dbuffer = GetVTKMPointer(result.Depths);
  memcpy(dbuffer, &image.m_depths[0], sizeof(float) * size);

  return result;
}

PayloadImage * ScalarRenderer::Convert(Result &result)
{
  const int num_fields = result.Scalars.size();
  const int payload_size = num_fields * sizeof(float);
  vtkm::Bounds bounds;
  bounds.X.Min = 1;
  bounds.Y.Min = 1;
  bounds.X.Max = result.Width;
  bounds.Y.Max = result.Height;

  const size_t size = result.Width * result.Height;

  PayloadImage *image = new PayloadImage(bounds, payload_size);
  unsigned char *loads = &image->m_payloads[0];

  float* dbuffer = GetVTKMPointer(result.Depths);
  memcpy(&image->m_depths[0], dbuffer, sizeof(float) * size);
  // copy scalars into payload
  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    vtkm::cont::ArrayHandle<vtkm::Float32> scalar = result.Scalars[i];
    float* buffer = GetVTKMPointer(scalar);
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

vtkh::DataSet *
ScalarRenderer::GetInput()
{
  return m_input;
}

} // namespace vtkh
