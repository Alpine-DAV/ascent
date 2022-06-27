#include "extract_slice.hpp"

#include <dray/data_model/data_set.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/filters/isosurfacing.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/dispatcher.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <sstream>
#include <string>

// Internal implementation
namespace
{

using namespace dray;
using VecType = Vec<Float, 3>;

static std::string
make_distance_field_name(int plane)
{
  std::ostringstream oss;
  oss << "__dray_distance" << plane;
  return oss.str();
}

static DataSet
remove_distance_fields(DataSet &domain)
{
  DataSet retval(domain);
  retval.clear_fields();
  const int nfields = domain.number_of_fields();
  for(int i = 0; i < nfields; i++)
  {
    const auto ptr = domain.field_shared(i);
    if(ptr->name().rfind("__dray_distance", 0) != std::string::npos)
    {
      // Skip the distance fields
      continue;
    }
    retval.add_field(ptr);
  }
  return retval;
}

template<typename MeshElemType>
static std::vector<std::shared_ptr<Field>>
compute_distance_fields(UnstructuredMesh<MeshElemType> &mesh,
                        const std::vector<VecType> &points,
                        const std::vector<VecType> &normals)
{
  // Setup points / normals as arrays
  const int nplanes = static_cast<int>(points.size());
  Array<VecType> arr_points(points.data(), nplanes);
  Array<VecType> arr_normals(normals.data(), nplanes);
  const auto *points_ptr = arr_points.get_device_ptr_const();
  const auto *normals_ptr = arr_normals.get_device_ptr_const();

  // Get mesh data
  const GridFunction<3> &mesh_gf = mesh.get_dof_data();
  const auto npts = mesh_gf.m_values.size();
  const auto *mesh_data_ptr = mesh_gf.m_values.get_device_ptr_const();

  // Create return values
  std::vector<GridFunction<1>> out_gfs(nplanes);
  std::vector<Vec<Float, 1>*> out_data_ptrs(nplanes);
  for(int p = 0; p < nplanes; p++)
  {
    out_gfs[p].m_el_dofs = mesh_gf.m_el_dofs;
    out_gfs[p].m_size_el = mesh_gf.m_size_el;
    out_gfs[p].m_size_ctrl = mesh_gf.m_size_ctrl;
    out_gfs[p].m_ctrl_idx = mesh_gf.m_ctrl_idx;
    out_gfs[p].m_values.resize(mesh_gf.m_values.size());
    out_data_ptrs[p] = out_gfs[p].m_values.get_device_ptr();
  }

  const RAJA::RangeSegment range(0, npts);
  // TODO: Benchmark which is faster, loop p inside kernel or outside kernel
#if 1
  RAJA::forall<for_policy>(range,
    [=](int i)
    {
      const Float x = mesh_data_ptr[i][0];
      const Float y = mesh_data_ptr[i][1];
      const Float z = mesh_data_ptr[i][2];
      for(int p = 0; p < nplanes; p++)
      {
        const Float px = points_ptr[p][0];
        const Float py = points_ptr[p][1];
        const Float pz = points_ptr[p][2];
        const Float nx = normals_ptr[p][0];
        const Float ny = normals_ptr[p][1];
        const Float nz = normals_ptr[p][2];
        out_data_ptrs[p][i][0] = ((x - px) * nx) + ((y - py) * ny) + ((z - pz) * nz);
      }
    });
#else
  for(int p = 0; p < nplanes; p++)
  {
    RAJA::forall<for_policy>(range,
      [=](int i)
      {
        const Float x = mesh_data_ptr[i][0];
        const Float y = mesh_data_ptr[i][1];
        const Float z = mesh_data_ptr[i][2];
        const Float px = points_ptr[p][0];
        const Float py = points_ptr[p][1];
        const Float pz = points_ptr[p][2];
        const Float nx = normals_ptr[p][0];
        const Float ny = normals_ptr[p][1];
        const Float nz = normals_ptr[p][2];
        out_data_ptrs[p][i] = ((x - px) * nx) + ((y - py) * ny) + ((z - pz) * nz);
      });
  }
#endif
  DRAY_ERROR_CHECK();

  using FieldElemType = Element<MeshElemType::get_dim(),
                                1,
                                MeshElemType::get_etype(),
                                MeshElemType::get_P()>;
  std::vector<std::shared_ptr<Field>> retval;
  for(int p = 0; p < nplanes; p++)
  {
    retval.emplace_back(
      std::make_shared<UnstructuredField<FieldElemType>>(
        out_gfs[p], mesh.order(), make_distance_field_name(p)));
  }
  return retval;
}

struct ComputeDistanceFields
{
  std::vector<VecType> *m_points;
  std::vector<VecType> *m_normals;
  std::vector<std::shared_ptr<Field>> m_output;

  ComputeDistanceFields() = delete;
  ComputeDistanceFields(std::vector<VecType> *points,
                        std::vector<VecType> *normals);
  ~ComputeDistanceFields() = default;

  template<typename MeshType>
  void operator()(MeshType &mesh);
};

ComputeDistanceFields::ComputeDistanceFields(std::vector<VecType> *points,
                                             std::vector<VecType> *normals)
  : m_points(points), m_normals(normals)
{

}

template<typename MeshType>
void
ComputeDistanceFields::operator()(MeshType &mesh)
{
  m_output = compute_distance_fields(mesh, *m_points, *m_normals);
}

}//anonymous namespace

// Public API
namespace dray
{

ExtractSlice::ExtractSlice()
  : m_points(), m_normals()
{
  // Default
}

ExtractSlice::~ExtractSlice()
{
  // Default
}

void
ExtractSlice::add_plane(Vec<Float, 3> point, Vec<Float, 3> normal)
{
  m_points.push_back(point);
  m_normals.push_back(normal);
}

void
ExtractSlice::clear()
{
  m_points.clear();
  m_normals.clear();
}

std::pair<Collection, Collection>
ExtractSlice::execute(Collection &input)
{
  if(m_normals.size() != m_points.size())
  {
    DRAY_ERROR("ExtractSlice, m_normals.size() != m_points.size()");
  }

  if(m_normals.size() == 0)
  {
    return std::make_pair(Collection(), Collection());
  }

  // Create distance fields
  Collection working_collection;
  auto input_domains = input.domains();
  for(DataSet &domain : input_domains)
  {
    DataSet working_domain(domain);
    Mesh *mesh = working_domain.mesh();
    ComputeDistanceFields dist(&m_points, &m_normals);
    dispatch(mesh, dist);
    auto &dist_fields = dist.m_output;
    for(auto &dist_field : dist_fields)
    {
      working_domain.add_field(dist_field);
    }
    working_collection.add_domain(working_domain);
  }

  // Isosurface them
  const int nplanes = static_cast<int>(m_normals.size());
  std::pair<Collection, Collection> retval;
  for(int p = 0; p < nplanes; p++)
  {
    ExtractIsosurface iso;
    iso.iso_field(make_distance_field_name(p));
    iso.iso_value(0.f);
    // Put all results in the retval collections
    auto iso_tris_quads = iso.execute(working_collection);
    auto tri_domains = iso_tris_quads.first.domains();
    for(DataSet &domain : tri_domains)
    {
      DataSet new_domain = remove_distance_fields(domain);
      retval.first.add_domain(new_domain);
    }
    auto quad_domains = iso_tris_quads.second.domains();
    for(DataSet &domain : quad_domains)
    {
      DataSet new_domain = remove_distance_fields(domain);
      retval.second.add_domain(new_domain);
    }
  }
  return retval;
}


}//namespace dray
