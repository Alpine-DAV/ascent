// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/marching_cubes.hpp>

#include <dray/data_model/data_set.hpp>
#include <dray/data_model/device_field.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/dispatcher.hpp>

namespace
{

using namespace dray;

template<typename Shape>
struct cell_table {};

template<>
struct cell_table<ShapeTet>
{
  const static int num_triangles[] = {
    0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0
  };
  const static int triangle_table[] = {
#define X -1
    X, X, X, X, X, X, X,
    0, 3, 2, X, X, X, X,
    0, 1, 4, X, X, X, X,
    1, 4, 2, 2, 4, 3, X,
    1, 2, 5, X, X, X, X,
    0, 3, 5, 0, 5, 1, X,
    0, 2, 5, 0, 5, 4, X,
    5, 4, 3, X, X, X, X,
    3, 4, 5, X, X, X, X,
    4, 5, 0, 5, 2, 0, X,
    1, 5, 0, 5, 3, 0, X,
    5, 2, 1, X, X, X, X,
    3, 4, 2, 2, 4, 1, X,
    4, 1, 0, X, X, X, X,
    2, 3, 0, X, X, X, X,
    X, X, X, X, X, X, X
#undef X
  };
  const static std::pair<uint8, uint8> edge_table[] = {
    {0, 1},
    {1, 2},
    {0, 2},
    {0, 3},
    {1, 3},
    {2, 3}
  };

  static const int *get_triangle_edges(uint32 flags) { return triangle_table + flags*7; }
  static int get_num_triangles(uint32 flags) { return num_triangles[flags]; }
  static std::pair<uint8, uint8> get_edge(int edge) { return edge_table[edge]; }
};

struct MarchingCubesFunctor
{
  DataSet m_input;
  DataSet m_output;
  std::string m_field;
  Float m_isovalue;

  MarchingCubesFunctor(DataSet &in,
                      const std::string &field,
                      Float isoval);

  template<typename METype, typename FEType>
  void operator()(UnstructuredMesh<METype> &mesh,
                  UnstructuredField<FEType> &field);
};

MarchingCubesFunctor::MarchingCubesFunctor(DataSet &in,
                                           const std::string &field,
                                           Float isoval)
  : m_input(in), m_output(), m_field(field), m_isovalue(isoval)
{

}

template<typename METype, typename FEType>
void
MarchingCubesFunctor::operator()(UnstructuredMesh<METype> &mesh,
                                 UnstructuredField<FEType> &field)
{
  static_assert(METype::get_P() == Order::Linear);
  // static_assert(FEType::get_P() == Order::Linear);

  DeviceField<FEType> dfield(field);
  using TableType = cell_table<ShapeTet>;
  const auto ndofs = 4;
  const int nelem = mesh.cells();

  Array<uint32> cut_info;
  cut_info.resize(nelem);
  uint32 *cut_info_ptr = cut_info.get_device_ptr();

  Array<uint32> num_triangles;
  num_triangles.resize(nelem);
  uint32 *num_triangles_ptr = num_triangles.get_device_ptr();

  // Determine triangle cases and number of triangles
  const auto elem_range = RAJA::RangeSegment(0, nelem);
  RAJA::forall<for_policy>(elem_range,
    [=](int eid) {
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      uint32 info = 0u;
      for(int i = 0; i < ndofs; i++)
      {
        info |= (rdp[i][0] > m_isovalue) << i;
      }
      num_triangles_ptr[eid] = TableType::get_num_triangles(info);
      cut_info_ptr[eid] = info;
    });

    uint32 total_triangles;
    Array<uint32> offsets_array = array_exc_scan_plus(num_triangles, total_triangles);
    uint32 *offsets_ptr = offsets_array.get_device_ptr();
    Array<Float> weights_array;
    weights_array.resize(total_triangles * 3);
    Float *weights_ptr = weights_array.get_device_ptr();

    // Compute interpolant weights
    RAJA::forall<for_policy>(elem_range,
      [=](int eid) {
        const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
        Float *weights_offset = weights_ptr + offsets_ptr[eid] * 3;
        const int *edges = TableType::get_triangle_edges(cut_info_ptr[eid]);
        while(*edges != -1)
        {
          const auto edge = TableType::get_edge(*edges++);
          const Float v0 = rdp[edge.first][0];
          const Float v1 = rdp[edge.second][0];
          *weights_offset++ = (m_isovalue - v0) / (v1 - v0);
        }
      });

  // TODO: Merge points

  Array<Vec<Float, 3>> new_pts_array;
  new_pts_array.resize(total_triangles * 3);
  Vec<Float, 3> *new_pts_ptr = new_pts_array.get_device_ptr();

  DeviceMesh<METype> dmesh(mesh);
  // Build output mesh
  RAJA::forall<for_policy>(elem_range,
    [=](int eid) {
      const ReadDofPtr<Vec<Float, 3>> rdp = dmesh.get_elem(eid).read_dof_ptr();
      Float *weights_offset = weights_ptr + offsets_ptr[eid] * 3;
      Vec<Float, 3> *new_pts_offset = new_pts_ptr + offsets_ptr[eid] * 3;
      const int *edges = TableType::get_triangle_edges(cut_info_ptr[eid]);
      while(*edges != -1)
      {
        std::pair<uint8, uint8> e[3];
        e[0] = TableType::get_edge(*edges++);
        e[1] = TableType::get_edge(*edges++);
        e[2] = TableType::get_edge(*edges++);
        for(int ei = 0; ei < 3; ei++)
        {
          const Float weight = *weights_offset;
          const auto &edge = e[ei];
          Vec<Float, 3> &pt = *new_pts_offset;
          for(int vi = 0; vi < 3; vi++)
          {
            pt[vi] = (1. - weight) * rdp[edge.first][vi] + weight * rdp[edge.second][vi];
          }
          weights_offset++;
          new_pts_offset++;
        }
      }
    });
  
  GridFunction<3> out_mesh_gf;
  out_mesh_gf.m_ctrl_idx = array_counting(total_triangles * 3, 0, 1);
  out_mesh_gf.m_values = new_pts_array;
  out_mesh_gf.m_el_dofs = 3;
  out_mesh_gf.m_size_el = total_triangles;
  out_mesh_gf.m_size_ctrl = total_triangles * 3;
  m_output.add_mesh(std::make_shared<UnstructuredMesh<Tri_P1>>(out_mesh_gf, 1));
}

template<typename Functor>
void dispatch_3d_linear(Mesh *mesh, Field *field, Functor &func)
{
  if (!dispatch_mesh_field((HexMesh_P1*)0, mesh, field, func) &&
      !dispatch_mesh_field((TetMesh_P1*)0, mesh, field, func))
  {
    ::dray::detail::cast_mesh_failed(mesh, __FILE__, __LINE__);
  }
}

}

namespace dray
{

MarchingCubes::MarchingCubes()
  : m_field(), m_isovalues()
{
}

MarchingCubes::~MarchingCubes()
{
}

void
MarchingCubes::set_field(const std::string &name)
{
  m_field = name;
}

void
MarchingCubes::set_isovalue(Float val)
{
  m_isovalues.resize(1);
  m_isovalues[0] = val;
}

void
MarchingCubes::set_isovalues(const Float *values,
                             int nvalues)
{
  m_isovalues.resize(nvalues);
  memcpy(m_isovalues.data(), values, nvalues * sizeof(Float));
}

Collection
MarchingCubes::execute(Collection &c)
{
  if(m_isovalues.empty())
  {
    DRAY_ERROR("MarchingCubes::execute() called with no isovalue set.");
  }

  if(!c.has_field(m_field))
  {
    DRAY_ERROR("The given collection does not contain a field called '" 
      << m_field << "'.");
  }

  Collection output;
  auto domains = c.domains();
  for(auto &domain : domains)
  {
    MarchingCubesFunctor func(domain, m_field, m_isovalues[0]);
    dispatch_3d_linear(domain.mesh(), domain.field(m_field), func);
    output.add_domain(func.m_output);
  }
  return output;
}

}//namespace dray
