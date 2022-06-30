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

static const int8 tet_triangle_table[7*16] = {
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

static const int8 tet_num_triangles[16] = {
  0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0
};

static const int8 tet_edge_table[12] = {
  0, 1,
  1, 2,
  0, 2,
  0, 3,
  1, 3,
  2, 3
};

static const int8 *get_triangle_edges(uint32 flags) { return tet_triangle_table + flags*7; }
static int get_num_triangles(uint32 flags) { return static_cast<int>(tet_num_triangles[flags]); }
static std::pair<int8, int8> get_edge(int edge) { return {tet_edge_table[edge*2], tet_edge_table[edge*2 + 1]}; }

template<typename T>
static Array<T>
unique_values(const Array<T> &input)
{
  Array<T> temp_array;
  array_copy(temp_array, input);

  // Sort the array
  T *temp_ptr = temp_array.get_device_ptr();
  RAJA::sort<for_policy>(RAJA::make_span(temp_ptr, temp_array.size()));

  // Create a mask of values to keep
  Array<bool> mask_array;
  mask_array.resize(temp_array.size());
  bool *mask_ptr = mask_array.get_device_ptr();
  mask_ptr[0] = 1;
  RAJA::forall<for_policy>(RAJA::make_span(temp_ptr+1, temp_array.size() - 1),
    [=] DRAY_LAMBDA (int idx) {
      mask_ptr[idx] = temp_ptr[idx] != temp_ptr[idx-1];
    });
  
  // Create offsets array and get the size of our final output array
  Array<uint32> offsets_array;
  offsets_array.resize(mask_array.size());
  uint32 *offsets_ptr = offsets_array.get_device_ptr();
  RAJA::exclusive_scan<for_policy>(RAJA::make_span(mask_ptr, mask_array.size()),
                                   RAJA::make_span(offsets_ptr, offsets_array.size()),
                                   RAJA::operators::plus<uint32>{});
  const uint32 final_size = offsets_array.get_value(offsets_array.size() - 1) + mask_array.get_value(mask_array.size() - 1);

  // Build the output array
  Array<T> retval_array;
  retval_array.resize(final_size);
  T *retval_ptr = retval_array.get_device_ptr();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, offsets_array.size()),
    [=] DRAY_LAMBDA (int idx) {
      const auto offset = offsets_ptr[idx];
      if(mask_ptr[idx])
      {
        retval_ptr[offset] = temp_ptr[idx];
      }
    });
  return retval_array;
}

template<typename T>
DRAY_EXEC static int32
binary_search(const T *data, int32 size, T value)
{
  int32 start = 0;
  int32 end = size;
  int32 current;
  while(start != end)
  {
    current = (start + end) / 2;
    if(data[current] == value)
    {
      break;
    }
    else if(value < data[current])
    {
      end = current;
    }
    else
    {
      start = current;
    }
  }
  return current;
}

struct MarchingCubesFunctor
{
  DataSet m_input;
  DataSet m_output;
  std::string m_field;
  Float m_isovalue;
  Array<Float> m_weights_array;

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
  const auto ndofs = 4;
  const int nelem = mesh.cells();

  Array<uint32> cut_info;
  cut_info.resize(nelem);
  uint32 *cut_info_ptr = cut_info.get_device_ptr();

  Array<uint32> num_triangles;
  num_triangles.resize(nelem);
  uint32 *num_triangles_ptr = num_triangles.get_device_ptr();

  // Determine triangle cases and number of triangles
  RAJA::ReduceSum<reduce_policy, uint32> reduce_total_triangles(0);
  const auto elem_range = RAJA::RangeSegment(0, nelem);
  RAJA::forall<for_policy>(elem_range,
    [=] DRAY_LAMBDA (int eid) {
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      uint32 info = 0u;
      for(int i = 0; i < ndofs; i++)
      {
        info |= (rdp[i][0] > m_isovalue) << i;
      }
      cut_info_ptr[eid] = info;
      reduce_total_triangles += get_num_triangles(info);
    });
  uint32 total_triangles = reduce_total_triangles.get();

  // Compute edge ids and new connectivity
  uint32 nedges = total_triangles * 3;
  Array<uint64> edge_ids_array;
  edge_ids_array.resize(nedges);
  uint64 *edge_ids_ptr = edge_ids_array.get_device_ptr();

  RAJA::forall<for_policy>(elem_range,
    [=] DRAY_LAMBDA (int eid) {
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      const int8 *edges = get_triangle_edges(cut_info_ptr[eid]);
      const int32 *ctrl_idx_ptr = rdp.m_offset_ptr;
      uint64 *edge_ids_offset = edge_ids_ptr;
      while(*edges != -1)
      {
        const auto edge = get_edge(*edges++);
        const bool should_swap = ctrl_idx_ptr[edge.first] > ctrl_idx_ptr[edge.second];
        const uint64 v0 = static_cast<uint64>(should_swap ? ctrl_idx_ptr[edge.second] : ctrl_idx_ptr[edge.first]);
        const uint64 v1 = static_cast<uint64>(should_swap ? ctrl_idx_ptr[edge.first] : ctrl_idx_ptr[edge.second]);
        const uint64 id = (v0 << 32) | v1;
        *edge_ids_offset++ = id;
      }
    });

  // Compute unique edges
  Array<uint64> unique_edges_array = unique_values(edge_ids_array);
  const uint64 *unique_edges_ptr = unique_edges_array.get_device_ptr_const();

  // Compute new mesh connectivity
  Array<int32> new_conn_array;
  new_conn_array.resize(unique_edges_array.size());
  int32 *new_conn_ptr = new_conn_array.get_device_ptr();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, edge_ids_array.size()),
    [=](int idx) {
      const uint64 edge_id = edge_ids_ptr[idx];
      new_conn_ptr[idx] = binary_search(unique_edges_ptr, unique_edges_array.size(), edge_id);
    });

  // Compute interpolant weights
  m_weights_array.resize(unique_edges_array.size());
  Float *weights_ptr = m_weights_array.get_device_ptr();
  const RAJA::RangeSegment edge_range(0, unique_edges_array.size());
  RAJA::forall<for_policy>(edge_range,
    [=] DRAY_LAMBDA (int idx) {
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(0).read_dof_ptr();
      // e0 stored in upper 32 bits, e1 stored in lower 32 bits
      const uint64 edge_id = unique_edges_ptr[idx];
      const uint32 e0 = static_cast<uint32>(edge_id >> 32);
      const uint32 e1 = static_cast<uint32>(edge_id & 0xFFFFFFFF);
      const Float v0 = rdp[e0][0];
      const Float v1 = rdp[e1][0];
      weights_ptr[idx] = (m_isovalue - v0) / (v1 - v0);
    });

  // Compute new point locations
  Array<Vec<Float, 3>> new_pts_array;
  new_pts_array.resize(unique_edges_array.size());
  Vec<Float, 3> *new_pts_ptr = new_pts_array.get_device_ptr();
  Array<Vec<Float, 1>> new_values_array;
  new_values_array.resize(new_pts_array.size());
  Vec<Float, 1> *new_values_ptr = new_values_array.get_device_ptr();
  DeviceMesh<METype> dmesh(mesh);
  RAJA::forall<for_policy>(edge_range,
    [=] DRAY_LAMBDA (int idx) {
      const ReadDofPtr<Vec<Float, 3>> rdp = dmesh.get_elem(0).read_dof_ptr();
      const uint64 edge_id = unique_edges_ptr[idx];
      const uint32 e0 = static_cast<uint32>(edge_id >> 32);
      const uint32 e1 = static_cast<uint32>(edge_id & 0xFFFFFFFF);
      const Vec<Float, 3> &v0 = rdp[e0];
      const Vec<Float, 3> &v1 = rdp[e1];
      const Float weight = weights_ptr[idx];
      Vec<Float, 3> &new_pt = new_pts_ptr[idx];
      for(int c = 0; c < 3; c++)
      {
        new_pt[c] = (1. - weight) * v0[c] + weight * v1[c];
      }
      new_values_ptr[idx] = 1.;
    });

  GridFunction<1> out_field_gf;
  out_field_gf.m_ctrl_idx = array_counting(unique_edges_array.size(), 1, 1);
  out_field_gf.m_values = new_values_array;
  out_field_gf.m_el_dofs = 3;
  out_field_gf.m_size_el = total_triangles;
  out_field_gf.m_ctrl_idx = new_conn_array;
  m_output.add_field(std::make_shared<UnstructuredField<Tri_P1>>(out_field_gf, 1, "example"));

  GridFunction<3> out_mesh_gf;
  out_mesh_gf.m_ctrl_idx = new_conn_array;
  out_mesh_gf.m_values = new_pts_array;
  out_mesh_gf.m_el_dofs = 3;
  out_mesh_gf.m_size_el = total_triangles;
  out_mesh_gf.m_size_ctrl = new_conn_array.size();
  m_output.add_mesh(std::make_shared<UnstructuredMesh<Tri_P1>>(out_mesh_gf, 1));
  std::cout << m_output.number_of_fields() << std::endl;
}

template<typename Functor>
void dispatch_3d_linear(Mesh *mesh, Field *field, Functor &func)
{
  if (/*!dispatch_mesh_field((HexMesh_P1*)0, mesh, field, func) && */
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
  std::cout << output.local_size() << std::endl;
  std::cout << output.domain(0).number_of_fields() << std::endl;
  return output;
}

}//namespace dray
