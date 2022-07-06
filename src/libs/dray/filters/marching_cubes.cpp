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
#include <dray/filters/internal/marching_cubes_lookup_tables.hpp>
#include <dray/dispatcher.hpp>

namespace
{

using namespace dray;

template<typename T>
static void
print_array(const Array<T> &input, const std::string name)
{
  std::cout << name << ":";
  const int N = input.size();
  const T *input_ptr = input.get_device_ptr_const();
  for(int i = 0; i < N; i++)
  {
    std::cout << "\n  [" << i << "]: " << input_ptr[i];
  }
  std::cout << std::endl;
}

template<>
void
print_array(const Array<uint8> &input, const std::string name)
{
  std::cout << name << ":";
  const int N = input.size();
  const uint8 *input_ptr = input.get_device_ptr_const();
  for(int i = 0; i < N; i++)
  {
    std::cout << "\n  [" << i << "]: " << static_cast<int>(input_ptr[i]);
  }
  std::cout << std::endl;
}

template<typename T>
static Array<T>
unique_values(const Array<T> &input)
{
  Array<T> temp_array;
  array_copy(temp_array, input);

  // Sort the array
  T *temp_ptr = temp_array.get_device_ptr();
  RAJA::sort<for_policy>(RAJA::make_span(temp_ptr, temp_array.size()));
  DRAY_ERROR_CHECK();

  // Create a mask of values to keep
  Array<uint8> mask_array;
  mask_array.resize(temp_array.size());
  uint8 *mask_ptr = mask_array.get_device_ptr();
  mask_ptr[0] = 1;
  RAJA::forall<for_policy>(RAJA::RangeSegment(1, temp_array.size()),
    [=] DRAY_LAMBDA (int idx) {
      mask_ptr[idx] = temp_ptr[idx] != temp_ptr[idx-1];
    });
  DRAY_ERROR_CHECK();

  // Create offsets array and get the size of our final output array
  Array<uint32> offsets_array;
  offsets_array.resize(mask_array.size());
  uint32 *offsets_ptr = offsets_array.get_device_ptr();
  RAJA::exclusive_scan<for_policy>(RAJA::make_span(mask_ptr, mask_array.size()),
                                   RAJA::make_span(offsets_ptr, offsets_array.size()),
                                   RAJA::operators::plus<uint32>{});
  DRAY_ERROR_CHECK();
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
  DRAY_ERROR_CHECK();
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

  // Get the proper lookup table for the current shape
  constexpr auto shape3d = adapt_get_shape<FEType>();
  const Array<int8> lookup_array = detail::get_lookup_table(shape3d);
  const int8 *lookup_ptr = lookup_array.get_device_ptr_const();

  DeviceField<FEType> dfield(field);
  const auto ndofs = 4;
  const int nelem = mesh.cells();

  Array<uint32> cut_info;
  cut_info.resize(nelem);
  uint32 *cut_info_ptr = cut_info.get_device_ptr();

  Array<uint32> num_triangles_array;
  num_triangles_array.resize(nelem);
  uint32 *num_triangles_ptr = num_triangles_array.get_device_ptr();

  // Determine triangle cases and number of triangles
  const auto elem_range = RAJA::RangeSegment(0, nelem);
  std::cout << "Connectivity:";
  RAJA::forall<for_policy>(elem_range,
    [=] DRAY_LAMBDA (int eid) {
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      const int32 *ctrl_idx_ptr = rdp.m_offset_ptr;
      uint32 info = 0u;
      std::cout << "\n  [" << eid << "]: (";
      for(int i = 0; i < ndofs; i++)
      {
        std::cout << ctrl_idx_ptr[i] << ((i == ndofs - 1) ? "" : ",");
      }
      std::cout << ")";
      std::cout << "\n  [" << eid << "]: (";
      for(int i = 0; i < ndofs; i++)
      {
        std::cout << rdp[i][0] << ((i == ndofs - 1) ? "" : ",");
        info |= (rdp[i][0] > m_isovalue) << i;
      }
      std::cout << ")";
      std::cout << "\n  [" << eid << "]: (";
      const ReadDofPtr<Vec<Float, 1>> nrdp = dfield.get_elem(0).read_dof_ptr();
      const Vec<Float, 1> *nrdp_ptr = nrdp.m_dof_ptr;
      for(int i = 0; i < ndofs; i++)
      {
        std::cout << nrdp_ptr[ctrl_idx_ptr[i]][0] << ((i == ndofs - 1) ? "" : ",");
      }
      std::cout << ")";
      cut_info_ptr[eid] = info;
      num_triangles_ptr[eid] = detail::get_num_triangles(shape3d, lookup_ptr, info);
    });
  DRAY_ERROR_CHECK();
  std::cout << std::endl;
  uint32 total_triangles;
  Array<uint32> triangle_offsets_array = array_exc_scan_plus(num_triangles_array, total_triangles);
  const uint32 *triangle_offsets_ptr = triangle_offsets_array.get_device_ptr_const();

  // Compute edge ids and new connectivity
  uint32 nedges = total_triangles * 3;
  Array<uint64> edge_ids_array;
  edge_ids_array.resize(nedges);
  uint64 *edge_ids_ptr = edge_ids_array.get_device_ptr();

  std::cout << "triangle_edge_defs:";
  RAJA::forall<for_policy>(elem_range,
    [=] DRAY_LAMBDA (int eid) {
      std::cout << "\n  [" << eid << "]: " << num_triangles_ptr[eid] << " " << cut_info_ptr[eid];
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      const int8 *edges = detail::get_triangle_edges(shape3d, lookup_ptr, cut_info_ptr[eid]);
      const int32 *ctrl_idx_ptr = rdp.m_offset_ptr;
      uint64 *edge_ids_offset = edge_ids_ptr + triangle_offsets_ptr[eid] * 3;
      while(*edges != detail::NO_EDGE)
      {
        const auto edge = detail::get_edge(shape3d, lookup_ptr, *edges++);
        const bool should_swap = ctrl_idx_ptr[edge[0]] > ctrl_idx_ptr[edge[1]];
        const uint64 v0 = static_cast<uint64>(should_swap ? ctrl_idx_ptr[edge[1]] : ctrl_idx_ptr[edge[0]]);
        const uint64 v1 = static_cast<uint64>(should_swap ? ctrl_idx_ptr[edge[0]] : ctrl_idx_ptr[edge[1]]);
        const uint64 id = (v0 << 32) | v1;
        std::cout << "\n    (" << v0 << "," << v1 << ") (" << rdp[edge[0]][0] << "," << rdp[edge[1]][0] << ")";
        *edge_ids_offset++ = id;
      }
    });
  DRAY_ERROR_CHECK();
  print_array(edge_ids_array, "edge_ids_array");

  // Compute unique edges
  Array<uint64> unique_edges_array = unique_values(edge_ids_array);
  const uint64 *unique_edges_ptr = unique_edges_array.get_device_ptr_const();

  // Compute new mesh connectivity
  Array<int32> new_conn_array;
  new_conn_array.resize(edge_ids_array.size());
  int32 *new_conn_ptr = new_conn_array.get_device_ptr();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, edge_ids_array.size()),
    [=](int idx) {
      const uint64 edge_id = edge_ids_ptr[idx];
      new_conn_ptr[idx] = binary_search(unique_edges_ptr, unique_edges_array.size(), edge_id);
    });
  DRAY_ERROR_CHECK();

  // Compute interpolant weights
  m_weights_array.resize(unique_edges_array.size());
  Float *weights_ptr = m_weights_array.get_device_ptr();
  // We don't want to use the ReadDofPtr object because we already have the indicies we want
  const Vec<Float, 1> *field_ptr = dfield.get_elem(0).read_dof_ptr().m_dof_ptr;
  const RAJA::RangeSegment edge_range(0, unique_edges_array.size());
  std::cout << "Compute interpolant weights:";
  RAJA::forall<for_policy>(edge_range,
    [=] DRAY_LAMBDA (int idx) {
      // e0 stored in upper 32 bits, e1 stored in lower 32 bits
      const uint64 edge_id = unique_edges_ptr[idx];
      const uint32 e0 = static_cast<uint32>(edge_id >> 32);
      const uint32 e1 = static_cast<uint32>(edge_id & 0xFFFFFFFF);
      const Float v0 = field_ptr[e0][0];
      const Float v1 = field_ptr[e1][0];
      const Float w = (m_isovalue - v0) / (v1 - v0);
      std::cout << "\n  " << edge_id << " -> (" << e0 << "," << e1 << ") = (" << v0 << "," << v1 << ") = " << w;
      weights_ptr[idx] = w;
    });
  DRAY_ERROR_CHECK();
  std::cout << std::endl;

  // Compute new point locations
  Array<Vec<Float, 3>> new_pts_array;
  new_pts_array.resize(unique_edges_array.size());
  Vec<Float, 3> *new_pts_ptr = new_pts_array.get_device_ptr();
  Array<Vec<Float, 1>> new_values_array;
  new_values_array.resize(new_pts_array.size());
  Vec<Float, 1> *new_values_ptr = new_values_array.get_device_ptr();
  // We don't want to use the ReadDofPtr object because we already have the indicies we want
  DeviceMesh<METype> dmesh(mesh);
  const Vec<Float, 3> *mesh_ptr = dmesh.get_elem(0).read_dof_ptr().m_dof_ptr;
  RAJA::forall<for_policy>(edge_range,
    [=] DRAY_LAMBDA (int idx) {
      const uint64 edge_id = unique_edges_ptr[idx];
      const uint32 e0 = static_cast<uint32>(edge_id >> 32);
      const uint32 e1 = static_cast<uint32>(edge_id & 0xFFFFFFFF);
      const Vec<Float, 3> &v0 = mesh_ptr[e0];
      const Vec<Float, 3> &v1 = mesh_ptr[e1];
      const Float weight = weights_ptr[idx];
      Vec<Float, 3> &new_pt = new_pts_ptr[idx];
      for(int c = 0; c < 3; c++)
      {
        new_pt[c] = (1. - weight) * v0[c] + weight * v1[c];
      }
      new_values_ptr[idx] = 1.;
    });
  DRAY_ERROR_CHECK();

  GridFunction<1> out_field_gf;
  out_field_gf.m_ctrl_idx = array_counting(unique_edges_array.size(), 1, 1);
  out_field_gf.m_values = new_values_array;
  out_field_gf.m_el_dofs = 3;
  out_field_gf.m_size_el = total_triangles;
  out_field_gf.m_ctrl_idx = new_conn_array;
  m_output.add_field(std::make_shared<UnstructuredField<Element<3, 1, ElemType::Simplex, Order::Linear>>>(out_field_gf, 1, "example"));

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
