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

// #define DEBUG_MARCHING_CUBES

namespace
{

using namespace dray;

#ifdef DEBUG_MARCHING_CUBES
#define DEBUG_PRINT(stream) do {\
  std::cout << stream ;\
} while(0)

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
#else // ifndef DEBUG_MARCHING_CUBES
#define DEBUG_PRINT(stream)

template<typename T>
static void
print_array(const Array<T> &, const std::string)
{
  return;
}

#endif

DRAY_EXEC static uint64
pack_ids(int32 e0, int32 e1)
{
  // Larger value gets stored in the lower bits
  const bool should_swap = e0 > e1;
  const uint64 v0 = static_cast<uint64>(should_swap ? e1 : e0);
  const uint64 v1 = static_cast<uint64>(should_swap ? e0 : e1);
  return (v0 << 32) | v1;
}

DRAY_EXEC static Vec<uint32, 2>
unpack_ids(uint64 edge_id)
{
  // e0 stored in upper 32 bits, e1 stored in lower 32 bits
  Vec<uint32, 2> retval;
  retval[0] = static_cast<uint32>(edge_id >> 32);
  retval[1] = static_cast<uint32>(edge_id & 0xFFFFFFFF);
  return retval;
}

// Simple binary search implementation, data must be sorted
// in ascending order and value must be in the data.
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

template<typename Functor>
void marching_cubes_dispatch(Field *field, Functor &func)
{
  if (/*!dispatch_field_only((UnstructuredField<HexScalar_P1>*)0, field, func) && */
      !dispatch_field_only((UnstructuredField<TetScalar_P1>*)0, field, func))
  {
    ::dray::detail::cast_field_failed(field, __FILE__, __LINE__);
  }
}

template<typename Functor>
void blend_edges_dispatch(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<HexScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P1>*)0, field, func))
  {
    ::dray::detail::cast_field_failed(field, __FILE__, __LINE__);
  }
}

template<typename Functor>
void blend_edges_dispatch(Mesh *mesh, Functor &func)
{
  if (!dispatch_mesh_only((HexMesh_P1*)0, mesh, func) &&
      !dispatch_mesh_only((TetMesh_P1*)0, mesh, func))
  {
    ::dray::detail::cast_mesh_failed(mesh, __FILE__, __LINE__);
  }
}

// --------------------------------------------------------------------------------------
//                         ***** Begin BlendEdgesFunctor *****
// --------------------------------------------------------------------------------------
struct BlendEdgesFunctor
{
  const Array<int32> *m_conn_array;
  const Array<uint64> *m_edges_array;
  const Array<Float> *m_weights_array;
  std::shared_ptr<Mesh> m_output_mesh;
  std::shared_ptr<Field> m_output_field;
  BlendEdgesFunctor(const Array<int32> *conn_array,
                    const Array<uint64> *edges, const Array<Float> *weights);

  template<typename T, int nc>
  GridFunction<nc> do_blend(const Array<Vec<T, nc>> &data_array);

  template<typename FEType>
  void operator()(UnstructuredField<FEType> &mesh);

  template<typename METype>
  void operator()(UnstructuredMesh<METype> &mesh);
};

BlendEdgesFunctor::BlendEdgesFunctor(const Array<int32> *conn_array,
                                     const Array<uint64> *edges, const Array<Float> *weights)
  : m_conn_array(conn_array), m_edges_array(edges), m_weights_array(weights),
    m_output_mesh(), m_output_field()
{
  // Default
}

template<typename T, int nc>
GridFunction<nc>
BlendEdgesFunctor::do_blend(const Array<Vec<T, nc>> &data_array)
{
  const auto &conn_array = *m_conn_array;
  const auto &edges_array = *m_edges_array;
  const auto &weights_array = *m_weights_array;

  // Allocate output array
  GridFunction<nc> retval;
  retval.m_values.resize(edges_array.size());
  Vec<Float, nc> *values_ptr = retval.m_values.get_device_ptr();

  const Float *weights_ptr = weights_array.get_device_ptr_const();
  const uint64 *edges_ptr = edges_array.get_device_ptr_const();
  const Vec<Float, nc> *data_ptr = data_array.get_device_ptr_const();

  const RAJA::RangeSegment edge_range(0, edges_array.size());
  RAJA::forall<for_policy>(edge_range,
    [=] DRAY_LAMBDA (int idx) {
      const Vec<uint32, 2> edge_def = unpack_ids(edges_ptr[idx]);
      const uint32 e0 = edge_def[0];
      const uint32 e1 = edge_def[1];
      const Vec<Float, nc> &v0 = data_ptr[e0];
      const Vec<Float, nc> &v1 = data_ptr[e1];
      const Float weight = weights_ptr[idx];
      Vec<Float, nc> &value = values_ptr[idx];
      for(int c = 0; c < nc; c++)
      {
        value[c] = (1. - weight) * v0[c] + weight * v1[c];
      }
    });
  DRAY_ERROR_CHECK();

  retval.m_ctrl_idx = conn_array;
  retval.m_el_dofs = 3;
  retval.m_size_el = conn_array.size() / 3;
  retval.m_size_ctrl = conn_array.size();
  return retval;
}

template<typename FEType>
void
BlendEdgesFunctor::operator()(UnstructuredField<FEType> &field)
{
  static_assert(FEType::get_P() == Order::Linear);
  const auto &in_gf = field.get_dof_data();
  const auto out_gf = do_blend(in_gf.m_values);
  using ElemType = Element<2, in_gf.get_ncomp(), ElemType::Simplex, Order::Linear>;
  m_output_field = std::make_shared<UnstructuredField<ElemType>>(out_gf, 1, field.name());
}

template<typename METype>
void
BlendEdgesFunctor::operator()(UnstructuredMesh<METype> &mesh)
{
  static_assert(METype::get_P() == Order::Linear);
  const auto &in_gf = mesh.get_dof_data();
  const auto out_gf = do_blend(in_gf.m_values);
  m_output_mesh = std::make_shared<TriMesh_P1>(out_gf, 1);
}

// --------------------------------------------------------------------------------------
//                       ***** Begin MarchingCubesFunctor *****
// --------------------------------------------------------------------------------------
struct MarchingCubesFunctor
{
  DataSet m_input;
  DataSet m_output;
  std::string m_field;
  Array<uint64> m_unique_edges_array;
  Array<int32> m_conn_array;
  Array<Float> m_weights_array;
  Float m_isovalue;
  uint32 m_total_triangles;

  MarchingCubesFunctor(DataSet &in,
                      const std::string &field,
                      Float isoval);

  DataSet execute(Field *field);

  template<typename FEType>
  void operator()(UnstructuredField<FEType> &field);
};

MarchingCubesFunctor::MarchingCubesFunctor(DataSet &in,
                                           const std::string &field,
                                           Float isoval)
  : m_input(in), m_output(), m_field(field), m_unique_edges_array(),
    m_conn_array(), m_weights_array(), m_isovalue(isoval), m_total_triangles(0)
{

}

template<typename FEType>
void
MarchingCubesFunctor::operator()(UnstructuredField<FEType> &field)
{
  static_assert(FEType::get_P() == Order::Linear);

  // Grab some useful information about the field
  constexpr OrderPolicy<Order::Linear> field_order_p;
  constexpr auto shape3d = adapt_get_shape<FEType>();
  constexpr auto ndofs = eattr::get_num_dofs(shape3d, field_order_p);
  const int nelem = field.get_num_elem();

  // Get the proper lookup table for the current shape
  const Array<int8> lookup_array = detail::get_lookup_table(shape3d);
  const int8 *lookup_ptr = lookup_array.get_device_ptr_const();

  Array<uint32> cut_info;
  cut_info.resize(nelem);
  uint32 *cut_info_ptr = cut_info.get_device_ptr();

  Array<uint32> num_triangles_array;
  num_triangles_array.resize(nelem);
  uint32 *num_triangles_ptr = num_triangles_array.get_device_ptr();

  // Determine triangle cases and number of triangles
  DeviceField<FEType> dfield(field);
  const auto elem_range = RAJA::RangeSegment(0, nelem);
  // TODO: Debug CUDA failure
  // terminate called after throwing an instance of 'std::runtime_error'
  //   what():  campCudaErrchk(cudaStreamSynchronize(stream)) an illegal memory access was encountered /home/cdl/Programs/camp-develop/include/camp/resource/cuda.hpp:172
  RAJA::forall<for_policy>(elem_range,
    [=] DRAY_LAMBDA (int eid) {
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      uint32 info = 0u;
      for(int i = 0; i < ndofs; i++)
      {
        info |= (rdp[i][0] > m_isovalue) << i;
      }
      cut_info_ptr[eid] = info;
      num_triangles_ptr[eid] = detail::get_num_triangles(shape3d, lookup_ptr, info);
    });
  DRAY_ERROR_CHECK();
  Array<uint32> triangle_offsets_array = array_exc_scan_plus(num_triangles_array, m_total_triangles);
  const uint32 *triangle_offsets_ptr = triangle_offsets_array.get_device_ptr_const();

  // Compute edge ids and new connectivity
  uint32 nedges = m_total_triangles * 3;
  Array<uint64> edge_ids_array;
  edge_ids_array.resize(nedges);
  uint64 *edge_ids_ptr = edge_ids_array.get_device_ptr();

  DEBUG_PRINT("triangle_edge_defs:");
  RAJA::forall<for_policy>(elem_range,
    [=] DRAY_LAMBDA (int eid) {
      DEBUG_PRINT("\n  [" << eid << "]: " << num_triangles_ptr[eid] << " " << cut_info_ptr[eid]);
      const ReadDofPtr<Vec<Float, 1>> rdp = dfield.get_elem(eid).read_dof_ptr();
      const int8 *edges = detail::get_triangle_edges(shape3d, lookup_ptr, cut_info_ptr[eid]);
      const int32 *ctrl_idx_ptr = rdp.m_offset_ptr;
      uint64 *edge_ids_offset = edge_ids_ptr + triangle_offsets_ptr[eid] * 3;
      while(*edges != detail::NO_EDGE)
      {
        const auto edge = detail::get_edge(shape3d, lookup_ptr, *edges++);
        const uint64 id = pack_ids(ctrl_idx_ptr[edge[0]], ctrl_idx_ptr[edge[1]]);
        DEBUG_PRINT("\n    (" << ctrl_idx_ptr[edge[0]] << "," << ctrl_idx_ptr[edge[1]] << ") (" << rdp[edge[0]][0] << "," << rdp[edge[1]][0] << ")");
        *edge_ids_offset++ = id;
      }
    });
  DRAY_ERROR_CHECK();
  DEBUG_PRINT(std::endl);
  print_array(edge_ids_array, "edge_ids_array");

  // Compute unique edges
  m_unique_edges_array = array_unique_values(edge_ids_array);
  const uint64 *unique_edges_ptr = m_unique_edges_array.get_device_ptr_const();

  // Compute new mesh connectivity
  const auto unique_edges_size = m_unique_edges_array.size();
  m_conn_array.resize(edge_ids_array.size());
  int32 *conn_ptr = m_conn_array.get_device_ptr();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, edge_ids_array.size()),
    [=] DRAY_LAMBDA (int idx) {
      const uint64 edge_id = edge_ids_ptr[idx];
      conn_ptr[idx] = binary_search(unique_edges_ptr, unique_edges_size, edge_id);
    });
  DRAY_ERROR_CHECK();

  // Compute interpolant weights
  m_weights_array.resize(m_unique_edges_array.size());
  Float *weights_ptr = m_weights_array.get_device_ptr();
  // We don't want to use the ReadDofPtr object because we already have the indicies we want
  const Vec<Float, 1> *field_ptr = dfield.get_elem(0).read_dof_ptr().m_dof_ptr;
  const RAJA::RangeSegment edge_range(0, m_unique_edges_array.size());
  DEBUG_PRINT("Compute interpolant weights:");
  RAJA::forall<for_policy>(edge_range,
    [=] DRAY_LAMBDA (int idx) {
      const Vec<uint32, 2> edge_def = unpack_ids(unique_edges_ptr[idx]);
      const uint32 e0 = edge_def[0];
      const uint32 e1 = edge_def[1];
      const Float v0 = field_ptr[e0][0];
      const Float v1 = field_ptr[e1][0];
      const Float w = (m_isovalue - v0) / (v1 - v0);
      DEBUG_PRINT("\n  " << unique_edges_ptr[idx] << " -> (" << e0 << "," << e1 << ") = (" << v0 << "," << v1 << ") = " << w);
      weights_ptr[idx] = w;
    });
  DRAY_ERROR_CHECK();
  DEBUG_PRINT(std::endl);
}

DataSet
MarchingCubesFunctor::execute(Field *in_field)
{
  // Compute new connectivity and interpolant weights
  marching_cubes_dispatch(in_field, *this);

  // Blend edges to create new mesh and new fields
  BlendEdgesFunctor blend(&m_conn_array, &m_unique_edges_array, &m_weights_array);
  blend_edges_dispatch(m_input.mesh(), blend);
  m_output.add_mesh(blend.m_output_mesh);

  // Iterate fields
  const int nfields = m_input.number_of_fields();
  for(int i = 0; i < nfields; i++)
  {
    Field *field = m_input.field(i);
    if(field == in_field)
    {
      // For the selected field just set the value to the isovalue
      GridFunction<1> out_field_gf;
      out_field_gf.m_ctrl_idx = m_conn_array;
      out_field_gf.m_el_dofs = 3;
      out_field_gf.m_size_el = m_conn_array.size() / 3;
      out_field_gf.m_size_ctrl = m_conn_array.size();
      out_field_gf.m_values.resize(m_unique_edges_array.size());
      array_memset(out_field_gf.m_values, Vec<Float, 1>{m_isovalue});
      m_output.add_field(std::make_shared<UnstructuredField<TriScalar_P1>>(out_field_gf, 1, field->name()));
    }
    else if(field->order() == 1)
    {
      blend_edges_dispatch(field, blend);
      m_output.add_field(blend.m_output_field);
    }
  }

  return m_output;
}

}//anonymous namespace

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
    DataSet iso_domain = func.execute(domain.field(m_field));
    output.add_domain(iso_domain);
  }
  return output;
}

}//namespace dray
