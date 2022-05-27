#include <dray/filters/subset.hpp>

#include <dray/dispatcher.hpp>
#include <dray/data_model/elem_utils.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{

namespace detail
{
static inline
Array<int32> index_flags_with_offses(Array<int32> &flags, Array<int32> &offsets)
{
  // The width of the flags array must match the width of the offsets array (int32).
  // Otherwise something goes wrong; either plus<small_type> overflows
  // or maybe the exclusive_scan<> template doesn't handle two different types.
  // Using a uint8 flags, things were broken, but changing to int32 fixed it.

  const int32 size = flags.size ();
  // TODO: there is an issue with raja where this can't be const
  // when using the CPU
  // const uint8 *flags_ptr = flags.get_device_ptr_const();
  int32 *flags_ptr = flags.get_device_ptr ();
  offsets.resize (size);
  int32 *offsets_ptr = offsets.get_device_ptr ();

  RAJA::operators::safe_plus<int32> plus{};
  RAJA::exclusive_scan<for_policy> (RAJA::make_span(flags_ptr, size),
                                    RAJA::make_span(offsets_ptr, size),
                                    plus);
  DRAY_ERROR_CHECK();

  int32 out_size = (size > 0) ? offsets.get_value (size - 1) : 0;
  // account for the exclusive scan by adding 1 to the
  // size if the last flag is positive
  if (size > 0 && flags.get_value (size - 1) > 0) out_size++;

  Array<int32> output;
  output.resize (out_size);
  int32 *output_ptr = output.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    int32 in_flag = flags_ptr[i];
    // if the flag is valid gather the sparse intput into
    // the compact output
    if (in_flag > 0)
    {
      const int32 out_idx = offsets_ptr[i];
      output_ptr[out_idx] = i;
    }
  });
  DRAY_ERROR_CHECK();

  return output;
}

template<int NComps>
GridFunction<NComps>
subset_grid_function(GridFunction<NComps> &input_gf, Array<int32> &flags)
{
  const int32 dofs_per_elem = input_gf.m_el_dofs;
  //std::cout<<"dofs per "<<dofs_per_elem<<"\n";
  Array<int32> conn = input_gf.m_ctrl_idx;
  Array<Vec<Float,NComps>> values = input_gf.m_values;

  // expand the cell based flags to length of connectivity
  const int32 expanded_size = conn.size();
  Array<int32> expanded_flags;
  expanded_flags.resize(expanded_size);

  int32 *expanded_flags_ptr = expanded_flags.get_device_ptr();
  const int32 *flags_ptr = flags.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, expanded_size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 index = i / dofs_per_elem;
    expanded_flags_ptr[i] = flags_ptr[index];
  });


  // compact the connectivity
  Array<int32> compact_conn_idx = index_flags(expanded_flags);
  Array<int32> compacted_conn = gather(conn, compact_conn_idx);

  // now we have to figure out which values we keep and compact
  const int32 compacted_size = compacted_conn.size();
  Array<int32> value_flags;
  value_flags.resize(values.size());
  array_memset_zero(value_flags);

  int32 *value_flags_ptr = value_flags.get_device_ptr();
  int32 *compacted_conn_ptr = compacted_conn.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, compacted_size), [=] DRAY_LAMBDA (int32 i)
  {
    // just let the race conditions happen
    const int32 index = compacted_conn_ptr[i];
    value_flags_ptr[index] = 1;
  });

  // compact the connectivity
  Array<int32> offsets; // we need the offsets to remap the connectivity
  Array<int32> compact_values_idx = index_flags_with_offses(value_flags, offsets);
  Array<Vec<Float, NComps>> compacted_values = gather(values, compact_values_idx);

  // remap the connectivity array to the compact values
  int32 *offset_ptr = offsets.get_device_ptr();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, compacted_size), [=] DRAY_LAMBDA (int32 i)
  {
    compacted_conn_ptr[i] = offset_ptr[compacted_conn_ptr[i]];
  });

  //std::cout<<"compacted conn size "<<compacted_size<<" in size "<<conn.size()<<"\n";
  //std::cout<<"compacted values size "<<compacted_values.size()<<" in size "<<values.size()<<"\n";

  GridFunction<NComps> output_gf;
  output_gf.m_el_dofs = input_gf.m_el_dofs;
  output_gf.m_size_el = compacted_conn.size() / output_gf.m_el_dofs;
  output_gf.m_size_ctrl = compacted_conn.size();
  output_gf.m_ctrl_idx = compacted_conn;
  output_gf.m_values = compacted_values;

  return output_gf;
}

struct SubsetTopologyFunctor
{
  DataSet m_res;
  Array<int32> m_flags;
  SubsetTopologyFunctor(Array<int32> &flags)
    : m_flags(flags)
  {
  }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    GridFunction<3u> mesh_gf = mesh.get_dof_data();
    GridFunction<3u> output = detail::subset_grid_function(mesh_gf, m_flags);
    MeshType omesh(output, mesh.order());
    m_res = DataSet(std::make_shared<MeshType>(omesh));
  }
};

struct SubsetFieldFunctor
{
  DataSet &m_dataset;
  Array<int32> m_flags;
  SubsetFieldFunctor(DataSet &dataset, Array<int32> &flags)
    : m_dataset(dataset),
      m_flags(flags)
  {
  }

  template<typename ElemType>
  void operator()(UnstructuredField<ElemType> &field)
  {
    GridFunction<ElemType::get_ncomp()> input_gf = field.get_dof_data();
    GridFunction<ElemType::get_ncomp()> output_gf = detail::subset_grid_function(input_gf, m_flags);
    int32 order = field.order();
    UnstructuredField<ElemType> output_field(output_gf, order, field.name());
    m_dataset.add_field(std::make_shared<UnstructuredField<ElemType>>(output_field));
  }
};

}//namespace detail

Subset::Subset()
{
}

DataSet
Subset::execute(DataSet &dataset, Array<int32> &cell_mask)
{
  DRAY_LOG_OPEN("subset");
  DRAY_LOG_ENTRY("input_cells", dataset.mesh()->cells());
  DataSet res;
  detail::SubsetTopologyFunctor func(cell_mask);
  dispatch(dataset.mesh(), func);
  res = func.m_res;

  DRAY_LOG_ENTRY("output_cells", res.mesh()->cells());
  const int num_fields = dataset.number_of_fields();
  for(int i = 0; i < num_fields; ++i)
  {
    Field *field = dataset.field(i);
    detail::SubsetFieldFunctor field_func(res, cell_mask);
    dispatch(field, field_func);
  }

  DRAY_LOG_CLOSE();
  return res;
}


}//namespace dray
