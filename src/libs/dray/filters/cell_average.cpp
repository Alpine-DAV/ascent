#include "cell_average.hpp"

#include <dray/data_model/data_set.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/dispatcher.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <iostream>

// Start internal implementation
namespace
{

using namespace dray;

template<typename FieldElemType>
static std::shared_ptr<Field>
compute_cell_average(const UnstructuredField<FieldElemType> &in_field,
                     const std::string &name)
{
  using OutElemType = Element<FieldElemType::get_dim(),
                              FieldElemType::get_ncomp(),
                              FieldElemType::get_etype(),
                              Order::Constant>;
  using VecType = Vec<Float, OutElemType::get_ncomp()>;

  // Retrieve important input information
  constexpr auto ncomp = FieldElemType::get_ncomp();
  const GridFunction<ncomp> &in_gf = in_field.get_dof_data();
  const auto *in_data_ptr = in_gf.m_values.get_device_ptr_const();
  const auto *in_idx_ptr = in_gf.m_ctrl_idx.get_device_ptr_const();

  // Create output array
  const auto nelem = in_field.get_num_elem();
  const auto nvalues = nelem * ncomp;
  Array<VecType> out_data;
  out_data.resize(nvalues);
  auto *out_data_ptr = out_data.get_device_ptr();

  // Execute parallel algorithm
  const RAJA::RangeSegment range(0, nelem);
  const auto ndof = in_gf.m_el_dofs;
  RAJA::forall<for_policy>(range,
    [=] DRAY_LAMBDA (int i)
    {
      const int idx_idx = i * ndof;
      Float sum[ncomp];
      // Initialize with value at dof=0
      for(int c = 0; c < ncomp; c++)
      {
        const int idx = in_idx_ptr[idx_idx];
        sum[c] = in_data_ptr[idx][c];
      }

      // Sum the values at each dof
      for(int dof = 1; dof < ndof; dof++)
      {
        const int idx = in_idx_ptr[idx_idx + dof];
        for(int c = 0; c < ncomp; c++)
        {
          sum[c] = sum[c] + in_data_ptr[idx][c];
        }
      }

      // Divide by ndof for the average
      for(int c = 0; c < ncomp; c++)
      {
        out_data_ptr[i][c] = sum[c] / Float(ndof);
      }
    });
  DRAY_ERROR_CHECK();

  // Build return value
  GridFunction<OutElemType::get_ncomp()> out_gf;
  out_gf.m_el_dofs = 1;
  out_gf.m_size_el = nelem;
  out_gf.m_values = out_data;
  // Q: Do we need conn for this?
  out_gf.m_size_ctrl = nelem;
  out_gf.m_ctrl_idx = array_counting(nelem, 0, 1);
  return std::make_shared<UnstructuredField<OutElemType>>(out_gf, Order::Constant, name);
}

struct CellAverageFunctor
{
  CellAverageFunctor() = delete;
  CellAverageFunctor(const std::string &name);
  ~CellAverageFunctor() = default;

  std::shared_ptr<Field> output() { return m_output; }

  // This method gets invoked by dispatch with a concrete field
  // type like UnstructuredField<T>.
  template<typename FieldType>
  void operator()(FieldType &field);

  std::shared_ptr<Field> m_output;
  std::string m_name;
};

CellAverageFunctor::CellAverageFunctor(const std::string &name)
  : m_output(), m_name(name)
{
  // Do nothing
}

template<typename FieldType>
inline void
CellAverageFunctor::operator()(FieldType &field)
{
  m_output = compute_cell_average(field, m_name);
}

/**
  @brief Need to ensure the output field created by this filter is on
         the output dataset. This means we cannot copy any field from
         the input that has out_name as its name.
*/
DataSet
initialize_output_domain(DataSet &domain, const std::string &out_name)
{
  DataSet retval(domain);

  retval.clear_fields();
  const int nfields = domain.number_of_fields();
  for(int i = 0; i < nfields; i++)
  {
    const auto ptr = domain.field_shared(i);
    if(ptr->name() == out_name)
    {
      // Skip the field with the same name as out_field
      continue;
    }
    retval.add_field(ptr);
  }
  return retval;
}

// End internal implementation
}

// Public interface
namespace dray
{

CellAverage::CellAverage()
  : in_field(), out_field()
{
  // Do nothing
}

CellAverage::~CellAverage()
{
  // Do nothing
}

void
CellAverage::set_field(const std::string &name)
{
  in_field = name;
}

void
CellAverage::set_output_field(const std::string &name)
{
  out_field = name;
}

Collection
CellAverage::execute(Collection &input)
{
  if(!input.has_field(in_field))
  {
    DRAY_ERROR("Cannot execute CellAverage for variable '" << in_field
            << "' because it does not exist in the given collection.");
  }

  // Default output name to input name.
  // If the user set a custom output name then use that.
  std::string out_name(in_field);
  if(!out_field.empty())
  {
    out_name = out_field;
  }

  Collection output;
  auto domains = input.domains();
  for(DataSet &domain : domains)
  {
    DataSet out_dom = initialize_output_domain(domain, out_name);
    if(domain.has_field(in_field))
    {
      Field *field = domain.field(in_field);
      CellAverageFunctor cellavg(out_name);
      try
      {
        // Covers Scalars and 3D vectors
        dispatch(field, cellavg);
      }
      catch(const DRayError &)
      {
        // Covers 2D / 3D vectors
        dispatch_vector(field, cellavg);
      }
      out_dom.add_field(cellavg.output());
    }
    output.add_domain(out_dom);
  }
  return output;
}

}
