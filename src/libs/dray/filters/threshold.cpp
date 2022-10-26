#include <dray/filters/threshold.hpp>

#include <dray/dispatcher.hpp>
#include <dray/filters/subset.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{

// Iterate over the thresh_field to determine an element mask of cells to preserve.
template <typename FieldType>
void
determine_elements_to_keep(FieldType &thresh_field,
    Range &range, bool return_all_in_range, Array<int32> &elem_mask)
{
  int32 nelem = thresh_field.get_dof_data().get_num_elem();
  elem_mask.resize(nelem);
  const auto thresh_ptr = thresh_field.get_dof_data().m_values.get_device_ptr();
  const auto conn_ptr = thresh_field.get_dof_data().m_ctrl_idx.get_device_ptr();
  const auto elem_mask_ptr = elem_mask.get_device_ptr();

  if(thresh_field.order() == 0)
  {
    // cell-centered. This means that the grid function contains m_ctrl_idx
    // that contains a list of 0..Ncells-1 values that we need to check to
    // decide whether to keep the cell.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
    {
      elem_mask_ptr[elid] = thresh_ptr[elid][0] >= range.min() && 
                            thresh_ptr[elid][0] <= range.max();
    });
    DRAY_ERROR_CHECK();
  }
  else
  {
    // node/dof-centered. Each cell contains a list of dofs it uses.
    auto el_dofs = thresh_field.get_dof_data().m_el_dofs;
    if(return_all_in_range)
    {
      // Keep if all values are in range.
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
      {
        // Examine all dof values for this cell and see whether we should
        // keep the cell.
        int32 keep = 1;
        int32 start = elid * el_dofs;
        for(int j = 0; j < el_dofs; j++)
        {
          int32 dofid = conn_ptr[start + j];
          keep &= thresh_ptr[dofid][0] >= range.min() && 
                  thresh_ptr[dofid][0] <= range.max();
        }
        elem_mask_ptr[elid] = keep;
      });
      DRAY_ERROR_CHECK();
    }
    else
    {
      // Keep if any values are in range.
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
      {
        // Examine all dof values for this cell and see whether we should
        // keep the cell.
        int32 keep = 0;
        int32 start = elid * el_dofs;
        for(int j = 0; j < el_dofs; j++)
        {
          int32 dofid = conn_ptr[start + j];
          keep |= thresh_ptr[dofid][0] >= range.min() && 
                  thresh_ptr[dofid][0] <= range.max();
        }
        elem_mask_ptr[elid] = keep;
      });
      DRAY_ERROR_CHECK();
    }
  }
}

// Applies a threshold operation on a DataSet.
struct ThresholdFunctor
{
  // Keep a handle to the original dataset because we need it to be able to
  // access the other fields.
  DataSet m_input;

  // Output dataset produced by the functor.
  DataSet m_output;

  // Threshold attributes.
  std::string m_field_name;
  Range m_range;
  bool m_return_all_in_range;

  ThresholdFunctor(DataSet &input, const std::string &field_name,
     const Range range, bool return_all_in_range)
    : m_input(input), m_output(), m_field_name(field_name),
      m_range(range), m_return_all_in_range(return_all_in_range)
  {
  }

  // Execute the filter for the input mesh across all possible field types.
  void execute()
  {
    // This iterates over the product of possible mesh and scalar field types
    // to call the operator() function that can operate on concrete types.
    Field *field = m_input.field(m_field_name);
    if(field != nullptr && field->components() == 1)
    {
      dispatch(field, *this);
    }
  }

  // This method gets invoked by dispatch, which will have converted the field
  // into a concrete derived type so this method is able to call methods on
  // the derived type with no virtual calls.
  template<typename ScalarField>
  void operator()(ScalarField &field)
  {
    DRAY_LOG_OPEN("mesh_threshold");

    // If the field range and threshold ranges overlap then we can check for
    // cells that overlap.
    auto range = field.range();
    if(m_range.contains(range[0].min()) ||
       m_range.contains(range[0].max()) ||
       range[0].contains(m_range.min()) ||
       range[0].contains(m_range.max()))
    {
      // Figure out which elements to keep based on the input field.
      Array<int32> elem_mask;
      determine_elements_to_keep(field, m_range, m_return_all_in_range, elem_mask);
#if 0
      std::cout << "elem_mask={";
      for(size_t i = 0; i < elem_mask.size(); i++)
          std::cout << ", " << elem_mask.get_value(i);
      std::cout << "}" << std::endl;
#endif
      // Use the element mask to subset the data.
      dray::Subset subset;
      m_output = subset.execute(m_input, elem_mask);
    }

    DRAY_LOG_CLOSE();
  }
};


}//namespace detail


Threshold::Threshold() : m_range(), m_field_name(), 
  m_return_all_in_range(false)
{
}

Threshold::~Threshold()
{
}

void
Threshold::set_upper_threshold(Float value)
{
  m_range.set_max(value);
}

void
Threshold::set_lower_threshold(Float value)
{
  m_range.set_min(value);
}

void
Threshold::set_field(const std::string &field_name)
{
  m_field_name = field_name;
}

void
Threshold::set_all_in_range(bool value)
{
  m_return_all_in_range = value;
}

Float
Threshold::get_upper_threshold() const
{
  return m_range.max();
}

Float
Threshold::get_lower_threshold() const
{
  return m_range.min();
}

bool
Threshold::get_all_in_range() const
{
  return m_return_all_in_range;
}

const std::string &
Threshold::get_field() const
{
  return m_field_name;
}

Collection
Threshold::execute(Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dom = collection.domain(i);
    detail::ThresholdFunctor func(dom, m_field_name, m_range, m_return_all_in_range);
    func.execute();
    res.add_domain(func.m_output);
  }
  return res;
}


}//namespace dray
