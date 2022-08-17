#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>
#include <vtkh/vtkm_filters/vtkmThreshold.hpp>
#include <vtkh/vtkm_filters/vtkmCleanGrid.hpp>
#include <vtkh/vtkm_filters/vtkmExtractStructured.hpp>

#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/BinaryOperators.h>

#include <limits>

namespace vtkh
{

namespace detail
{
// only do reductions for positive numbers
struct MinMaxIgnore
{
  VTKM_EXEC_CONT
  vtkm::Vec<vtkm::Id, 2> operator()(const vtkm::Id& a) const
  {
    return vtkm::make_Vec(a, a);
  }

  VTKM_EXEC_CONT
  vtkm::Vec<vtkm::Id, 2> operator()(const vtkm::Vec<vtkm::Id, 2>& a,
                                    const vtkm::Vec<vtkm::Id, 2>& b) const
  {
    vtkm::Vec<vtkm::Id,2> min_max;
    if(a[0] >= 0 && b[0] >=0)
    {
      min_max[0] = vtkm::Min(a[0], b[0]);
    }
    else if(a[0] < 0)
    {
      min_max[0] = b[0];
    }
    else
    {
      min_max[0] = a[0];
    }

    if(a[1] >= 0 && b[1] >=0)
    {
      min_max[1] = vtkm::Max(a[1], b[1]);
    }
    else if(a[1] < 0)
    {
      min_max[1] = b[1];
    }
    else
    {
      min_max[1] = a[1];
    }
    return min_max;
  }

};

template<int DIMS>
VTKM_EXEC_CONT
vtkm::Vec<vtkm::Id,3> get_logical(const vtkm::Id &index, const vtkm::Vec<vtkm::Id,3> &cell_dims);

template<>
VTKM_EXEC_CONT
vtkm::Vec<vtkm::Id,3> get_logical<3>(const vtkm::Id &index, const vtkm::Vec<vtkm::Id,3> &cell_dims)
{
  vtkm::Vec<vtkm::Id,3> res(0,0,0);
  res[0] = index % cell_dims[0];
  res[1] = (index / (cell_dims[0])) % (cell_dims[1]);
  res[2] = index / ((cell_dims[0]) * (cell_dims[1]));
  return res;
}

template<>
VTKM_EXEC_CONT
vtkm::Vec<vtkm::Id,3> get_logical<2>(const vtkm::Id &index, const vtkm::Vec<vtkm::Id,3> &cell_dims)
{
  vtkm::Vec<vtkm::Id,3> res(0,0,0);
  res[0] = index % cell_dims[0];
  res[1] = index / cell_dims[0];
  return res;
}

template<>
VTKM_EXEC_CONT
vtkm::Vec<vtkm::Id,3> get_logical<1>(const vtkm::Id &index, const vtkm::Vec<vtkm::Id,3> &cell_dims)
{
  vtkm::Vec<vtkm::Id,3> res(0,0,0);
  res[0] = index;
  return res;
}

template<int DIMS>
class GhostIndex : public vtkm::worklet::WorkletMapField
{
protected:
  vtkm::Vec<vtkm::Id,3> m_cell_dims;
  vtkm::Int32 m_min_value;
  vtkm::Int32 m_max_value;
  vtkm::Id m_default_value;
  vtkm::Int32 m_dim;
public:
  VTKM_CONT
  GhostIndex(vtkm::Vec<vtkm::Id,3> cell_dims,
             vtkm::Int32 min_value,
             vtkm::Int32 max_value,
             vtkm::Id default_value,
             vtkm::Id dim)
    : m_cell_dims(cell_dims),
      m_min_value(min_value),
      m_max_value(max_value),
      m_default_value(default_value),
      m_dim(dim)
  {
  }

  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, WorkIndex, _2);

  template<typename T>
  VTKM_EXEC
  void operator()(const T &value, const vtkm::Id &index, vtkm::Id &ghost_index) const
  {

    // we are finding the logical min max of valid zones
    if( value < m_min_value || value > m_max_value)
    {
      ghost_index = m_default_value;
    }
    else
    {
      vtkm::Vec<vtkm::Id,3> logical = get_logical<DIMS>(index, m_cell_dims);
      ghost_index = logical[m_dim];
    }
  }
}; //class GhostIndex

template<int DIMS>
class CanStructuredStrip : public vtkm::worklet::WorkletMapField
{
protected:
  vtkm::Vec<vtkm::Id,3> m_cell_dims;
  vtkm::Int32 m_min_value;
  vtkm::Int32 m_max_value;
  vtkm::Vec<vtkm::Id,3> m_valid_min;
  vtkm::Vec<vtkm::Id,3> m_valid_max;
public:
  VTKM_CONT
  CanStructuredStrip(vtkm::Vec<vtkm::Id,3> cell_dims,
           vtkm::Int32 min_value,
           vtkm::Int32 max_value,
           vtkm::Vec<vtkm::Id,3> valid_min,
           vtkm::Vec<vtkm::Id,3> valid_max)
    : m_cell_dims(cell_dims),
      m_min_value(min_value),
      m_max_value(max_value),
      m_valid_min(valid_min),
      m_valid_max(valid_max)
  {
  }

  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, WorkIndex, _2);

  template<typename T>
  VTKM_EXEC
  void operator()(const T &value, const vtkm::Id &index, vtkm::UInt8 &can_do) const
  {
    can_do = 0; // this is a valid zone
    // we are validating if non-valid cells fall completely outside
    // the min max range of valid cells
    if(value >= m_min_value && value <= m_max_value) return;

    vtkm::Vec<vtkm::Id,3> logical = get_logical<DIMS>(index, m_cell_dims);
    bool inside = true;
    for(vtkm::Int32 i = 0; i < DIMS; ++i)
    {
      if(logical[i] < m_valid_min[i] || logical[i] > m_valid_max[i])
      {
        inside = false;
      }
    }
    // this is a 'ghost' zone that is inside the valid range
    // so we cannot structured strip
    if(inside)
    {
      can_do = 1;
    }

  }
}; //class CanStructuredStrip

template<int DIMS>
bool CanStrip(vtkm::cont::Field  &ghost_field,
              const vtkm::Int32 min_value,
              const vtkm::Int32 max_value,
              vtkm::Vec<vtkm::Id,3> &min,
              vtkm::Vec<vtkm::Id,3> &max,
              vtkm::Vec<vtkm::Id,3> cell_dims,
              vtkm::Id size,
              bool &should_strip)
{

  VTKH_DATA_OPEN("can_strip");
  vtkm::cont::ArrayHandle<vtkm::Id> dim_indices;

  vtkm::Vec<vtkm::Id, 3> valid_min = {0,0,0};
  vtkm::Vec<vtkm::Id, 3> valid_max = {0,0,0};

  for(vtkm::Int32 i = 0; i < DIMS; ++i)
  {
    vtkm::worklet::DispatcherMapField<GhostIndex<DIMS>>(
        GhostIndex<DIMS>(cell_dims,
                      min_value,
                      max_value,
                      -1,
                      i))
       .Invoke(ghost_field.GetData().ResetTypes(vtkm::TypeListScalarAll(),
                                                VTKM_DEFAULT_STORAGE_LIST{}),
           dim_indices);

    vtkm::Vec<vtkm::Id,2> d = {-1, -1};
    auto mm = vtkm::cont::Algorithm::Reduce(dim_indices,
                                            d,
                                            detail::MinMaxIgnore());

    valid_min[i] = mm[0];
    valid_max[i] = mm[1];
  }

  vtkm::cont::ArrayHandle<vtkm::UInt8> valid_flags;
  valid_flags.Allocate(size);

  min = valid_min;
  max = valid_max;

  vtkm::worklet::DispatcherMapField<CanStructuredStrip<DIMS>>
    (CanStructuredStrip<DIMS>(cell_dims,
                              min_value,
                              max_value,
                              valid_min,
                              valid_max))
     .Invoke(ghost_field.GetData().ResetTypes(vtkm::TypeListScalarAll(),
                                              VTKM_DEFAULT_STORAGE_LIST{}),
         valid_flags);

  vtkm::UInt8 res = vtkm::cont::Algorithm::Reduce(valid_flags,
                                                  vtkm::UInt8(0),
                                                  vtkm::Maximum());
  VTKH_DATA_CLOSE();

  bool can_strip = res == 0;
  if(can_strip)
  {
    should_strip = false;
    for(int i = 0; i < DIMS; ++i)
    {
      if(cell_dims[i] != (valid_max[i] - valid_min[i] + 1))
      {
        should_strip = true;
      }
    }
  }
  else
  {
    should_strip = true;
  }

  return can_strip;
}

bool StructuredStrip(vtkm::cont::DataSet &dataset,
                     vtkm::cont::Field   &ghost_field,
                     const vtkm::Int32 min_value,
                     const vtkm::Int32 max_value,
                     vtkm::Vec<vtkm::Id,3> &min,
                     vtkm::Vec<vtkm::Id,3> &max,
                     bool &should_strip)
{
  VTKH_DATA_OPEN("structured_strip");
  vtkm::cont::DynamicCellSet cell_set = dataset.GetCellSet();
  int dims[3];
  VTKMDataSetInfo::GetPointDims(cell_set, dims);
  vtkm::Vec<vtkm::Id,3> cell_dims(0,0,0);


  bool can_strip = false;
  vtkm::Id size = 0;
  should_strip = false;
  if(cell_set.IsSameType(vtkm::cont::CellSetStructured<1>()))
  {
    cell_dims[0] = dims[0] - 1;
    size = cell_dims[0];

    can_strip = CanStrip<1>(ghost_field,
                            min_value,
                            max_value,
                            min,
                            max,
                            cell_dims,
                            size,
                            should_strip);
  }
  else if(cell_set.IsSameType(vtkm::cont::CellSetStructured<2>()))
  {
    cell_dims[0] = dims[0] - 1;
    cell_dims[1] = dims[1] - 1;
    size = cell_dims[0] * cell_dims[1];

    can_strip = CanStrip<2>(ghost_field,
                            min_value,
                            max_value,
                            min,
                            max,
                            cell_dims,
                            size,
                            should_strip);
  }
  else if(cell_set.IsSameType(vtkm::cont::CellSetStructured<3>()))
  {
    cell_dims[0] = dims[0] - 1;
    cell_dims[1] = dims[1] - 1;
    cell_dims[2] = dims[2] - 1;
    size = cell_dims[0] * cell_dims[1] * cell_dims[2];

    can_strip = CanStrip<3>(ghost_field,
                            min_value,
                            max_value,
                            min,
                            max,
                            cell_dims,
                            size,
                            should_strip);
  }

  VTKH_DATA_CLOSE();
  return can_strip;
}

} // namespace detail

GhostStripper::GhostStripper()
  : m_min_value(0),  // default to real zones only
    m_max_value(0)   // 0 = real, 1 = valid ghost, 2 = garbage ghost
{

}

GhostStripper::~GhostStripper()
{

}

void
GhostStripper::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
GhostStripper::SetMinValue(const vtkm::Int32 min_value)
{
  m_min_value = min_value;
}

void
GhostStripper::SetMaxValue(const vtkm::Int32 max_value)
{
  m_max_value = max_value;
}

void GhostStripper::PreExecute()
{
  Filter::PreExecute();
  if(m_min_value > m_max_value)
  {
    throw Error("GhostStripper: min_value is greater than max value.");
  }
  Filter::CheckForRequiredField(m_field_name);
}

void GhostStripper::PostExecute()
{
  Filter::PostExecute();
}

void GhostStripper::DoExecute()
{
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {

    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkm::cont::Field field = dom.GetField(m_field_name);
    vtkm::Range ghost_range = field.GetRange().ReadPortal().Get(0);

    if(ghost_range.Min >= m_min_value &&
       ghost_range.Max <= m_max_value)
    {
      // nothing to do here
      m_output->AddDomain(dom, domain_id);
      continue;
    }

    int topo_dims = 0;
    bool do_threshold = true;

    if(VTKMDataSetInfo::IsStructured(dom, topo_dims))
    {
      vtkm::Vec<vtkm::Id,3> min, max;
      bool should_strip; // just because we can doesn't mean we should
      bool can_strip = detail::StructuredStrip(dom,
                                              field,
                                              m_min_value,
                                              m_max_value,
                                              min,
                                              max,
                                              should_strip);
      if(can_strip)
      {
        do_threshold = false;
        if(should_strip)
        {
          VTKH_DATA_OPEN("extract_structured");
          //vtkm::RangeId3 range(min[0],max[0]+1, min[1], max[1]+1, min[2], max[2]+1);
          vtkm::RangeId3 range(min[0],max[0]+2, min[1], max[1]+2, min[2], max[2]+2);
          vtkm::Id3 sample(1, 1, 1);

          vtkh::vtkmExtractStructured extract;
          auto output = extract.Run(dom,
                                    range,
                                    sample,
                                    this->GetFieldSelection());

          m_output->AddDomain(output, domain_id);
          VTKH_DATA_CLOSE();
        }
        else
        {
          // All zones are valid so just pass through
          m_output->AddDomain(dom, domain_id);
        }
      }

    }

    if(do_threshold)
    {
      vtkmThreshold thresholder;

      auto tout = thresholder.Run(dom,
                                  m_field_name,
                                  m_min_value,
                                  m_max_value,
                                  this->GetFieldSelection());

      vtkh::vtkmCleanGrid cleaner;
      auto clout = cleaner.Run(tout, this->GetFieldSelection());
      m_output->AddDomain(clout, domain_id);
    }

  }

}

std::string
GhostStripper::GetName() const
{
  return "vtkh::GhostStripper";
}

} //  namespace vtkh
