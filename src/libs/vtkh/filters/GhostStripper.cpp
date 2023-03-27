#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>
#include <vtkh/vtkm_filters/vtkmGhostStripper.hpp>
#include <vtkh/vtkm_filters/vtkmThreshold.hpp>
#include <vtkh/vtkm_filters/vtkmCleanGrid.hpp>
#include <vtkh/vtkm_filters/vtkmExtractStructured.hpp>

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
    if(!dom.HasField(m_field_name))
    {
      m_output->AddDomain(dom,domain_id);
    }
    else
    {
      vtkmGhostStripper stripper;
      vtkm::cont::DataSet stripper_out = stripper.Run(dom, m_field_name);
      m_output->AddDomain(stripper_out,domain_id);
    }

  }

}

std::string
GhostStripper::GetName() const
{
  return "vtkh::GhostStripper";
}

} //  namespace vtkh
