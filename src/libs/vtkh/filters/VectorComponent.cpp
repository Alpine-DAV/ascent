//#include <vtkm/filter/your_vtkm_filter.h>
#include <vtkh/filters/VectorComponent.hpp>
#include <vtkh/Error.hpp>

#include <vtkm/cont/ArrayHandleExtractComponent.h>
#include <vtkm/cont/Algorithm.h>


namespace vtkh
{

namespace detail
{

struct VectorCompositeFunctor
{
  int m_component;
  vtkm::cont::Field m_in_field;
  vtkm::cont::Field m_out_field;
  std::string m_name;

  template<typename T, vtkm::IdComponent Size, typename S>
  void operator()(const vtkm::cont::ArrayHandle<vtkm::Vec<T,Size>,S> &array)
  {
    auto comp_handle = vtkm::cont::make_ArrayHandleExtractComponent(array, m_component);
    vtkm::cont::ArrayHandle<T> result;
    vtkm::cont::Algorithm::Copy(comp_handle, result);

    m_out_field = vtkm::cont::Field(m_name, m_in_field.GetAssociation(), result);
  }
};

}// namespace detail

VectorComponent::VectorComponent()
  : m_component(-1)
{

}

VectorComponent::~VectorComponent()
{

}

void
VectorComponent::SetField(const std::string &field)
{
  m_field_name = field;
}

void
VectorComponent::SetComponent(const int component)
{
  m_component = component;
}

void
VectorComponent::SetResultField(const std::string &result_name)
{
  m_result_name = result_name;
}

void VectorComponent::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);

  vtkm::Id comps = this->m_input->NumberOfComponents(m_field_name);

  if(comps == 1)
  {
    throw Error("VectorComponent: input field is a scalar");
  }

  if(m_component == -1)
  {
    throw Error("VectorComponent: component never set");
  }

  if(m_component >= comps)
  {
    std::stringstream ss;
    ss<<"VectorComponent: component("<<m_component<<") is greater than";
    ss<<" the number of field components("<<comps<<").";
    ss<<" Note: the component should be zero indexed";
    throw Error(ss.str());
  }

  if(m_result_name == "")
  {
    throw Error("VectorComponent: result name never set");
  }
}

void VectorComponent::PostExecute()
{
  Filter::PostExecute();
}

void VectorComponent::DoExecute()
{
  this->m_output = new DataSet();
  // shallow copy input data set and bump internal ref counts
  *m_output = *m_input;

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet &dom =  this->m_output->GetDomain(i);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkm::cont::Field in_field = dom.GetField(m_field_name);
    detail::VectorCompositeFunctor func;
    func.m_component = m_component;
    func.m_in_field = in_field;
    func.m_name = m_result_name;

    in_field.GetData().ResetTypes(vtkm::TypeListVecCommon(),VTKM_DEFAULT_STORAGE_LIST{}).CastAndCall(func);

    dom.AddField(func.m_out_field);
  }
}

std::string
VectorComponent::GetName() const
{
  return "vtkh::VectorComponent";
}

} //  namespace vtkh
