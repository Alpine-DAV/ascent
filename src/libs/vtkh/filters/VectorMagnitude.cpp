#include <vtkh/Error.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>

#include <vtkh/vtkm_filters/vtkmVectorMagnitude.hpp>
//TODO: new wrapped vtkm filter header
//#include <vtkm/filter/vector_analysis/VectorMagnitude.h>
#include <vtkm/filter/vector_analysis/worklet/Magnitude.h>
#include <vtkm/TypeList.h>

namespace vtkh
{

namespace detail
{
//TODO: Rewrite using vtkm::filter::vector_analysis::VectorMagnitude
struct VectorMagFunctor
{
  vtkm::cont::Field result;
  std::string m_result_name;
  vtkm::cont::Field::Association m_assoc;
  template<typename T, vtkm::IdComponent Size, typename S>
  void operator()(const vtkm::cont::ArrayHandle<vtkm::Vec<T,Size>,S> &array)
  {

     vtkm::cont::ArrayHandle<T> mag_res;
     vtkm::worklet::DispatcherMapField<vtkm::worklet::Magnitude>()
          .Invoke(array, mag_res);

     result = vtkm::cont::Field(m_result_name, m_assoc, mag_res);
  }
};

} // namespace detail
VectorMagnitude::VectorMagnitude()
{

}

VectorMagnitude::~VectorMagnitude()
{

}

void
VectorMagnitude::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
VectorMagnitude::SetResultName(const std::string name)
{
  m_out_name = name;
}

void VectorMagnitude::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);

  if(m_out_name == "")
  {
    m_out_name = m_field_name + "_magnitude";
  }
}

void VectorMagnitude::PostExecute()
{
  Filter::PostExecute();
}

void VectorMagnitude::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkm::cont::Field field = dom.GetField(m_field_name);
    detail::VectorMagFunctor mag_func;
    mag_func.m_result_name = m_out_name;
    mag_func.m_assoc = field.GetAssociation();
    field.GetData().ResetTypes(vtkm::TypeListVecCommon(), VTKM_DEFAULT_STORAGE_LIST{}).CastAndCall(mag_func);
    vtkm::cont::DataSet dataset = dom;
    dataset.AddField(mag_func.result);
    // The current vtkm vector mag does not support
    // vec2s. One day we might be able to use it again
    //
    //vtkh::vtkmVectorMagnitude mag;
    //auto dataset = mag.Run(dom,
    //                       m_field_name,
    //                       m_out_name,
    //                       this->GetFieldSelection());

    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
VectorMagnitude::GetName() const
{
  return "vtkh::VectorMagnitude";
}

} //  namespace vtkh
