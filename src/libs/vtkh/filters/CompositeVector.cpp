//#include <vtkm/filter/your_vtkm_filter.h>
#include <vtkh/filters/CompositeVector.hpp>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkh/Error.hpp>

namespace vtkh
{

namespace detail
{

std::string to_string(vtkm::cont::Field::Association assoc)
{
  std::string res = "unknown";
  if(assoc == vtkm::cont::Field::Association::WholeMesh)
  {
    res = "whole mesh";
  }
  else if(assoc == vtkm::cont::Field::Association::Any)
  {
    res = "any";
  }
  else if(assoc == vtkm::cont::Field::Association::Points)
  {
    res = "points";
  }
  else if(assoc == vtkm::cont::Field::Association::Cells)
  {
    res = "cell set";
  }
  return res;
}

class MakeCompositeVector : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  MakeCompositeVector()
  {}

  typedef void ControlSignature(FieldIn, FieldIn, FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2, _3, _4);

  template<typename T, typename U, typename V>
  VTKM_EXEC
  void operator()(const T &value1,
                  const U &value2,
                  const V &value3,
                  vtkm::Vec<vtkm::Float64,3> &output) const
  {
    output[0] = static_cast<vtkm::Float64>(value1);
    output[1] = static_cast<vtkm::Float64>(value2);
    output[2] = static_cast<vtkm::Float64>(value3);
  }
};

class MakeVector2d : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  MakeVector2d()
  {}

  typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2, _3);

  template<typename T, typename U>
  VTKM_EXEC
  void operator()(const T &value1,
                  const U &value2,
                  vtkm::Vec<vtkm::Float64,2> &output) const
  {
    output[0] = static_cast<vtkm::Float64>(value1);
    output[1] = static_cast<vtkm::Float64>(value2);
  }
};

}// namespace detail

CompositeVector::CompositeVector()
{

}

CompositeVector::~CompositeVector()
{

}

void
CompositeVector::SetFields(const std::string &field1,
                    const std::string &field2,
                    const std::string &field3)
{
  m_field_1 = field1;
  m_field_2 = field2;
  m_field_3 = field3;
  m_mode_3d = true;
}

void
CompositeVector::SetFields(const std::string &field1,
                    const std::string &field2)
{
  m_field_1 = field1;
  m_field_2 = field2;
  m_mode_3d = false;
}

void
CompositeVector::SetResultField(const std::string &result_name)
{
  m_result_name = result_name;
}

void CompositeVector::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_1);
  Filter::CheckForRequiredField(m_field_2);
  if(m_mode_3d)
  {
    Filter::CheckForRequiredField(m_field_3);
  }

  vtkm::Id field_1_comps = this->m_input->NumberOfComponents(m_field_1);
  vtkm::Id field_2_comps = this->m_input->NumberOfComponents(m_field_2);

  vtkm::Id min_comps = std::min(field_1_comps,field_2_comps);
  vtkm::Id max_comps = std::max(field_1_comps, field_2_comps);

  vtkm::Id field_3_comps;
  if(m_mode_3d)
  {
    field_3_comps = this->m_input->NumberOfComponents(m_field_3);
    min_comps = std::min(min_comps, field_3_comps);
    max_comps = std::max(max_comps, field_3_comps);
  }

  if((min_comps != 1) || (min_comps != max_comps))
  {
    std::stringstream ss;
    ss<<"CompositeVector: all fields need to be scalars. ";
    ss<<"'"<<m_field_1<<"' has "<<field_1_comps<<". ";
    ss<<"'"<<m_field_2<<"' has "<<field_2_comps<<". ";
    if(m_mode_3d)
    {
      ss<<"'"<<m_field_3<<"' has "<<field_3_comps<<". ";
    }
    throw Error(ss.str());
  }

  bool valid;
  vtkm::cont::Field::Association assoc_1 =
    this->m_input->GetFieldAssociation(m_field_1, valid);

  vtkm::cont::Field::Association assoc_2 =
    this->m_input->GetFieldAssociation(m_field_2, valid);

  vtkm::cont::Field::Association assoc_3;
  bool same_assoc = (assoc_1 == assoc_2);
  if(m_mode_3d)
  {
    assoc_3 = this->m_input->GetFieldAssociation(m_field_3, valid);
    same_assoc &= assoc_1 == assoc_3;
  }


  if(!same_assoc)
  {
    std::stringstream ss;
    ss<<"CompositeVector: all fields need to have same associations. ";
    ss<<"'"<<m_field_1<<"' is "<<detail::to_string(assoc_1)<<". ";
    ss<<"'"<<m_field_2<<"' is "<<detail::to_string(assoc_2)<<". ";
    if(m_mode_3d)
    {
      ss<<"'"<<m_field_3<<"' is "<<detail::to_string(assoc_3)<<". ";
    }
    throw Error(ss.str());
  }

  if(m_result_name == "")
  {
    throw Error("Vector: result name never set");
  }
}

void CompositeVector::PostExecute()
{
  Filter::PostExecute();
}

void CompositeVector::DoExecute()
{
  this->m_output = new DataSet();
  // shallow copy input data set and bump internal ref counts
  *m_output = *m_input;

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet &dom =  this->m_output->GetDomain(i);

    if(!dom.HasField(m_field_1))
    {
      continue;
    }

    vtkm::cont::Field in_field_1 = dom.GetField(m_field_1);
    vtkm::cont::Field in_field_2 = dom.GetField(m_field_2);

    if(m_mode_3d)
    {
      vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>> vec_field;
      vtkm::cont::Field in_field_3 = dom.GetField(m_field_3);

      vtkm::worklet::DispatcherMapField<detail::MakeCompositeVector>(detail::MakeCompositeVector())
        .Invoke(in_field_1.GetData().ResetTypes(vtkm::TypeListFieldScalar(),
                                                VTKM_DEFAULT_STORAGE_LIST{}),
                in_field_2.GetData().ResetTypes(vtkm::TypeListFieldScalar(),
                                                VTKM_DEFAULT_STORAGE_LIST{}),
                in_field_3.GetData().ResetTypes(vtkm::TypeListFieldScalar(),
                                                VTKM_DEFAULT_STORAGE_LIST{}),
                vec_field);

      vtkm::cont::Field out_field(m_result_name,
                                  in_field_1.GetAssociation(),
                                  vec_field);
      dom.AddField(out_field);
    }
    else
    {
      vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>> vec_field;

      vtkm::worklet::DispatcherMapField<detail::MakeVector2d>(detail::MakeVector2d())
        .Invoke(in_field_1.GetData().ResetTypes(vtkm::TypeListFieldScalar(),
                                                VTKM_DEFAULT_STORAGE_LIST{}),
                in_field_2.GetData().ResetTypes(vtkm::TypeListFieldScalar(),
                                                VTKM_DEFAULT_STORAGE_LIST{}),
                vec_field);

      vtkm::cont::Field out_field(m_result_name,
                                  in_field_1.GetAssociation(),
                                  vec_field);
      dom.AddField(out_field);

    }
  }
}

std::string
CompositeVector::GetName() const
{
  return "vtkh::CompositeVector";
}

} //  namespace vtkh
