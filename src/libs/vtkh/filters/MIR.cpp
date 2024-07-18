#include "MIR.hpp"

#include <vtkm/filter/contour/MIRFilter.h>

namespace vtkh
{

namespace detail
{

bool
isMaterial(std::string matset_name, std::string field_name)
{

  // Create the substring to search for
  std::string searchString = matset_name + "_";

  // Check if the fieldName contains the searchString
  if (field_name.find(searchString) != std::string::npos) {
      return true;
  }
  return false;
}

}//end detail

MIR::MIR()
{

}

MIR::~MIR()
{

}

void
MIR::SetMatSet(const std::string matset_name)
{
  m_matset_name = matset_name;
  m_lengths_name = "sizes";// matset_name + "_lengths";
  m_offsets_name = "offsets";// matset_name + "_offsets";
  m_ids_name = "material_ids";// matset_name + "_ids";
  m_vfs_name = "volume_fractions";// matset_name + "_vfs";
}

void 
MIR::SetErrorScaling(const double error_scaling)
{
  m_error_scaling = error_scaling;
}

void 
MIR::SetScalingDecay(const double scaling_decay)
{
  m_scaling_decay = scaling_decay;
}

void 
MIR::SetIterations(const int iterations)
{
  m_iterations = iterations;
}

void 
MIR::SetMaxError(const double max_error)
{
  m_max_error = max_error;
}

void
MIR::PreExecute()
{
  Filter::PreExecute();
  std::string lengths_field ="sizes";// m_matset_name + "_lengths";
  std::string offsets_field = "offsets";//m_matset_name + "_offsets";
  std::string ids_field = "material_ids";//m_matset_name + "_ids";
  std::string vfs_field = "volume_fractions";//m_matset_name + "_vfs";

  Filter::CheckForRequiredField(lengths_field);
  Filter::CheckForRequiredField(offsets_field);
  Filter::CheckForRequiredField(ids_field);
  Filter::CheckForRequiredField(vfs_field);
}

void
MIR::PostExecute()
{
  Filter::PostExecute();
}

void MIR::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();
  //set fake discret color table
  vtkm::Range ids_range = this->m_input->GetGlobalRange(m_ids_name).ReadPortal().Get(0);

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    vtkm::filter::contour::MIRFilter mir; 
    mir.SetLengthCellSetName(m_lengths_name);
    mir.SetPositionCellSetName(m_offsets_name);
    mir.SetIDWholeSetName(m_ids_name);
    mir.SetVFWholeSetName(m_vfs_name);
    mir.SetErrorScaling(vtkm::Float64(m_error_scaling));
    mir.SetScalingDecay(vtkm::Float64(m_scaling_decay));
    mir.SetMaxIterations(vtkm::IdComponent(m_iterations));
    mir.SetMaxPercentError(vtkm::Float64(m_max_error));
    vtkm::cont::DataSet output = mir.Execute(dom);
    //cast and call error if cellMat stays as ints
    vtkm::cont::UnknownArrayHandle float_field = output.GetField("cellMat").GetDataAsDefaultFloat();
    vtkm::cont::Field::Association field_assoc = output.GetField("cellMat").GetAssociation();
    vtkm::cont::Field matset_field(m_matset_name,field_assoc, float_field);
    output.AddField(matset_field);
    //output.GetField("cellMat").SetData(float_field);
    this->m_output->AddDomain(output, i);
//    this->m_output->AddDomain(dom, i); //original data
  }
}

std::string
MIR::GetName() const
{
  return "vtkh::MIR";
}

} //  namespace vtkh
