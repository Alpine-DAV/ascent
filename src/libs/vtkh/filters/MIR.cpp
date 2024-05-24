#include "MIR.hpp"

#include <vtkh/vtkm_filters/vtkmMIR.hpp>

namespace vtkh
{

MIR::MIR()
{

}

MIR::~MIR()
{

}

void
MIR::SetField(const std::string field_name)
{
  m_field_name = field_name;
}

void
MIR::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void
MIR::PostExecute()
{
  Filter::PostExecute();
}

void MIR::DoExecute()
{

  ClipField max_clip;
  max_clip.SetInput(this->m_input);
  max_clip.SetField(m_field_name);
  max_clip.SetClipValue(m_range.Max);
  max_clip.SetInvertClip(true);
  max_clip.Update();

  DataSet *clipped = max_clip.GetOutput();

  ClipField min_clip;
  min_clip.SetInput(clipped);
  min_clip.SetField(m_field_name);
  min_clip.SetClipValue(m_range.Min);
  min_clip.Update();

  delete clipped;
  DataSet *iso = min_clip.GetOutput();
  CleanGrid cleaner;
  cleaner.SetInput(iso);
  cleaner.Update();
  delete iso;
  this->m_output = cleaner.GetOutput();

}

std::string
MIR::GetName() const
{
  return "vtkh::MIR";
}

} //  namespace vtkh
