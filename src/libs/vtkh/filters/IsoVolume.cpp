#include "IsoVolume.hpp"

#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/CleanGrid.hpp>

namespace vtkh
{

IsoVolume::IsoVolume()
{

}

IsoVolume::~IsoVolume()
{

}

void
IsoVolume::SetRange(const vtkm::Range range)
{
  m_range = range;
}

void
IsoVolume::SetField(const std::string field_name)
{
  m_field_name = field_name;
}

void
IsoVolume::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void
IsoVolume::PostExecute()
{
  Filter::PostExecute();
}

void IsoVolume::DoExecute()
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
IsoVolume::GetName() const
{
  return "vtkh::IsoVolume";
}

} //  namespace vtkh
