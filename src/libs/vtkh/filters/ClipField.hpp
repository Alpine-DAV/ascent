#ifndef VTK_H_CLIP_FIELD_HPP
#define VTK_H_CLIP_FIELD_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>
#include <memory>

namespace vtkh
{

class VTKH_API ClipField: public Filter
{
public:
  ClipField();
  virtual ~ClipField();
  std::string GetName() const override;
  void SetClipValue(const vtkm::Float64 clip_value);
  void SetField(const std::string field_name);
  void SetInvertClip(const bool invert);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  vtkm::Float64 m_clip_value;
  std::string   m_field_name;
  bool m_invert;
};

} //namespace vtkh
#endif
