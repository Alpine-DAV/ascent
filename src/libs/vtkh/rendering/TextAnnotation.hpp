#ifndef VTK_H_TEXT_ANNOTATION_HPP
#define VTK_H_TEXT_ANNOTATION_HPP

#include <string>

#include <viskores/Types.h>
#include <viskores/rendering/Color.h>
#include <vtkh/vtkh_exports.h>

namespace vtkh
{

struct VTKH_API TextAnnotation
{
  enum class PositionType
  {
    Screen,
    World
  };

  TextAnnotation()
    : m_text(""),
      m_position(0.f, 0.f, 0.f),
      m_position_type(PositionType::Screen),
      m_scale(0.045f),
      m_angle(0.f),
      m_horizontal_alignment(-1.f),
      m_vertical_alignment(-1.f),
      m_color(1.f, 1.f, 1.f, 1.f),
      m_use_color(false)
  {}

  std::string m_text;
  viskores::Vec3f_32 m_position;
  PositionType m_position_type;
  viskores::Float32 m_scale;
  viskores::Float32 m_angle;
  viskores::Float32 m_horizontal_alignment;
  viskores::Float32 m_vertical_alignment;
  viskores::rendering::Color m_color;
  bool m_use_color;
};

} // namespace vtkh

#endif
