#ifndef VTKH_ANNOTATOR_HPP
#define VTKH_ANNOTATOR_HPP

#include <vtkm/rendering/AxisAnnotation3D.h>
#include <vtkm/rendering/BoundingBoxAnnotation.h>
#include <vtkm/rendering/Canvas.h>
#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/ColorBarAnnotation.h>
#include <vtkm/rendering/WorldAnnotator.h>

#include<vtkh/vtkh_exports.h>

namespace vtkh
{

class VTKH_API Annotator
{
public:
  Annotator(vtkm::rendering::Canvas &canvas,
            vtkm::rendering::Camera &camera,
            vtkm::Bounds bounds);
  ~Annotator();

  void RenderWorldAnnotations(vtkm::Vec<float,3> axis_scale);
  void RenderScreenAnnotations(const std::vector<std::string> &field_names,
                               const std::vector<vtkm::Range> &ranges,
                               const std::vector<vtkm::cont::ColorTable> &color_tables);

protected:
  Annotator();
  bool                                    m_is_3d;
  vtkm::rendering::Canvas                &m_canvas;
  vtkm::rendering::Camera                &m_camera;
  vtkm::Bounds                            m_bounds;
  vtkm::rendering::BoundingBoxAnnotation  m_box_annotation;
  vtkm::rendering::AxisAnnotation3D       m_x_axis_annotation;
  vtkm::rendering::AxisAnnotation3D       m_y_axis_annotation;
  vtkm::rendering::AxisAnnotation3D       m_z_axis_annotation;
  vtkm::rendering::ColorBarAnnotation     m_color_bar_annotation;
  vtkm::rendering::WorldAnnotator        *m_world_annotator;
  std::vector<vtkm::Bounds>               m_color_bar_pos;
  //std::vector<vtkm::rendering::TextAnnotation*> m_text_annotations;
  //void RenderScreen2DAnnotations(vtkm::Range range, const  std::string &field_name);
  //void RenderScreen3DAnnotations(vtkm::Range range, const std::string &field_name);
};

}// namespace vtkh
#endif
