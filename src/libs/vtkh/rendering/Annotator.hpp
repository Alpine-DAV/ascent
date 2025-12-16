#ifndef VTKH_ANNOTATOR_HPP
#define VTKH_ANNOTATOR_HPP

#include <viskores/rendering/AxisAnnotation3D.h>
#include <viskores/rendering/BoundingBoxAnnotation.h>
#include <viskores/rendering/Canvas.h>
#include <viskores/rendering/Camera.h>
#include <viskores/rendering/ColorBarAnnotation.h>
#include <viskores/rendering/WorldAnnotator.h>

#include<vtkh/vtkh_exports.h>

namespace vtkh
{

class VTKH_API Annotator
{
public:
  Annotator(viskores::rendering::Canvas &canvas,
            viskores::rendering::Camera &camera,
            viskores::Bounds bounds);
  ~Annotator();

  void RenderWorldAnnotations(viskores::Vec<float,3> axis_scale);
  void RenderScreenAnnotations(const std::vector<std::string> &field_names,
                               const std::vector<viskores::Range> &ranges,
                               const std::vector<viskores::cont::ColorTable> &color_tables,
                               const std::vector<int> &is_discrete);

  void RenderScreenAnnotations(const std::vector<std::string> &field_names,
                               const std::vector<viskores::Range> &ranges,
                               const std::vector<viskores::cont::ColorTable> &color_tables,
			                         const std::vector<viskores::Bounds> &color_bar_position,
			                         const std::vector<int> &is_discrete);
protected:
  Annotator();
  bool                                    m_is_3d;
  viskores::rendering::Canvas                &m_canvas;
  viskores::rendering::Camera                &m_camera;
  viskores::Bounds                            m_bounds;
  viskores::rendering::BoundingBoxAnnotation  m_box_annotation;
  viskores::rendering::AxisAnnotation3D       m_x_axis_annotation;
  viskores::rendering::AxisAnnotation3D       m_y_axis_annotation;
  viskores::rendering::AxisAnnotation3D       m_z_axis_annotation;
  viskores::rendering::ColorBarAnnotation     m_color_bar_annotation;
  viskores::rendering::WorldAnnotator        *m_world_annotator;
  std::vector<viskores::Bounds>               m_color_bar_pos;
  //std::vector<viskores::rendering::TextAnnotation*> m_text_annotations;
  //void RenderScreen2DAnnotations(viskores::Range range, const  std::string &field_name);
  //void RenderScreen3DAnnotations(viskores::Range range, const std::string &field_name);
};

}// namespace vtkh
#endif
