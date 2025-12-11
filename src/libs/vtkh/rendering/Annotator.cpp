#include <vtkh/rendering/Annotator.hpp>

namespace vtkh
{

Annotator::Annotator(viskores::rendering::Canvas &canvas,
                     viskores::rendering::Camera &camera,
                     viskores::Bounds bounds)
  : m_canvas(canvas),
    m_camera(camera),
    m_bounds(bounds)
{
  m_is_3d = m_camera.GetMode() == viskores::rendering::Camera::Mode::ThreeD;
  m_world_annotator = m_canvas.CreateWorldAnnotator();
  // add defualt color bar positions
  viskores::Bounds p1(viskores::Range(0.84, 0.92), viskores::Range(+0.1, +0.8), viskores::Range(0, 0));
  viskores::Bounds p2(viskores::Range(0.84, 0.92), viskores::Range(-0.8, -0.1), viskores::Range(0, 0));
  viskores::Bounds p3(viskores::Range(-0.8, -0.72), viskores::Range(+0.1, +0.8), viskores::Range(0, 0));
  viskores::Bounds p4(viskores::Range(-0.8, -0.72), viskores::Range(-0.8, -0.1), viskores::Range(0, 0));

  m_color_bar_pos.push_back(p1);
  m_color_bar_pos.push_back(p2);
  m_color_bar_pos.push_back(p3);
  m_color_bar_pos.push_back(p4);
}

Annotator::~Annotator()
{
  delete m_world_annotator;
}

void
Annotator::RenderScreenAnnotations(const std::vector<std::string> &field_names,
                                    const std::vector<viskores::Range> &ranges,
                                    const std::vector<viskores::cont::ColorTable> &color_tables,
                                    const std::vector<int> &is_discrete)
{
  m_canvas.SetViewToScreenSpace(m_camera, true);
  // currently we only support 4 color bars, so grab the first 4
  int num_bars = std::min(int(field_names.size()),4);
  m_canvas.BeginTextRenderingBatch();
  m_world_annotator->BeginLineRenderingBatch();
  for(int i = 0; i < num_bars; ++i)
  {
    //TODO: What if we have a large range max? i.e. lots of materials
    //Need to extend color bar in proportion somehow??
    if(is_discrete[i])
    {
      int num_tics = abs(ranges[i].Max - ranges[i].Min) + 1;
      this->m_color_bar_annotation.SetRange(ranges[i],num_tics);
    }
    else
      this->m_color_bar_annotation.SetRange(ranges[i], 5);
    this->m_color_bar_annotation.SetFieldName(field_names[i]);
    this->m_color_bar_annotation.SetPosition(m_color_bar_pos[i]);
    this->m_color_bar_annotation.SetColorTable(color_tables[i]);
    this->m_color_bar_annotation.Render(m_camera, *m_world_annotator, m_canvas);
  }
  m_world_annotator->EndLineRenderingBatch();
  m_canvas.EndTextRenderingBatch();
}

void
Annotator::RenderScreenAnnotations(const std::vector<std::string> &field_names,
                                    const std::vector<viskores::Range> &ranges,
                                    const std::vector<viskores::cont::ColorTable> &color_tables,
                                    const std::vector<viskores::Bounds> &color_bar_pos,
                                    const std::vector<int> &is_discrete)
	                         
{
  m_canvas.SetViewToScreenSpace(m_camera, true);
  //a user can put as many color bars as they want anywhere they want
  int num_bars = std::min(int(field_names.size()),int(color_bar_pos.size()));
  m_canvas.BeginTextRenderingBatch();
  m_world_annotator->BeginLineRenderingBatch();
  for(int i = 0; i < num_bars; ++i)
  {
    if(is_discrete[i])
      this->m_color_bar_annotation.SetRange(ranges[i],ranges[i].Max);
    else
      this->m_color_bar_annotation.SetRange(ranges[i], 5);
    this->m_color_bar_annotation.SetFieldName(field_names[i]);
    this->m_color_bar_annotation.SetPosition(color_bar_pos[i]);
    this->m_color_bar_annotation.SetColorTable(color_tables[i]);
    this->m_color_bar_annotation.Render(m_camera, *m_world_annotator, m_canvas);
  }
  m_world_annotator->EndLineRenderingBatch();
  m_canvas.EndTextRenderingBatch();
}

void Annotator::RenderWorldAnnotations(viskores::Vec<float,3> axis_scale)
{
  if(!m_is_3d) return;
  m_canvas.SetViewToWorldSpace(m_camera, false);

  m_canvas.BeginTextRenderingBatch();
  viskores::Float64 xmin = m_bounds.X.Min, xmax = m_bounds.X.Max;
  viskores::Float64 ymin = m_bounds.Y.Min, ymax = m_bounds.Y.Max;
  viskores::Float64 zmin = m_bounds.Z.Min, zmax = m_bounds.Z.Max;
  viskores::Float64 dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
  viskores::Float64 size = viskores::Sqrt(dx * dx + dy * dy + dz * dz);

  //TODO: get forground color
  m_world_annotator->BeginLineRenderingBatch();
  this->m_box_annotation.SetColor(m_canvas.GetForegroundColor());
  this->m_box_annotation.SetExtents(m_bounds);
  this->m_box_annotation.Render(m_camera, *m_world_annotator);
  viskores::Vec<viskores::Float32, 3> lookAt = m_camera.GetLookAt();
  viskores::Vec<viskores::Float32, 3> position = m_camera.GetPosition();
  bool xtest = lookAt[0] > position[0];
  bool ytest = lookAt[1] > position[1];
  bool ztest = lookAt[2] > position[2];
  m_world_annotator->EndLineRenderingBatch();

  const bool outsideedges = true; // if false, do closesttriad
  if (outsideedges)
  {
    xtest = !xtest;
    //ytest = !ytest;
  }

  viskores::Float64 xrel = viskores::Abs(dx) / size;
  viskores::Float64 yrel = viskores::Abs(dy) / size;
  viskores::Float64 zrel = viskores::Abs(dz) / size;
  float major_tick_size = size / 40.f;
  float minor_tick_size = size / 80.f;

  m_world_annotator->BeginLineRenderingBatch();
  this->m_x_axis_annotation.SetAxis(0);
  this->m_x_axis_annotation.SetColor(m_canvas.GetForegroundColor());
  this->m_x_axis_annotation.SetTickInvert(xtest, ytest, ztest);
  this->m_x_axis_annotation.SetWorldPosition(
    xmin, ytest ? ymin : ymax, ztest ? zmin : zmax, xmax, ytest ? ymin : ymax, ztest ? zmin : zmax);
  this->m_x_axis_annotation.SetRange(xmin * axis_scale[0], xmax * axis_scale[0]);
  this->m_x_axis_annotation.SetMajorTickSize(major_tick_size, 0);
  this->m_x_axis_annotation.SetMinorTickSize(minor_tick_size, 0);
  this->m_x_axis_annotation.SetLabelFontOffset(viskores::Float32(size / 15.f));
  this->m_x_axis_annotation.SetMoreOrLessTickAdjustment(-1);
  //this->m_x_axis_annotation.SetMoreOrLessTickAdjustment(xrel < .3 ? -1 : 0);
  this->m_x_axis_annotation.Render(m_camera, *m_world_annotator, m_canvas);

  this->m_y_axis_annotation.SetAxis(1);
  this->m_y_axis_annotation.SetColor(m_canvas.GetForegroundColor());
  this->m_y_axis_annotation.SetTickInvert(xtest, ytest, ztest);
  this->m_y_axis_annotation.SetWorldPosition(
    xtest ? xmin : xmax, ymin, ztest ? zmin : zmax, xtest ? xmin : xmax, ymax, ztest ? zmin : zmax);
  this->m_y_axis_annotation.SetRange(ymin * axis_scale[1], ymax * axis_scale[0]);
  this->m_y_axis_annotation.SetMajorTickSize(major_tick_size, 0);
  this->m_y_axis_annotation.SetMinorTickSize(minor_tick_size, 0);
  this->m_y_axis_annotation.SetLabelFontOffset(viskores::Float32(size / 15.f));
  this->m_y_axis_annotation.SetMoreOrLessTickAdjustment(-1);
  //this->m_y_axis_annotation.SetMoreOrLessTickAdjustment(yrel < .3 ? -1 : 0);
  this->m_y_axis_annotation.Render(m_camera, *m_world_annotator, m_canvas);

  this->m_z_axis_annotation.SetAxis(2);
  this->m_z_axis_annotation.SetColor(m_canvas.GetForegroundColor());
  this->m_z_axis_annotation.SetTickInvert(xtest, ytest, ztest);
  this->m_z_axis_annotation.SetWorldPosition(
    xtest ? xmin : xmax, ytest ? ymin : ymax, zmin, xtest ? xmin : xmax, ytest ? ymin : ymax, zmax);
  this->m_z_axis_annotation.SetRange(zmin * axis_scale[2], zmax * axis_scale[2]);
  this->m_z_axis_annotation.SetMajorTickSize(major_tick_size, 0);
  this->m_z_axis_annotation.SetMinorTickSize(minor_tick_size, 0);
  this->m_z_axis_annotation.SetLabelFontOffset(viskores::Float32(size / 15.f));
  //this->m_z_axis_annotation.SetMoreOrLessTickAdjustment(zrel < .3 ? -1 : 0);
  this->m_z_axis_annotation.SetMoreOrLessTickAdjustment(-1);
  this->m_z_axis_annotation.Render(m_camera, *m_world_annotator, m_canvas);
  m_world_annotator->EndLineRenderingBatch();

  m_canvas.EndTextRenderingBatch();
}

} //namespace vtkh
