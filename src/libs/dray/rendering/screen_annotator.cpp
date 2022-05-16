// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/line_renderer.hpp>
#include <dray/rendering/screen_annotator.hpp>
#include <dray/rendering/screen_text_annotator.hpp>
#include <dray/rendering/color_bar_annotator.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <dray/error.hpp>
#include <dray/math.hpp>

#include <cmath>
#include <sstream>

namespace dray
{

namespace detail
{
void simple_ticks(Framebuffer &fb, Array<Vec<float32,2>> &ticks, const Float length)
{
  const int32 size = ticks.size();
  Vec<float32,2> *hline_ptr = ticks.get_host_ptr();
  Array<int32> offsets;
  offsets.resize(ticks.size());
  int32 *hoffsets_ptr = offsets.get_host_ptr();

  int32 pixel_count = 0;
  for(int32 i = 0; i < size; ++i)
  {
    hoffsets_ptr[i] = pixel_count;
    Vec<float32,2> xline = hline_ptr[i];
    int32 x_start = std::floor(xline[0]);
    int32 x_end = x_start + length;
    pixel_count += x_end - x_start + 1;
  }


  const int32 width = fb.width();
  const int32 height = fb.height();

  DeviceFramebuffer d_framebuffer(fb);
  const Vec<float32,4> color = fb.foreground_color();

  const Vec<float32,2> *line_ptr = ticks.get_device_ptr_const();
  const int32 *offsets_ptr = offsets.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, pixel_count), [=] DRAY_LAMBDA (int32 i)
  {
    /// figure out what pixel/box we belong to
    int line_id = size-1;
    while(i < offsets_ptr[line_id])
    {
      line_id--;
    }

    const int local_id = i - offsets_ptr[line_id];
    Vec<float32,2> start = line_ptr[line_id];
    const int32 y = start[1];
    const int32 x = floor(start[0]) + local_id;
    const int32 pixel_id = y * width + x;

    d_framebuffer.m_colors[pixel_id] = color;

  });
  DRAY_ERROR_CHECK();

}

} // namespace detail

ScreenAnnotator::ScreenAnnotator()
: m_max_color_bars(2)
{
  // ranges are (-1,1)
  AABB<2> p0;
  p0.m_ranges[0].include(0.84f);
  p0.m_ranges[0].include(0.92f);
  p0.m_ranges[1].include(0.1f);
  p0.m_ranges[1].include(0.8f);

  AABB<2> p1;
  p1.m_ranges[0].include(0.84f);
  p1.m_ranges[0].include(0.92f);
  p1.m_ranges[1].include(-0.1f);
  p1.m_ranges[1].include(-0.8f);

  AABB<2> p2;
  p2.m_ranges[0].include(-0.95f);
  p2.m_ranges[0].include(-0.87f);
  p2.m_ranges[1].include(0.1f);
  p2.m_ranges[1].include(0.8f);

  AABB<2> p3;
  p3.m_ranges[0].include(-0.95f);
  p3.m_ranges[0].include(-0.87f);
  p3.m_ranges[1].include(-0.1f);
  p3.m_ranges[1].include(-0.8f);

  m_color_bar_pos.push_back(p2);
  m_color_bar_pos.push_back(p3);
  m_color_bar_pos.push_back(p0);
  m_color_bar_pos.push_back(p1);
}


void
ScreenAnnotator::max_color_bars(const int32 max_bars)
{
  // Technically we can do more that this, but we need to get text
  // alignment working before we enable the other positions
  if(max_bars > 2)
  {
    DRAY_ERROR("Max bars cannot exceed 2");
  }
  if(max_bars < 0)
  {
    DRAY_ERROR("Max bars cannot be less than 0");
  }
  m_max_color_bars = max_bars;
}

void
ScreenAnnotator::draw_color_bars(Framebuffer &fb,
                                 const std::vector<std::string> &field_names,
                                 std::vector<ColorMap> &color_maps)
{
  // TODO: capping at 2
  // we need to justify text to the left of right
  // oriented color bars
  const int32 size = std::min(int32(field_names.size()), m_max_color_bars);

  const int32 height = fb.height();
  const int32 width = fb.width();

  ScreenTextAnnotator text_annot;
  ColorBarAnnotator color_annot;

  Array<Vec<float32,2>> tick_lines;
  tick_lines.resize(5 * size);
  Vec<float32,2> *line_ptr = tick_lines.get_host_ptr();

  for(int32 i = 0; i < size; ++i)
  {
    AABB<2> pos = m_color_bar_pos[i];

    // translate scree to pixels
    float32 x0 = (pos.m_ranges[0].min()+1.f) * 0.5 * width;
    float32 x1 = (pos.m_ranges[0].max()+1.f) * 0.5 * width;
    float32 y0 = (pos.m_ranges[1].min()+1.f) * 0.5 * height;
    float32 y1 = (pos.m_ranges[1].max()+1.f) * 0.5 * height;

    // add the color bar
    Vec<float32,2> color_pos({{x0,y0}});
    Vec<float32,2> color_size({{x1-x0,y1-y0}});
    color_annot.render(fb, color_maps[i].colors(), color_pos, color_size);

    // calculate the color bar ticks
    float32 tick_step = pos.m_ranges[1].length() / 4.f;
    float32 ticks[5];
    ticks[0] = pos.m_ranges[1].min();
    ticks[1] = pos.m_ranges[1].min() + tick_step;
    ticks[2] = pos.m_ranges[1].min() + 2.f*tick_step;
    ticks[3] = pos.m_ranges[1].min() + 3.f*tick_step;
    ticks[4] = pos.m_ranges[1].max();
    float32 tick_pad = 0.01f * width;

    // get the scalar range of the plot
    Range range = color_maps[i].scalar_range();
    bool log_scale = color_maps[i].log_scale();
    float32 rmin = range.min();
    float32 rmax = range.max();

    if(log_scale)
    {
      rmin = log(rmin);
      rmax = log(rmax);
    }

    // genereate the text labels for the range
    float32 length = rmax - rmin;
    float32 step_length = length / 4.f;
    std::stringstream ss;

    float32 text_x = x1 + tick_pad;
    float32 text_size = Float(width) / 52.f;
    float32 y_offset = text_size * 0.5f;

    ss<<rmin;
    float32 t0_y = (ticks[0] + 1.f) * 0.5f * height;
    line_ptr[i*5+0][0] = x1;
    line_ptr[i*5+0][1] = t0_y;
    text_annot.add_text(ss.str(),
                        {{text_x, t0_y - y_offset}},
                        text_size);
    ss.str("");

    ss<<rmin + step_length;
    float32 t1_y = (ticks[1] + 1.f) * 0.5f * height;
    line_ptr[i*5+1][0] = x1;
    line_ptr[i*5+1][1] = t1_y;
    text_annot.add_text(ss.str(),
                        {{text_x, t1_y - y_offset}},
                        text_size);
    ss.str("");

    ss<<rmin + 2.f * step_length;
    float32 t2_y = (ticks[2] + 1.f) * 0.5f * height;
    line_ptr[i*5+2][0] = x1;
    line_ptr[i*5+2][1] = t2_y;
    text_annot.add_text(ss.str(),
                        {{text_x, t2_y - y_offset}},
                        text_size);
    ss.str("");

    ss<<rmin + 3.f * step_length;
    float32 t3_y = (ticks[3] + 1.f) * 0.5f * height;
    line_ptr[i*5+3][0] = x1;
    line_ptr[i*5+3][1] = t3_y;
    text_annot.add_text(ss.str(),
                        {{text_x, t3_y - y_offset}},
                        text_size);
    ss.str("");

    ss<<rmax;
    float32 t4_y = (ticks[4] + 1.f) * 0.5f * height - 1.f;
    line_ptr[i*5+4][0] = x1;
    line_ptr[i*5+4][1] = t4_y;
    text_annot.add_text(ss.str(),
                        {{text_x, t4_y - y_offset}},
                        text_size);
    ss.str("");

    // add the variable name
    text_annot.add_text("Var: "+field_names[i],
                        {{x0, y1 + tick_pad}},
                        text_size);

    // log or linear scale
    std::string scale = "Scale: ";
    scale += log_scale ? "log" : "linear";
    text_annot.add_text(scale,
                        {{x0, y0 - text_size*1.5f}},
                        text_size);



  }

  detail::simple_ticks(fb,tick_lines, 4.f);
  text_annot.render(fb);

}

void ScreenAnnotator::draw_triad(
  Framebuffer &fb,
  Vec<float32, 2> pos, // screen space coords where we want the triad to be centered
  float32 distance,
  Camera &camera)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  int width = fb.width();
  int height = fb.height();

  Camera triad_camera;
  triad_camera.set_width (width);
  triad_camera.set_height (height);

  // set origin and basis vectors
  Vec<float32, 3> o = {{0,0,0}};
  Vec<float32, 3> i = {{1,0,0}};
  Vec<float32, 3> j = {{0,1,0}};
  Vec<float32, 3> k = {{0,0,1}};

  Vec<float32, 3> look = (camera.get_look_at() - camera.get_pos()).normalized();
  Vec<float32, 3> up = camera.get_up().normalized();

  triad_camera.set_pos(o - distance * look);
  triad_camera.set_up(up);
  triad_camera.set_look_at(o);

  Matrix<float32, 4, 4> V = triad_camera.view_matrix();

  o = transform_point(V, o);
  i = transform_point(V, i);
  j = transform_point(V, j);
  k = transform_point(V, k);

  int num_lines = 3;
  Array<Vec<float32,3>> starts;
  Array<Vec<float32,3>> ends;
  starts.resize(num_lines);
  ends.resize(num_lines);
  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();
  starts_ptr[0] = o;
  ends_ptr[0] = i;
  starts_ptr[1] = o;
  ends_ptr[1] = j;
  starts_ptr[2] = o;
  ends_ptr[2] = k;

  AABB<3> triad_aabb;
  triad_aabb.m_ranges[0].set_range(-1.f, 1.f);
  triad_aabb.m_ranges[1].set_range(-1.f, 1.f);
  triad_aabb.m_ranges[2].set_range(-1.f, 1.f);

  Matrix<float32, 4, 4> P = triad_camera.projection_matrix(triad_aabb);

  // for the triad labels
  Array<Vec<float32,2>> xyz_text_pos;
  xyz_text_pos.resize(3);
  Vec<float32,2> *xyz_text_pos_ptr = xyz_text_pos.get_host_ptr();

  // we want to transform our points now, and then offset them correctly before drawing
  // we also wish to extract information for annotations later
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> start;
    start[0] = starts_ptr[i][0];
    start[1] = starts_ptr[i][1];
    start[2] = starts_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = ends_ptr[i][0];
    end[1] = ends_ptr[i][1];
    end[2] = ends_ptr[i][2];
    end[3] = 1;

    // for the annotations
    Vec<float32, 4> text_pos = P * ((end - start) * 1.1f + start);
    text_pos = text_pos / text_pos[3];

    // transform via projection matrix
    start = P * start;
    end = P * end;

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    // discover screen space coords
    float32 x1,x2,y1,y2,z1,z2;
    x1 = start[0];  y1 = start[1];  z1 = start[2];
    x2 = end[0];    y2 = end[1];    z2 = end[2];

    // transform starts and ends of each line to the position specified
    float32 xmov = pos[0] - x1;
    float32 ymov = pos[1] - y1;
    x1 = pos[0];
    y1 = pos[1];
    x2 += xmov;
    y2 += ymov;

    // offset text as well - and put into pixel space
    int text_x = (((text_pos[0] + xmov) + 1.f) / 2.f) * width;
    int text_y = (((text_pos[1] + ymov) + 1.f) / 2.f) * height;
    xyz_text_pos_ptr[i] = {{(float32) text_x, (float32) text_y}};

    // save new locations of lines in SS
    starts_ptr[i][0] = x1;
    starts_ptr[i][1] = y1;
    starts_ptr[i][2] = z1;
    ends_ptr[i][0] = x2;
    ends_ptr[i][1] = y2;
    ends_ptr[i][2] = z2;
  });

  Matrix<float32, 4, 4> transform;
  transform.identity();

  // call line renderer
  LineRenderer lines;
  bool should_depth_be_zero = true;
  lines.render(fb, transform, starts, ends, should_depth_be_zero);

  ScreenTextAnnotator annot;
  annot.clear();

  int32 text_size = 20;
  annot.add_text("X", xyz_text_pos_ptr[0], text_size);
  annot.add_text("Y", xyz_text_pos_ptr[1], text_size);
  annot.add_text("Z", xyz_text_pos_ptr[2], text_size);

  annot.render(fb);
}

} // namespace dray
