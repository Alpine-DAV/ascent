// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// I believe this is this codes third migration, at least
// VisIt (maybe) -> EAVL -> vtkm -> here

#include <dray/rendering/world_annotator.hpp>
#include <dray/rendering/line_renderer.hpp>
#include <dray/rendering/font.hpp>
#include <dray/rendering/world_text_annotator.hpp>

#include <cstdio>
#include <cstdarg>
#include <string>
#include <iostream>

#include <conduit/conduit_fmt/conduit_fmt.h>

namespace dray
{

namespace detail
{

float32 axis_scale(const AABB<3> &bounds, std::string &scale_string)
{
  scale_string = "";
  float32 scale = 1.f;

  Float axis_mag = 0.f;
  // find the largest exponent for all three axes
  for(int32 axis = 0; axis < 3; ++axis)
  {
    axis_mag = max(abs(bounds.m_ranges[axis].min()), axis_mag);
    axis_mag = max(abs(bounds.m_ranges[axis].max()), axis_mag);
  }
  if(axis_mag != 0.f)
  {
    // clamp to nearest power of 3
    Float exponent = log10(axis_mag);
    int32 iexp = round(exponent);
    int32 modiexp = iexp % 3;
    int32 modsign = (0 < modiexp) - (modiexp < 0);
    if(modsign == 1) iexp -= modiexp;
    else if(modsign == -1) iexp += modiexp;
    if(iexp != 0)
    {
      scale = pow(10.f, float32(iexp));
      scale_string = "(10^"+std::to_string(iexp) + ")";
    }
  }
  return scale;
}

std::string format_string(const float32 num)
{
  return conduit_fmt::format("{:.1f}", num);
}

void
find_anchor(const Vec<float32,3> &cam_pos,
            const AABB<3> &bounds,
            Vec<float32,3> &anchor,
            Vec<float32,3> ends[3]) // ends in x,y,z
{
  float32 minx, miny, minz, maxx, maxy, maxz;
  minx = bounds.m_ranges[0].min();
  miny = bounds.m_ranges[1].min();
  minz = bounds.m_ranges[2].min();
  maxx = bounds.m_ranges[0].max();
  maxy = bounds.m_ranges[1].max();
  maxz = bounds.m_ranges[2].max();

  Vec<float32, 3> bounds_points[8];
  bounds_points[0] = {{minx, miny, minz}};
  bounds_points[1] = {{minx, miny, maxz}};
  bounds_points[2] = {{minx, maxy, minz}};
  bounds_points[3] = {{minx, maxy, maxz}};
  bounds_points[4] = {{maxx, miny, minz}};
  bounds_points[5] = {{maxx, miny, maxz}};
  bounds_points[6] = {{maxx, maxy, minz}};
  bounds_points[7] = {{maxx, maxy, maxz}};

  float32 min_dist = infinity32();
  int32 index = -1;

  for(int32 i = 0; i < 8; i++)
  {
    float32 dist = (bounds_points[i] - cam_pos).magnitude();

    if(dist < min_dist)
    {
      index = i;
      min_dist = dist;
    }
  }

  anchor = bounds_points[index];
  ends[0] = anchor;
  ends[0][0] = ends[0][0] == maxx ? minx : maxx;
  ends[1] = anchor;
  ends[1][1] = ends[1][1] == maxy ? miny : maxy;
  ends[2] = anchor;
  ends[2][2] = ends[2][2] == maxz ? minz : maxz;
}


inline float32 ffix(float32 value)
{
  int32 ivalue = (int32)(value);
  float32 v = (value - ivalue);
  if (v > 0.9999f)
  {
    ivalue++;
  }
  return static_cast<float32>(ivalue);
}

void calculate_ticks(const Range &range,
                     bool minor,
                     std::vector<float32> &positions,
                     std::vector<float32> &proportions,
                     int modify_tick_quantity)
{
 positions.clear();
 proportions.clear();

  if (range.is_empty())
  {
    return;
  }

  float32 length = range.length();

  if(length == 0.f)
  {
    return;
  }

  // Find the integral points.
  float32 pow10 = log10(length);

  // Build in numerical tolerance
  float32 eps = 10.0e-10f;
  pow10 += eps;

  // ffix moves you in the wrong direction if pow10 is negative.
  if (pow10 < 0.f)
  {
    pow10 = pow10 - 1.f;
  }

  float32 fxt = pow(10.f, ffix(pow10));

  // Find the number of integral points in the interval.
  int numTicks = int(ffix(length / fxt) + 1);

  // We should get about major 10 ticks on a length that's near
  // the power of 10.  (e.g. length=1000).  If the length is small
  // enough we have less than 5 ticks (e.g. length=400), then
  // divide the step by 2, or if it's about 2 ticks (e.g. length=150)
  // or less, then divide the step by 5.  That gets us back to
  // about 10 major ticks.
  //
  // But we might want more or less.  To adjust this up by
  // approximately a factor of 2, instead of the default
  // 1/2/5 dividers, use 2/5/10, and to adjust it down by
  // about a factor of two, use .5/1/2 as the dividers.
  // (We constrain to 1s, 2s, and 5s, for the obvious reason
  // that only those values are factors of 10.....)
  float32 divs[5] = { 0.5, 1, 2, 5, 10 };
  int divindex = (numTicks >= 5) ? 1 : (numTicks >= 3 ? 2 : 3);
  divindex += modify_tick_quantity;

  float32 div = divs[divindex];

  // If there aren't enough major tick points in this decade, use the next
  // decade.
  float32 majorStep = fxt / div;
  float32 minorStep = (fxt / div) / 10.;

  // When we get too close, we lose the tickmarks. Run some special case code.
  if (numTicks <= 1)
  {
    if (minor)
    {
      // no minor ticks
      return;
    }
    else
    {
      positions.resize(3);
      proportions.resize(3);
      positions[0] = range.min();
      positions[1] = range.center();
      positions[2] = range.max();
      proportions[0] = 0.0;
      proportions[1] = 0.5;
      proportions[2] = 1.0;
      return;
    }
  }

  // Figure out the first major and minor tick locations, relative to the
  // start of the axis.
  float32 majorStart, minorStart;
  if (range.min() < 0.)
  {
    majorStart = majorStep * (ffix(range.min() * (1.f / majorStep)));
    minorStart = minorStep * (ffix(range.min() * (1.f / minorStep)));
  }
  else
  {
    majorStart = majorStep * (ffix(range.min() * (1.f / majorStep) + .999f));
    minorStart = minorStep * (ffix(range.min() * (1.f / minorStep) + .999f));
  }

  // Create all of the minor ticks
  const int max_count_cutoff = 1000;
  numTicks = 0;
  float32 location = minor ? minorStart : majorStart;
  float32 step = minor ? minorStep : majorStep;
  while (location <= range.max() && numTicks < max_count_cutoff)
  {
    positions.push_back(location);
    proportions.push_back((location - range.min()) / length);
    numTicks++;
    location += step;
  }
}

} // namespace detail

WorldAnnotator::WorldAnnotator(AABB<3> bounds)
  : m_bounds(bounds)
{
}

void
WorldAnnotator::add_axes(const Camera &camera)
{
  float32 xmin = m_bounds.m_ranges[0].min();
  float32 ymin = m_bounds.m_ranges[1].min();
  float32 zmin = m_bounds.m_ranges[2].min();
  float32 xmax = m_bounds.m_ranges[0].max();
  float32 ymax = m_bounds.m_ranges[1].max();
  float32 zmax = m_bounds.m_ranges[2].max();
  float32 dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
  float32 size = sqrt(dx * dx + dy * dy + dz * dz);

  Vec<float32, 3> center = ((Vec<float32, 3>) {{xmax - xmin, ymax - ymin, zmax - zmin}}) / 2.f +
    ((Vec<float32, 3>) {{xmin, ymin, zmin}});

  const Vec<float32,3> look_at = camera.get_look_at();
  const Vec<float32,3> position = camera.get_pos();
  bool xtest = look_at[0] > position[0];
  bool ytest = look_at[1] > position[1];
  bool ztest = look_at[2] > position[2];

  // Tick settings
  Vec<float32,3> tick_invert{{1.f,1.f,1.f}};
  // swap these if you want the ticks to not wrap around
  // the bounding box
  tick_invert[0] = xtest ? -1.f : 1.f;
  tick_invert[1] = ytest ? -1.f : 1.f;
  tick_invert[2] = ztest ? -1.f : 1.f;

  float32 major_tick_size = size / 40.f;
  float32 minor_tick_size = size / 80.f;
  // offset of 0 means the tick is inside the frame
  // offset of 1 means the tick is outside the frame
  // offset of 0.5 means the tick is centered on the frame
  float32 major_tick_offset = 0.f;
  float32 minor_tick_offset = 0.f;

  int more_or_less_ticks = 0;

  Vec<float32,3> anchor;
  Vec<float32,3> ends[3];
  detail::find_anchor(position, m_bounds, anchor, ends);

  // there are two lines for each tick that wrap around the box
  // in orthogonal directions to the axis
  constexpr Vec<float32,3> tick_dirs[3][2]
    = {
        { {{0.f,1.f,0.f}}, {{0.f, 0.f, 1.f}} },
        { {{1.f,0.f,0.f}}, {{0.f, 0.f, 1.f}} },
        { {{1.f,0.f,0.f}}, {{0.f, 1.f, 0.f}} }
      };

  std::vector<float32> positions;
  std::vector<float32> proportions;

  float32 large_text_size = m_bounds.max_length() * 0.05;
  float32 small_text_size = m_bounds.max_length() * 0.025;

  std::string axis_scale_string;
  float32 axis_scale = detail::axis_scale(m_bounds, axis_scale_string);

  for(int32 axis = 0; axis < 3; ++axis)
  {

    // major ticks
    bool minor = false;
    detail::calculate_ticks(m_bounds.m_ranges[axis],
                            minor,
                            positions,
                            proportions,
                            more_or_less_ticks);

    Vec<float32,3> start = anchor, end = ends[axis];
    Vec<float32,3> dir = end - start;

    Vec<float32,3> mid = (start + end) / 2.f;
    const float32 scale_axis_labels = 1.08f;
    Vec<float32,3> axis_label_pos = center + (mid - center) * scale_axis_labels;
    m_annot_positions.push_back(axis_label_pos);
    if (axis == 0)
    {
      m_annotations.push_back("X-Axis" + axis_scale_string);
      m_annot_sizes.push_back(large_text_size);
    }
    if (axis == 1)
    {
      m_annotations.push_back("Y-Axis" + axis_scale_string);
      m_annot_sizes.push_back(large_text_size);
    }
    if (axis == 2)
    {
      m_annotations.push_back("Z-Axis" + axis_scale_string);
      m_annot_sizes.push_back(large_text_size);
    }

    const int32 major_size = positions.size();

    Vec<float32,3> tick1_size = major_tick_size * tick_dirs[axis][0];
    Vec<float32,3> tick2_size = major_tick_size * tick_dirs[axis][1];
    for(int32 i = 0; i < 3; ++i)
    {
      tick1_size[i] = tick1_size[i] * tick_invert[i];
      tick2_size[i] = tick2_size[i] * tick_invert[i];
    }

    for(int32 i = 0; i < major_size; ++i)
    {
      Vec<float32,3> tick_pos = start + proportions[i] * dir;

      Vec<float32,3> start_pos = tick_pos - tick1_size * major_tick_offset;
      Vec<float32,3> end_pos = tick_pos - tick1_size * (1.f - major_tick_offset);

      m_starts.push_back(start_pos);
      m_ends.push_back(end_pos);

      // push back WS text position and the desired text there
      const float32 scale_major_tick_labels = 1.03f;
      Vec<float32,3> text_pos = center + (start_pos - center) * scale_major_tick_labels;

      float32 scaled  = start_pos[axis] / axis_scale;
      //float rounded_value = floor(100.f * scaled) / 100.f;
      const std::string str = detail::format_string(scaled);

      m_annotations.push_back(str);
      m_annot_sizes.push_back(small_text_size);
      m_annot_positions.push_back(text_pos);

      // start ought to remain the same...
      start_pos = tick_pos - tick2_size * major_tick_offset;

      m_starts.push_back(start_pos);
      m_ends.push_back(tick_pos - tick2_size * (1.f - major_tick_offset));
    }

    minor = true;
    // minor ticks
    detail::calculate_ticks(m_bounds.m_ranges[axis],
                            minor,
                            positions,
                            proportions,
                            more_or_less_ticks);

    const int32 minor_size = positions.size();

    tick1_size = minor_tick_size * tick_dirs[axis][0];
    tick2_size = minor_tick_size * tick_dirs[axis][1];

    for(int32 i = 0; i < 3; ++i)
    {
      tick1_size[i] = tick1_size[i] * tick_invert[i];
      tick2_size[i] = tick2_size[i] * tick_invert[i];
    }

    for(int32 i = 0; i < minor_size; ++i)
    {
      Vec<float32,3> tick_pos = start + proportions[i] * dir;
      m_starts.push_back(tick_pos - tick1_size * minor_tick_offset);
      m_ends.push_back(tick_pos - tick1_size * (1.f - minor_tick_offset));
    }
  }
}

void
WorldAnnotator::add_bounding_box()
{
  float32 minx, miny, minz, maxx, maxy, maxz;
  minx = m_bounds.m_ranges[0].min();
  miny = m_bounds.m_ranges[1].min();
  minz = m_bounds.m_ranges[2].min();
  maxx = m_bounds.m_ranges[0].max();
  maxy = m_bounds.m_ranges[1].max();
  maxz = m_bounds.m_ranges[2].max();

  // TODO: check 2d;
  m_starts.push_back({{minx,miny,minz}});
  m_ends.push_back({{minx,miny,maxz}});

  m_starts.push_back({{minx,maxy,minz}});
  m_ends.push_back({{minx,maxy,maxz}});

  m_starts.push_back({{maxx,miny,minz}});
  m_ends.push_back({{maxx,miny,maxz}});

  m_starts.push_back({{maxx,maxy,minz}});
  m_ends.push_back({{maxx,maxy,maxz}});

  // x
  m_starts.push_back({{minx,miny,minz}});
  m_ends.push_back({{maxx,miny,minz}});

  m_starts.push_back({{minx,miny,maxz}});
  m_ends.push_back({{maxx,miny,maxz}});

  m_starts.push_back({{minx,maxy,minz}});
  m_ends.push_back({{maxx,maxy,minz}});

  m_starts.push_back({{minx,maxy,maxz}});
  m_ends.push_back({{maxx,maxy,maxz}});

  //// y
  m_starts.push_back({{minx,miny,minz}});
  m_ends.push_back({{minx,maxy,minz}});

  m_starts.push_back({{minx,miny,maxz}});
  m_ends.push_back({{minx,maxy,maxz}});

  m_starts.push_back({{maxx,miny,minz}});
  m_ends.push_back({{maxx,maxy,minz}});

  m_starts.push_back({{maxx,miny,maxz}});
  m_ends.push_back({{maxx,maxy,maxz}});

}

void
WorldAnnotator::render(Framebuffer &fb, Array<Ray> &rays, const Camera &camera)
{
  add_bounding_box();
  add_axes(camera);

  if(m_starts.size() != m_ends.size())
  {
    std::cout<<"Internal error: starts and ends do not match\n";
  }
  Array<Vec<float32,3>> line_starts;
  line_starts.resize(m_starts.size());
  Array<Vec<float32,3>> line_ends;
  line_ends.resize(m_starts.size());
  Vec<float32,3> *s_ptr = line_starts.get_host_ptr();
  Vec<float32,3> *e_ptr = line_ends.get_host_ptr();
  for(int32 i = 0; i < line_starts.size(); ++i)
  {
    s_ptr[i] = m_starts[i];
    e_ptr[i] = m_ends[i];
  }

  Matrix<float32, 4, 4> view = camera.view_matrix();
  Matrix<float32, 4, 4> proj = camera.projection_matrix(m_bounds);
  Matrix<float32, 4, 4> transform = proj * view;

  LineRenderer lines;
  lines.render(fb, transform, line_starts, line_ends);

  constexpr Vec<float32,2> justification_bl = {{0.f,0.f}};
  if(true)
  {
    WorldTextAnnotator annot;

    for (int i = 0; i < m_annotations.size(); i ++)
    {
      annot.add_text(m_annotations[i],
                     m_annot_positions[i],
                     justification_bl,
                     m_annot_sizes[i]);
    }

    annot.render(camera, rays, fb);
  }
}

} // namespace dray
