// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/line_renderer.hpp>
#include <dray/math.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/utils/timer.hpp>
#include <dray/rendering/rasterbuffer.hpp>

namespace dray
{

// returns true if the line cross through the view
bool crop_matt(Vec<float32, 3> &p1, Vec<float32, 3> &p2)
{
  // this is clipping line segments in 3d space after its be transformed
  // into normalized screen coordinates [-1, 1]
  // My strat is to turn this into a ray aabb intersection problem
  Vec<float32,3> dir = p2-p1;
  const float32 dist = dir.magnitude();
  dir.normalize();
  Vec<float32,3> inv_dir;
  inv_dir[0] = rcp_safe(dir[0]);
  inv_dir[1] = rcp_safe(dir[1]);
  inv_dir[2] = rcp_safe(dir[2]);
  float32 xmin0 = (-1.f - p1[0]) * inv_dir[0];
  float32 ymin0 = (-1.f - p1[1]) * inv_dir[1];
  float32 zmin0 = (-1.f - p1[2]) * inv_dir[2];
  float32 xmax0 = (1.f - p1[0]) * inv_dir[0];
  float32 ymax0 = (1.f - p1[1]) * inv_dir[1];
  float32 zmax0 = (1.f - p1[2]) * inv_dir[2];

  constexpr float32 min_dist = 0.f;

  float32 min0 = fmaxf(
    fmaxf(fmaxf(fminf(ymin0, ymax0), fminf(xmin0, xmax0)), fminf(zmin0, zmax0)),
    min_dist);
  float32 max0 = fminf(
    fminf(fminf(fmaxf(ymin0, ymax0), fmaxf(xmin0, xmax0)), fmaxf(zmin0, zmax0)),
    dist);
  bool render = (max0 >= min0);
  if(render)
  {
    Vec<float32,3> new_p1 = p1 + min0 * dir;
    Vec<float32,3> new_p2 = p1 + max0 * dir;
    p1 = new_p1;
    p2 = new_p2;
  }
  return true;
}

void crop_line_to_bounds(Vec<int32, 2> &p1, Vec<int32, 2> &p2, int32 width, int32 height)
{
  // booleans to record if p1 and p2 are within bounds or not
  bool p1_ok, p2_ok;
  p1_ok = p2_ok = false;

  float32 x1, y1, x2, y2;
  x1 = p1[0];
  y1 = p1[1];
  x2 = p2[0];
  y2 = p2[1];

  // check that out points are within bounds
  if (x1 > -1 && x1 < width && y1 > -1 && y1 < height)
  {
    p1_ok = true;
  }
  if (x2 > -1 && x2 < width && y2 > -1 && y2 < height)
  {
    p2_ok = true;
  }

  // if both points are within bounds then there is nothing further to do
  if (p1_ok && p2_ok)
  {
    return;
  }

  // calculate the equation of the line
  float32 m = (y2 - y1) / (x2 - x1);
  float32 b = y1 - m * x1;

  // const values used to index into buffers
  const int32 top = 0;
  const int32 bottom = 1;
  const int32 left = 2;
  const int32 right = 3;

  // a buffer to store true or false values for whether or not intersections are within bounds
  int32 intersections_within_bounds[4];
  // a buffer to store coordinates for the 4 intersections of the line with the lines making up the edges of the screen
  Vec<float32, 2> intersect_coords[4];

  // calculate the intersection points for each of the 4 sides
  intersect_coords[top] = {{((height - 1) - b) / m, (float32) (height - 1)}};
  intersect_coords[bottom] = {{(-1 * b) / m, 0}};
  intersect_coords[left] = {{0, b}};
  intersect_coords[right] = {{(float32) (width - 1), m * (width - 1) + b}};
  // determine which of the intersection points are within bounds
  bool none_within_bounds = true;
  for (int32 i = 0; i < 4; i ++)
  {
    // if the coordinates are within bounds...
    if (intersect_coords[i][0] > -1 && intersect_coords[i][0] < width &&
        intersect_coords[i][1] > -1 && intersect_coords[i][1] < height)
    {
      intersections_within_bounds[i] = true;
      none_within_bounds = false;
    }
    else
    {
      intersections_within_bounds[i] = false;
    }
  }

  // if our line never passes across screen space
  if (none_within_bounds)
  {
    if (p1_ok || p2_ok)
    {
      fprintf(stderr, "line cropping has determined that the current line "
                      "never crosses the screen, yet at least one of the "
                      "endpoints is simultaneously on the screen, which is a contradiction.\n");
      exit(1);
    }
    // then we can return the following so that while loops in render will complete quickly
    p1[0] = -1;
    p1[1] = -1;
    p2[0] = -1;
    p2[1] = -1;
    return;
  }

  // tie breaking - make sure that a maximum of two sides are marked as having valid intersections
  // so if top and bottom are both valid, then left and right cannot be, and vice versa
  // this saves us from issues down the road without a loss in correctness
  if (intersections_within_bounds[top] && intersections_within_bounds[bottom])
  {
    intersections_within_bounds[left] = false;
    intersections_within_bounds[right] = false;
  }
  if (intersections_within_bounds[right] && intersections_within_bounds[left])
  {
    intersections_within_bounds[top] = false;
    intersections_within_bounds[bottom] = false;
  }

  // next we set up a data structure to house information about our intersections
  // only two intersections max will actually be in view of the camera
  // so for each of the two intersections, we record distance^2 to p1,
  // and the x and y vals of the intersection point, hence the six slots
  float32 intersection_info[6];

  int32 index = 0;
  // we iterate over each of the four sides
  for (int32 i = 0; i < 4; i ++)
  {
    if (intersections_within_bounds[i])
    {
      float32 y1_minus_newy = y1 - intersect_coords[i][1];
      float32 x1_minus_newx = x1 - intersect_coords[i][0];
      // the first three spots are for one intersection
      // calculate the distance squared to p1
      intersection_info[index + 0] = (int32) (y1_minus_newy * y1_minus_newy + x1_minus_newx * x1_minus_newx);
      // then x and y coordinates of the intersection
      intersection_info[index + 1] = intersect_coords[i][0];
      intersection_info[index + 2] = intersect_coords[i][1];
      // then we increment by 3 to get to the next three spots, for the next intersection
      index += 3;
    }
  }

  // with this information we can assign new values to p1 and p2 if needed
  float32 distance1 = intersection_info[0];
  float32 distance2 = intersection_info[3];
  index = distance1 < distance2 ? 0 : 1;
  // if p1 was out of bounds
  if (!p1_ok)
  {
    // then we replace it with the intersection that makes sense
    p1[0] = intersection_info[index * 3 + 1];
    p1[1] = intersection_info[index * 3 + 2];
  }

  // we flip the index so it points at the other half of our array
  index = !index;
  // if p2 was out of bounds
  if (!p2_ok)
  {
    // then we replace it with the intersection that makes sense
    p2[0] = intersection_info[index * 3 + 1];
    p2[1] = intersection_info[index * 3 + 2];
  }
}

void LineRenderer::render(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends,
  bool should_depth_be_zero)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int32 num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  int32 width = fb.width();
  int32 height = fb.height();

  // draw pixels using bresenham's alg
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};

    Vec<float32,4> start;
    start[0] = start_ptr[i][0];
    start[1] = start_ptr[i][1];
    start[2] = start_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = end_ptr[i][0];
    end[1] = end_ptr[i][1];
    end[2] = end_ptr[i][2];
    end[3] = 1;

    start = transform * start;
    end = transform * end;

    float32 start_depth = start[3];
    float32 end_depth = end[3];

    // divide by the w component
    start = start / start[3];
    end = end / end[3];


    Vec<float32,3> s_p1 = {{start[0], start[1], start[2]}};
    Vec<float32,3> s_p2 = {{end[0], end[1], end[2]}};

    bool cool = crop_matt(s_p1,s_p2);
    // return if no part of the line is visible
    if(!cool) return;

    // transform to pixel space
    int32 x1,x2,y1,y2;
    x1 = ((s_p1[0] + 1.f) / 2.f) * width;
    y1 = ((s_p1[1] + 1.f) / 2.f) * height;
    x2 = ((s_p2[0] + 1.f) / 2.f) * width;
    y2 = ((s_p2[1] + 1.f) / 2.f) * height;

    // crop the line
    Vec<int32, 2> p1, p2;
    p1[0] = x1;    p1[1] = y1;
    p2[0] = x2;    p2[1] = y2;

    x1 = p1[0];    y1 = p1[1];
    x2 = p2[0];    y2 = p2[1];

    // to keep track of which pixel we are on... used for lin interp of depths
    int32 myindex = 0;

    // Bresenham calculations, more info on wikipedia
    int32 dx = abs(x2 - x1);
    int32 sx = x1 < x2 ? 1 : -1;
    int32 dy = -1 * abs(y2 - y1);
    int32 sy = y1 < y2 ? 1 : -1;
    int32 err = dx + dy;

    int32 abs_dx = abs(dx);
    int32 abs_dy = abs(dy);

    // we want to determine the total number of pixels needed to draw this line
    // so we can use is to determine the percentage of the way done we are
    // which is used for lin interp of depths
    float32 pixels_to_draw = 0.f;

    if (abs_dy > abs_dx)
    {
      // then slope is greater than 1
      // so we need one pixel for every y value
      pixels_to_draw = abs_dy + 1.f;
    }
    else
    {
      // then slope is less than 1
      // then we need one pixel for every x value
      pixels_to_draw = abs_dx + 1.f;
    }

    while (true)
    {
      float32 depth = 0.f;
      if (!should_depth_be_zero)
      {
        // calculate a reasonable depth with lin interp
        float32 progress = ((float) myindex) / pixels_to_draw;
        depth = (1.f - progress) * start_depth + progress * end_depth;
      }

      d_raster.write_pixel(x1, y1, color, depth);

      myindex += 1;
      if (x1 == x2 && y1 == y2)
      {
        break;
      }
      int32 e2 = 2 * err;
      if (e2 >= dy)
      {
        err += dy;
        x1 += sx;
      }
      if (e2 <= dx)
      {
        err += dx;
        y1 += sy;
      }
    }
  });

  // write this back to the original framebuffer
  raster.finalize();
}

void LineRenderer::render2(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends,
  bool should_depth_be_zero)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int32 num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  // a buffer to store the number of pixels per line
  Array<int32> pixels_per_line;
  pixels_per_line.resize(num_lines);
  int32 *pixels_per_line_ptr = pixels_per_line.get_device_ptr();

  int32 width = fb.width();
  int32 height = fb.height();

  // buffers to store the start and end depths of each line
  Array<float32> start_depths;
  start_depths.resize(num_lines);
  float32 *start_depths_ptr = start_depths.get_device_ptr();
  Array<float32> end_depths;
  end_depths.resize(num_lines);
  float32 *end_depths_ptr = end_depths.get_device_ptr();

  // a buffer to store SS coords of the starts and ends of each line
  Array<Vec<int32,4>> SS_starts_and_ends;
  SS_starts_and_ends.resize(num_lines);
  Vec<int32,4> *SS_starts_and_ends_ptr = SS_starts_and_ends.get_device_ptr();

  // count the number of pixels in each line
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> start;
    start[0] = start_ptr[i][0];
    start[1] = start_ptr[i][1];
    start[2] = start_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = end_ptr[i][0];
    end[1] = end_ptr[i][1];
    end[2] = end_ptr[i][2];
    end[3] = 1;

    start = transform * start;
    end = transform * end;

    start_depths_ptr[i] = start[3];
    end_depths_ptr[i] = end[3];

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    int32 x1,x2,y1,y2;
    x1 = ((start[0] + 1.f) / 2.f) * width;
    y1 = ((start[1] + 1.f) / 2.f) * height;
    x2 = ((end[0] + 1.f) / 2.f) * width;
    y2 = ((end[1] + 1.f) / 2.f) * height;

    // lines get cropped
    Vec<int32, 2> p1, p2;
    p1[0] = x1;    p1[1] = y1;
    p2[0] = x2;    p2[1] = y2;
    crop_line_to_bounds(p1, p2, width, height);
    x1 = p1[0];    y1 = p1[1];
    x2 = p2[0];    y2 = p2[1];

    // we want to save these cropped and transformed values so we don't need to calculate them again
    SS_starts_and_ends_ptr[i] = {{x1,y1,x2,y2}};

    int32 dx = abs(x2 - x1);
    int32 dy = -1 * abs(y2 - y1);

    int32 abs_dx = abs(dx);
    int32 abs_dy = abs(dy);

    if (abs_dy > abs_dx)
    {
      // then slope is greater than 1
      // so we need one pixel for every y value
      pixels_per_line_ptr[i] = abs_dy + 1;
    }
    else
    {
      // then slope is less than 1
      // then we need one pixel for every x value
      pixels_per_line_ptr[i] = abs_dx + 1;
    }
  });

  // should this prefix sum be parallelized???
  // calculate offsets
  Array<int32> offsets;
  offsets.resize(num_lines);
  int32 *offsets_ptr = offsets.get_device_ptr();
  offsets_ptr[0] = 0;
  for (int32 i = 1; i < num_lines; i ++)
  {
    offsets_ptr[i] = offsets_ptr[i - 1] + pixels_per_line_ptr[i - 1];
  }

  int32 num_pixels = offsets_ptr[num_lines - 1] + pixels_per_line_ptr[num_lines - 1];

  // new containers for the next step's data
  Array<int32> x_values;
  Array<int32> y_values;
  Array<Vec<float32, 4>> colors;
  Array<float32> depths;
  x_values.resize(num_pixels);
  y_values.resize(num_pixels);
  colors.resize(num_pixels);
  depths.resize(num_pixels);
  int32 *x_values_ptr = x_values.get_device_ptr();
  int32 *y_values_ptr = y_values.get_device_ptr();
  Vec<float32, 4> *colors_ptr = colors.get_device_ptr();
  float32 *depths_ptr = depths.get_device_ptr();

  // save the colors, depths, and coordinates of the pixels to draw
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};

    int32 x1,y1,x2,y2;
    x1 = SS_starts_and_ends_ptr[i][0];
    y1 = SS_starts_and_ends_ptr[i][1];
    x2 = SS_starts_and_ends_ptr[i][2];
    y2 = SS_starts_and_ends_ptr[i][3];

    int32 myindex = 0;

    int32 dx = abs(x2 - x1);
    int32 sx = x1 < x2 ? 1 : -1;
    int32 dy = -1 * abs(y2 - y1);
    int32 sy = y1 < y2 ? 1 : -1;
    int32 err = dx + dy;

    while (true)
    {
      float32 depth = 0.f;
      if (!should_depth_be_zero)
      {
        float32 progress = ((float32) myindex) / ((float32) pixels_per_line_ptr[i]);
        depth = (1.f - progress) * start_depths_ptr[i] + progress * end_depths_ptr[i];
      }

      // save everything in buffers to write later
      x_values_ptr[myindex + offsets_ptr[i]] = x1;
      y_values_ptr[myindex + offsets_ptr[i]] = y1;
      colors_ptr[myindex + offsets_ptr[i]] = color;
      depths_ptr[myindex + offsets_ptr[i]] = depth;

      myindex += 1;
      if (x1 == x2 && y1 == y2)
      {
        break;
      }
      int32 e2 = 2 * err;
      if (e2 >= dy)
      {
        err += dy;
        x1 += sx;
      }
      if (e2 <= dx)
      {
        err += dx;
        y1 += sy;
      }
    }
  });

  // finally, render pixels using saved information
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels), [=] DRAY_LAMBDA (int32 i)
  {
    d_raster.write_pixel(x_values_ptr[i], y_values_ptr[i], colors_ptr[i], depths_ptr[i]);
  });

  // write this back to the original framebuffer
  raster.finalize();
}

void LineRenderer::render3(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends,
  bool should_depth_be_zero)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int32 num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  int32 width = fb.width();
  int32 height = fb.height();

  // a new container for unit vectors in the direction (end - start) in SS
  Array<Vec<int32,2>> directions_array;
  directions_array.resize(num_lines);
  Vec<int32,2> *directions = directions_array.get_device_ptr();

  // a buffer to store the number of pixels needed to draw each line
  Array<int32> pixels_per_line;
  pixels_per_line.resize(num_lines);
  int32 *pixels_per_line_ptr = pixels_per_line.get_device_ptr();

  // buffers to store start and end depths for each line
  Array<float32> start_depths;
  start_depths.resize(num_lines);
  float32 *start_depths_ptr = start_depths.get_device_ptr();
  Array<float32> end_depths;
  end_depths.resize(num_lines);
  float32 *end_depths_ptr = end_depths.get_device_ptr();

  // a buffer to store SS coords of starting points
  Array<Vec<int32,2>> SS_starts;
  SS_starts.resize(num_lines);
  Vec<int32,2> *SS_starts_ptr = SS_starts.get_device_ptr();

  // save the colors and coordinates of the pixels to draw
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> start;
    start[0] = start_ptr[i][0];
    start[1] = start_ptr[i][1];
    start[2] = start_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = end_ptr[i][0];
    end[1] = end_ptr[i][1];
    end[2] = end_ptr[i][2];
    end[3] = 1;

    start = transform * start;
    end = transform * end;

    start_depths_ptr[i] = start[3];
    end_depths_ptr[i] = end[3];

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    // transform to pixel space
    int32 x1,x2,y1,y2;
    x1 = ((start[0] + 1.f) / 2.f) * width;
    y1 = ((start[1] + 1.f) / 2.f) * height;
    x2 = ((end[0] + 1.f) / 2.f) * width;
    y2 = ((end[1] + 1.f) / 2.f) * height;

    // crop line to SS
    Vec<int32, 2> p1, p2;
    p1[0] = x1;    p1[1] = y1;
    p2[0] = x2;    p2[1] = y2;
    crop_line_to_bounds(p1, p2, width, height);
    x1 = p1[0];    y1 = p1[1];
    x2 = p2[0];    y2 = p2[1];

    int32 dx = x2 - x1;
    int32 dy = y2 - y1;

    int32 abs_dx = abs(dx);
    int32 abs_dy = abs(dy);

    if (abs_dy > abs_dx)
    {
      // then slope is greater than 1
      // so we need one pixel for every y value
      pixels_per_line_ptr[i] = abs_dy + 1;
    }
    else
    {
      // then slope is less than 1
      // then we need one pixel for every x value
      pixels_per_line_ptr[i] = abs_dx + 1;
    }

    // we want to save these cropped and transformed values so we don't need to calculate them again
    SS_starts_ptr[i] = {{x1,y1}};

    directions[i] = {{dx, dy}};
  });

  // should this prefix sum be parallelized???
  // calculate offsets
  Array<int32> offsets;
  offsets.resize(num_lines + 1);
  int32 *offsets_ptr = offsets.get_device_ptr();
  offsets_ptr[0] = 0;
  for (int32 i = 1; i < num_lines + 1; i ++)
  {
    offsets_ptr[i] = offsets_ptr[i - 1] + pixels_per_line_ptr[i - 1];
  }

  int32 num_pixels = offsets_ptr[num_lines];

  // parallelize over pixels
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels), [=] DRAY_LAMBDA (int32 i)
  {
    // determine which line the pixel belongs to
    int32 which_line;
    // determine the index of the pixel with respect to its line
    int32 index;
    // the percentage of the way along the line this pixel is
    float32 percentage;

    // loop to determine which line
    for (int32 j = 0; j < num_lines; j ++)
    {
      if (offsets_ptr[j] <= i && offsets_ptr[j + 1] > i)
      {
        which_line = j;
        break;
      }
    }
    // calculate the index
    int32 offset = offsets_ptr[which_line];
    index = offset == 0 ? i : i % offset;

    // and percentage of the way done with the line
    percentage = ((float32) index) / ((float32) pixels_per_line_ptr[which_line]);

    // get our starting pos and direction
    float32 x1,y1;
    x1 = SS_starts_ptr[which_line][0];
    y1 = SS_starts_ptr[which_line][1];

    float32 dx, dy;
    dx = directions[which_line][0];
    dy = directions[which_line][1];

    // and use them to determine the coords of a pixel
    int32 x,y;
    x = x1 + dx * percentage;
    y = y1 + dy * percentage;

    // any color, as long as it's black
    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};

    float32 depth = 0.f;
    if (!should_depth_be_zero)
    {
      // calc depth with lin interp using progress along the line
      float32 progress = ((float32) index) / ((float32) pixels_per_line_ptr[which_line]);
      depth = (1.f - progress) * start_depths_ptr[which_line] + progress * end_depths_ptr[which_line];
    }

    d_raster.write_pixel(x, y, color, depth);
  });

  // write this back to the original framebuffer
  raster.finalize();
}

} // namespace dray

