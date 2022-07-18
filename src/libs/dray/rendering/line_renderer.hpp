// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LINE_RENDERER_HPP
#define DRAY_LINE_RENDERER_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/matrix.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/transform_3d.hpp>

namespace dray
{

class LineRenderer
{
public:
  // there are 3 rendering functions which accomplish virtually the same thing
  // but with differing approaches that parallelize differently.
  // The outputs may differ slightly but not enough for any to be considered "more correct"
  // than others.

// 1. follows the Bresenham alg, one raja loop that parallelizes over lines
  // for each line it runs a while loop over the pixels and draws them
  void render(
  	Framebuffer &fb, 
  	Matrix<float32, 4, 4> transform,
  	Array<Vec<float32,3>> starts, 
  	Array<Vec<float32,3>> ends,
  	bool should_depth_be_zero = false);
  
// 2. inspired by Bresenham.
  // 3 raja loops
  	// 1) parallelizes over lines and calculates the number of pixels each line must render
  	// 2) parallelizes over lines and stores all the drawing info for each pixel to buffers
  	// 3) parallelizes over pixels and renders information from the buffers
  // note: step 3 might not be necessary, instead of writing info to buffers and then writing 
  // it again to the framebuffer, it could be faster to simply write to the framebuffer directly
  // in step 2.
  void render2(
  	Framebuffer &fb, 
  	Matrix<float32, 4, 4> transform,
  	Array<Vec<float32,3>> starts, 
  	Array<Vec<float32,3>> ends,
  	bool should_depth_be_zero = false);
// 3. very different from Bresenham's line alg
  // 2 raja loops
  	// 1) parallelizes over lines and calculates the number of pixels each line must render, as well as
  	//    the direction of the line and the starting point in screen space
  	// 2) parallelizes over pixels, and, for each pixel, figures out which line it is a part of, and then 
  	//    determines how far along the line it ought to be based on the number of total pixels in the line,
  	//    and renders using this information
  void render3(
  	Framebuffer &fb, 
  	Matrix<float32, 4, 4> transform,
  	Array<Vec<float32,3>> starts, 
  	Array<Vec<float32,3>> ends,
  	bool should_depth_be_zero = false);
  
  // all three rendering methods use linear interpolation to determine the depth of a given pixel
};

DRAY_EXEC
void crop_line_to_bounds(
	Vec<int32, 2> &p1, 
	Vec<int32, 2> &p2, 
	int32 width, 
	int32 height)
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
      // CANT CALL THIS ON DEVICE,       
      // TODO SOME SORT OF DEVICE ABORT?
      /*
      fprintf(stderr, "line cropping has determined that the current line "
                      "never crosses the screen, yet at least one of the "
                      "endpoints is simultaneously on the screen, which is a contradiction.\n");
      exit(1);
      */
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

} // namespace dray

#endif
