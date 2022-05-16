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

void crop_line_to_bounds(
	Vec<int32, 2> &p1, 
	Vec<int32, 2> &p2, 
	int32 width, 
	int32 height);

} // namespace dray

#endif
