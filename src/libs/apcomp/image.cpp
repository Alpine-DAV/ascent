//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "image.hpp"
#include <apcomp/error.hpp>
#include <apcomp/utils/png_encoder.hpp>
#include <limits>
#include <assert.h>

namespace apcomp
{

Image::Image()
  : m_orig_rank(-1),
    m_has_transparency(false),
    m_composite_order(-1),
    m_gl_depth(true)
{
}

Image::Image(const Bounds &bounds)
  : m_orig_bounds(bounds),
    m_bounds(bounds),
    m_orig_rank(-1),
    m_has_transparency(false),
    m_composite_order(-1),
    m_gl_depth(true)

{
    const int dx  = bounds.m_max_x - bounds.m_min_x + 1;
    const int dy  = bounds.m_max_y - bounds.m_min_y + 1;
    m_pixels.resize(dx * dy * 4);
    m_depths.resize(dx * dy);
}

void
Image::InitOriginal(const Image &other)
{
  m_orig_bounds = other.m_orig_bounds;
  m_bounds = other.m_orig_bounds;

  const int dx  = m_bounds.m_max_x - m_bounds.m_min_x + 1;
  const int dy  = m_bounds.m_max_y - m_bounds.m_min_y + 1;
  m_pixels.resize(dx * dy * 4);
  m_depths.resize(dx * dy);

  m_orig_rank = -1;
  m_has_transparency = false;
  m_composite_order = -1;
  m_gl_depth = other.m_gl_depth;
}

void
Image::Init(const float *color_buffer,
            const float *depth_buffer,
            int width,
            int height,
            bool gl_depth,
            int composite_order)
{
  m_composite_order = composite_order;
  m_bounds.m_min_x = 1;
  m_bounds.m_min_y = 1;
  m_bounds.m_max_x = width;
  m_bounds.m_max_y = height;
  m_orig_bounds = m_bounds;
  m_gl_depth = gl_depth;
  const int size = width * height;
  m_pixels.resize(size * 4);
  m_depths.resize(size);

#ifdef APCOMP_USE_OPENMP
      #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    const int offset = i * 4;
    m_pixels[offset + 0] = static_cast<unsigned char>(color_buffer[offset + 0] * 255.f);
    m_pixels[offset + 1] = static_cast<unsigned char>(color_buffer[offset + 1] * 255.f);
    m_pixels[offset + 2] = static_cast<unsigned char>(color_buffer[offset + 2] * 255.f);
    m_pixels[offset + 3] = static_cast<unsigned char>(color_buffer[offset + 3] * 255.f);
    float depth = depth_buffer[i];
    //make sure we can do a single comparison on depth
    if(gl_depth)
    {
      depth = depth < 0 ? 2.f : depth;
    }
    m_depths[i] =  depth;
  }
}

void
Image::Init(const unsigned char *color_buffer,
            const float *depth_buffer,
            int width,
            int height,
            bool gl_depth,
            int composite_order)
{
  m_composite_order = composite_order;
  m_bounds.m_min_x = 1;
  m_bounds.m_min_y = 1;
  m_bounds.m_max_x = width;
  m_bounds.m_max_y = height;
  m_orig_bounds = m_bounds;
  m_gl_depth = gl_depth;

  const int size = width * height;
  m_pixels.resize(size * 4);
  m_depths.resize(size);

  std::copy(color_buffer,
            color_buffer + size * 4,
            &m_pixels[0]);

#ifdef apcomp_USE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    float depth = depth_buffer[i];
    //make sure we can do a single comparison on depth
    if(gl_depth)
    {
      depth = depth < 0 ? 2.f : depth;
    }
    m_depths[i] = depth;
  } // for
}

void Image::SetHasTransparency(bool has_transparency)
{
  m_has_transparency = has_transparency;
}

bool Image::HasTransparency()
{
  return m_has_transparency;
}

int Image::GetNumberOfPixels() const
{
  return static_cast<int>(m_pixels.size() / 4);
}

void
Image::CompositeBackground(const float color[4])
{

  const int size = static_cast<int>(m_pixels.size() / 4);
  unsigned char bg_color[4];
  for(int i = 0; i < 4; ++i)
  {
    bg_color[i] = static_cast<unsigned char>(color[i] * 255.f);
  }

#ifdef APCOMP_USE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    const int offset = i * 4;
    unsigned int alpha = static_cast<unsigned int>(m_pixels[offset + 3]);
    const float opacity = (255 - alpha);
    m_pixels[offset + 0] += static_cast<unsigned char>(opacity * bg_color[0] / 255);
    m_pixels[offset + 1] += static_cast<unsigned char>(opacity * bg_color[1] / 255);
    m_pixels[offset + 2] += static_cast<unsigned char>(opacity * bg_color[2] / 255);
    m_pixels[offset + 3] += static_cast<unsigned char>(opacity * bg_color[3] / 255);
  }
}

void
Image::SubsetFrom(const Image &image,
                  const Bounds &sub_region)
{
  m_orig_bounds = image.m_orig_bounds;
  m_bounds = sub_region;
  m_orig_rank = image.m_orig_rank;
  m_composite_order = image.m_composite_order;
  m_gl_depth = image.m_gl_depth;

  assert(sub_region.m_min_x >= image.m_bounds.m_min_x);
  assert(sub_region.m_min_y >= image.m_bounds.m_min_y);
  assert(sub_region.m_max_x <= image.m_bounds.m_max_x);
  assert(sub_region.m_max_y <= image.m_bounds.m_max_y);

  const int s_dx  = m_bounds.m_max_x - m_bounds.m_min_x + 1;
  const int s_dy  = m_bounds.m_max_y - m_bounds.m_min_y + 1;

  const int dx  = image.m_bounds.m_max_x - image.m_bounds.m_min_x + 1;
  //const int dy  = image.m_bounds.m_max_y - image.m_bounds.m_min_y + 1;

  const int start_x = m_bounds.m_min_x - image.m_bounds.m_min_x;
  const int start_y = m_bounds.m_min_y - image.m_bounds.m_min_y;
  const int end_y = start_y + s_dy;

  m_pixels.resize(s_dx * s_dy * 4);
  m_depths.resize(s_dx * s_dy);

#ifdef APCOMP_USE_OPENMP
  #pragma omp parallel for
#endif
  for(int y = start_y; y < end_y; ++y)
  {
    const int copy_to = (y - start_y) * s_dx;
    const int copy_from = y * dx + start_x;

    std::copy(&image.m_pixels[copy_from * 4],
              &image.m_pixels[copy_from * 4] + s_dx * 4,
              &m_pixels[copy_to * 4]);
    std::copy(&image.m_depths[copy_from],
              &image.m_depths[copy_from] + s_dx,
              &m_depths[copy_to]);
  }
}

void
Image::SubsetTo(Image &image) const
{
  image.m_composite_order = m_composite_order;
  image.m_gl_depth = m_gl_depth;
  assert(m_bounds.m_min_x >= image.m_bounds.m_min_x);
  assert(m_bounds.m_min_y >= image.m_bounds.m_min_y);
  assert(m_bounds.m_max_x <= image.m_bounds.m_max_x);
  assert(m_bounds.m_max_y <= image.m_bounds.m_max_y);

  const int s_dx  = m_bounds.m_max_x - m_bounds.m_min_x + 1;
  const int s_dy  = m_bounds.m_max_y - m_bounds.m_min_y + 1;

  const int dx  = image.m_bounds.m_max_x - image.m_bounds.m_min_x + 1;
  //const int dy  = image.m_bounds.m_max_y - image.m_bounds.m_min_y + 1;

  const int start_x = m_bounds.m_min_x - image.m_bounds.m_min_x;
  const int start_y = m_bounds.m_min_y - image.m_bounds.m_min_y;

#ifdef APCOMP_USE_OPENMP
  #pragma omp parallel for
#endif
  for(int y = 0; y < s_dy; ++y)
  {
    const int copy_to = (y + start_y) * dx + start_x;
    const int copy_from = y * s_dx;

    std::copy(&m_pixels[copy_from * 4],
              &m_pixels[copy_from * 4] + s_dx * 4,
              &image.m_pixels[copy_to * 4]);

    std::copy(&m_depths[copy_from],
              &m_depths[copy_from] + s_dx,
              &image.m_depths[copy_to]);
  }
}

void
Image::Swap(Image &other)
{
  Bounds orig   = m_orig_bounds;
  Bounds bounds = m_bounds;

  m_orig_bounds = other.m_orig_bounds;
  m_bounds      = other.m_bounds;
  m_gl_depth = other.m_gl_depth;

  other.m_orig_bounds = orig;
  other.m_bounds      = bounds;

  m_pixels.swap(other.m_pixels);
  m_depths.swap(other.m_depths);
}

void
Image::Color(int color)
{
  unsigned char c[4];
  c[3] = 255;

  c[0] = 0;
  c[1] = 0;
  c[2] = 0;
  int index = color % 3;
  c[index] = 255 - color * 1;;
  const int size = static_cast<int>(m_pixels.size());
  for(int i = 0; i < size; ++i)
  {
    float d = m_depths[i / 4];
    if(d >0 && d < 1)
    {
     m_pixels[i] = c[i%4];
    }
    else
    {
     m_pixels[i] = 155;
    }
  }

}

std::string
Image::ToString() const
{
  std::stringstream ss;
  ss<<"Total size pixels "<< (int) m_pixels.size() / 4;
  ss<<" tile dims: {"<<m_bounds.m_min_x<<","<< m_bounds.m_min_y<<"} - ";
  ss<<"{"<<m_bounds.m_max_x<<","<<m_bounds.m_max_y<<"}\n";;
  return ss.str();
}

void Image::Clear()
{
  Bounds empty;
  m_orig_bounds = empty;
  m_bounds = empty;
  m_pixels.clear();
  m_depths.clear();
}

void Image::Save(std::string name)
{
  int width = m_bounds.m_max_x - m_bounds.m_min_x + 1;
  int height = m_bounds.m_max_y - m_bounds.m_min_y + 1;

  if(width * height <= 0)
  {
    throw Error("Image: cannot save empty image");
  }

  PNGEncoder encoder;
  encoder.Encode(&m_pixels[0], width, height);
  encoder.Save(name +  ".png");
}

void Image::SaveDepth(std::string name)
{
  int width = m_bounds.m_max_x - m_bounds.m_min_x + 1;
  int height = m_bounds.m_max_y - m_bounds.m_min_y + 1;

  if(width * height <= 0)
  {
    throw Error("Image: cannot save empty image");
  }

  float inf = std::numeric_limits<float>::infinity();
  float min_v = inf;
  float max_v = -inf;
  for(int i = 0; i < width * height;++i)
  {
    float d = m_depths[i];
    if(d != inf)
    {
      min_v = std::min(min_v, d);
      max_v = std::max(max_v, d);
    }
  }

  const float len = max_v - min_v;
  std::vector<float> ndepths(width*height*4);

  for(int i = 0; i < width * height;++i)
  {
    const float depth = m_depths[i];
    float value = 0.f;
    const int offset = i * 4;
    if(depth != inf)
    {
      value = (depth - min_v) / len;
    }
    ndepths[offset + 0] = value;
    ndepths[offset + 1] = value;
    ndepths[offset + 2] = value;
    ndepths[offset + 3] = 1.f;
  }

  PNGEncoder encoder;
  encoder.Encode(&ndepths[0], width, height);
  encoder.Save(name +  ".png");
}

} // namespace apcomp
