#ifndef APCOMP_IMAGE_COMPOSITOR_HPP
#define APCOMP_IMAGE_COMPOSITOR_HPP

#include <apcomp/apcomp_config.h>

#include <apcomp/image.hpp>
#include <algorithm>

#include <apcomp/apcomp_exports.h>
#include <apcomp/error.hpp>

namespace apcomp
{

struct CompositeOrderSort
{
  inline bool operator()(const Image &lhs, const Image &rhs) const
  {
    return lhs.m_composite_order < rhs.m_composite_order;
  }
};

class APCOMP_API ImageCompositor
{
public:
 void Blend(apcomp::Image &front, apcomp::Image &back)
 {

   bool valid = true;
   valid &= front.m_bounds.m_min_x == back.m_bounds.m_min_x;
   valid &= front.m_bounds.m_min_y == back.m_bounds.m_min_y;
   valid &= front.m_bounds.m_max_x == back.m_bounds.m_max_x;
   valid &= front.m_bounds.m_max_y == back.m_bounds.m_max_y;

   if(!valid)
   {
     throw Error("image bounds do not match");
   }

   const int size = static_cast<int>(front.m_pixels.size() / 4);

#ifdef APCOMP_OPENMP_ENABLED
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    const int offset = i * 4;
    unsigned int alpha = front.m_pixels[offset + 3];
    const unsigned int opacity = 255 - alpha;

    front.m_pixels[offset + 0] +=
      static_cast<unsigned char>(opacity * back.m_pixels[offset + 0] / 255);
    front.m_pixels[offset + 1] +=
      static_cast<unsigned char>(opacity * back.m_pixels[offset + 1] / 255);
    front.m_pixels[offset + 2] +=
      static_cast<unsigned char>(opacity * back.m_pixels[offset + 2] / 255);
    front.m_pixels[offset + 3] +=
      static_cast<unsigned char>(opacity * back.m_pixels[offset + 3] / 255);

    float d1 = std::min(front.m_depths[i], 1.001f);
    float d2 = std::min(back.m_depths[i], 1.001f);
    float depth = std::min(d1,d2);
    front.m_depths[i] = depth;
  }
}

void ZBufferComposite(apcomp::Image &front, const apcomp::Image &image)
{
  bool valid = true;
  valid &= front.m_depths.size() == front.m_pixels.size() / 4;
  valid &= front.m_bounds.m_min_x == image.m_bounds.m_min_x;
  valid &= front.m_bounds.m_min_y == image.m_bounds.m_min_y;
  valid &= front.m_bounds.m_max_x == image.m_bounds.m_max_x;
  valid &= front.m_bounds.m_max_y == image.m_bounds.m_max_y;
  if(!valid)
  {
    throw Error("image bounds do not match");
  }

  const int size = static_cast<int>(front.m_depths.size());
  bool gl_depth = front.m_gl_depth;
  if(gl_depth)
  {
    // Only composite values the GL depths range (0,1)
#ifdef APCOMP_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      const float depth = image.m_depths[i];
      if(depth > 1.f  || front.m_depths[i] < depth)
      {
        continue;
      }
      const int offset = i * 4;
      front.m_depths[i] = depth;
      front.m_pixels[offset + 0] = image.m_pixels[offset + 0];
      front.m_pixels[offset + 1] = image.m_pixels[offset + 1];
      front.m_pixels[offset + 2] = image.m_pixels[offset + 2];
      front.m_pixels[offset + 3] = image.m_pixels[offset + 3];
    }
  }
  else
  {
#ifdef APCOMP_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      const float depth = image.m_depths[i];
      if(front.m_depths[i] < depth)
      {
        continue;
      }
      const int offset = i * 4;
      front.m_depths[i] = depth;
      front.m_pixels[offset + 0] = image.m_pixels[offset + 0];
      front.m_pixels[offset + 1] = image.m_pixels[offset + 1];
      front.m_pixels[offset + 2] = image.m_pixels[offset + 2];
      front.m_pixels[offset + 3] = image.m_pixels[offset + 3];
    }
  }
}

void OrderedComposite(std::vector<apcomp::Image> &images)
{
  const int total_images = images.size();
  std::sort(images.begin(), images.end(), CompositeOrderSort());
  for(int i = 1; i < total_images; ++i)
  {
    Blend(images[0], images[i]);
  }
}

void ZBufferComposite(std::vector<apcomp::Image> &images)
{
  const int total_images = images.size();
  for(int i = 1; i < total_images; ++i)
  {
    ZBufferComposite(images[0], images[i]);
  }
}

struct Pixel
{
  unsigned char m_color[4];
  float         m_depth;
  int           m_pixel_id;  // local (sub-image) pixels id

  bool operator < (const Pixel &other) const
  {
    if(m_pixel_id != other.m_pixel_id)
    {
      return m_pixel_id < other.m_pixel_id;
    }
    else
    {
      return m_depth < other.m_depth;
    }
  }
};

void CombineImages(const std::vector<apcomp::Image> &images, std::vector<Pixel> &pixels)
{

  const int num_images = static_cast<int>(images.size());
  for(int i = 0; i < num_images; ++i)
  {
    //
    //  Extract the partial composites into a contiguous array
    //

    const int image_size = images[i].GetNumberOfPixels();
    const int offset = i * image_size;
#ifdef APCOMP_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int j = 0; j < image_size; ++j)
    {
      const int image_offset = j * 4;
      pixels[offset + j].m_color[0] = images[i].m_pixels[image_offset + 0];
      pixels[offset + j].m_color[1] = images[i].m_pixels[image_offset + 1];
      pixels[offset + j].m_color[2] = images[i].m_pixels[image_offset + 2];
      pixels[offset + j].m_color[3] = images[i].m_pixels[image_offset + 3];
      pixels[offset + j].m_depth = images[i].m_depths[j];
      pixels[offset + j].m_pixel_id = j;
    } // for pixels
  } // for images

}

void ZBufferBlend(std::vector<apcomp::Image> &images)
{
  const int image_pixels = images[0].GetNumberOfPixels();
  const int num_images = static_cast<int>(images.size());
  std::vector<Pixel> pixels;
  CombineImages(images, pixels);
#ifdef APCOMP_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(int i = 0; i < image_pixels; ++i)
  {
    const int begin = image_pixels * i;
    const int end = image_pixels * i - 1;
    std::sort(pixels.begin() + begin, pixels.begin() + end);
  }

#ifdef APCOMP_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(int i = 0; i < image_pixels; ++i)
  {
    const int index = i * num_images;
    Pixel pixel = pixels[index];
    for(int j = 1; j < num_images; ++j)
    {
      if(pixel.m_color[3] == 255 || pixel.m_depth > 1.f)
      {
        break;
      }
      unsigned int alpha = pixel.m_color[3];
      const unsigned int opacity = 255 - alpha;
      pixel.m_color[0]
        += static_cast<unsigned char>(opacity * pixels[index + j].m_color[0] / 255);
      pixel.m_color[1]
        += static_cast<unsigned char>(opacity * pixels[index + j].m_color[1] / 255);
      pixel.m_color[2]
        += static_cast<unsigned char>(opacity * pixels[index + j].m_color[2] / 255);
      pixel.m_color[3]
        += static_cast<unsigned char>(opacity * pixels[index + j].m_color[3] / 255);
      pixel.m_depth = pixels[index + j].m_depth;
    } // for each image
    images[0].m_pixels[i * 4 + 0] = pixel.m_color[0];
    images[0].m_pixels[i * 4 + 1] = pixel.m_color[1];
    images[0].m_pixels[i * 4 + 2] = pixel.m_color[2];
    images[0].m_pixels[i * 4 + 3] = pixel.m_color[3];
    images[0].m_depths[i] = pixel.m_depth;
  } // for each pixel

}


};

} // namespace apcomp
#endif
