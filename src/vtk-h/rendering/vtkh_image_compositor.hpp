#ifndef VTKH_DIY_IMAGE_COMPOSITOR_HPP
#define VTKH_DIY_IMAGE_COMPOSITOR_HPP

#include <rendering/vtkh_image.hpp>

namespace vtkh
{

class ImageCompositor
{
public:
  void Blend(vtkh::Image &front, vtkh::Image &back)
  {

    assert(front.m_bounds.X.Min == back.m_bounds.X.Min); 
    assert(front.m_bounds.Y.Min == back.m_bounds.Y.Min); 
    assert(front.m_bounds.X.Max == back.m_bounds.X.Max); 
    assert(front.m_bounds.Y.Max == back.m_bounds.Y.Max); 

    const int size = static_cast<int>(front.m_pixels.size() / 4); 
  
#ifdef VTKH_USE_OPENMP
    #pragma omp parallel for 
#endif
    for(int i = 0; i < size; ++i)
    {
      const int offset = i * 4;
      unsigned int alpha = front.m_pixels[offset + 3];// / 255.f;
      const unsigned int opacity = 255 - alpha;//(1.f - alpha) * alpha2;
      front.m_pixels[offset + 0] += 
        static_cast<unsigned char>(opacity * back.m_pixels[offset + 0] / 255); 
      front.m_pixels[offset + 1] += 
        static_cast<unsigned char>(opacity * back.m_pixels[offset + 1] / 255); 
      front.m_pixels[offset + 2] += 
        static_cast<unsigned char>(opacity * back.m_pixels[offset + 2] / 255); 
      front.m_pixels[offset + 3] += 
        static_cast<unsigned char>(opacity * back.m_pixels[offset + 3] / 255); 
    }
  }

void ZBufferComposite(vtkh::Image &front, const vtkh::Image &image)
{
  assert(front.m_depths.size() == front.m_pixels.size() / 4);
  assert(front.m_bounds.X.Min == image.m_bounds.X.Min); 
  assert(front.m_bounds.Y.Min == image.m_bounds.Y.Min); 
  assert(front.m_bounds.X.Max == image.m_bounds.X.Max); 
  assert(front.m_bounds.Y.Max == image.m_bounds.Y.Max); 

  const int size = static_cast<int>(front.m_depths.size()); 

#ifdef vtkh_USE_OPENMP
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

void OrderedComposite(std::vector<vtkh::Image> &images)
{
  const int total_images = images.size();
  std::sort(images.begin(), images.end(), CompositeOrderSort());
  for(int i = 1; i < total_images; ++i)
  {
    Blend(images[0], images[i]);
  }
}

void ZBufferComposite(std::vector<vtkh::Image> &images)
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

void CombineImages(const std::vector<vtkh::Image> &images, std::vector<Pixel> &pixels)
{

  const int num_images = static_cast<int>(images.size());
  int total_pixels = images[0].GetNumberOfPixels() * num_images;

  for(int i = 0; i < num_images; ++i)
  {
    //
    //  Extract the partial composites into a contiguous array
    //

    const int image_size = images[i].GetNumberOfPixels(); 
    const int offset = i * image_size;
    #pragma omp parallel for
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

void ZBufferBlend(std::vector<vtkh::Image> &images)
{
  const int image_pixels = images[0].GetNumberOfPixels();
  const int num_images = static_cast<int>(images.size());
  std::vector<Pixel> pixels;
  CombineImages(images, pixels);
  #pragma omp parallel for
  for(int i = 0; i < image_pixels; ++i)
  {
    const int begin = image_pixels * i; 
    const int end = image_pixels * i - 1; 
    std::sort(pixels.begin() + begin, pixels.begin() + end);  
  }
  
  // check to see if that worked
  int pixel_id_0 = pixels[0].m_pixel_id;
  for(int i = 1; i < num_images; ++i)
  {
    assert(pixel_id_0 == pixels[i].m_pixel_id);
  }
 

  #pragma omp parallel for
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

} // namespace vtkh
#endif
