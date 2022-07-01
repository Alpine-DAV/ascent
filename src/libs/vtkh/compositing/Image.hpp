#ifndef VTKH_DIY_IMAGE_HPP
#define VTKH_DIY_IMAGE_HPP

#include <sstream>
#include <vector>
#include <vtkm/Bounds.h>

#include <vtkh/vtkh_exports.h>

namespace vtkh
{

struct VTKH_API Image
{
    // The image bounds are indicated by a grid starting at
    // 1-width and 1-height. Actual width would be calculated
    // m_bounds.X.Max - m_bounds.X.Min + 1
    // 1024 - 1 + 1 = 1024
    vtkm::Bounds                 m_orig_bounds;
    vtkm::Bounds                 m_bounds;
    std::vector<unsigned char>   m_pixels;
    std::vector<float>           m_depths;
    int                          m_orig_rank;
    bool                         m_has_transparency;
    int                          m_composite_order;

    Image()
      : m_orig_rank(-1),
        m_has_transparency(false),
        m_composite_order(-1)
    {}


    Image(const vtkm::Bounds &bounds)
      : m_orig_bounds(bounds),
        m_bounds(bounds),
        m_orig_rank(-1),
        m_has_transparency(false),
        m_composite_order(-1)

    {
        const int dx  = bounds.X.Max - bounds.X.Min + 1;
        const int dy  = bounds.Y.Max - bounds.Y.Min + 1;
        m_pixels.resize(dx * dy * 4);
        m_depths.resize(dx * dy);
    }

    // init this image based on the original bounds
    // of the other image
    void InitOriginal(const Image &other)
    {
      m_orig_bounds = other.m_orig_bounds;
      m_bounds = other.m_orig_bounds;

      const int dx  = m_bounds.X.Max - m_bounds.X.Min + 1;
      const int dy  = m_bounds.Y.Max - m_bounds.Y.Min + 1;
      m_pixels.resize(dx * dy * 4);
      m_depths.resize(dx * dy);

      m_orig_rank = -1;
      m_has_transparency = false;
      m_composite_order = -1;
    }

    int GetNumberOfPixels() const
    {
      return static_cast<int>(m_pixels.size() / 4);
    }

    void SetHasTransparency(bool has_transparency)
    {
      m_has_transparency = has_transparency;
    }

    bool HasTransparency()
    {
      return m_has_transparency;
    }

    void Init(const float *color_buffer,
              const float *depth_buffer,
              int width,
              int height,
              int composite_order = -1)
    {
      m_composite_order = composite_order;
      m_bounds.X.Min = 1;
      m_bounds.Y.Min = 1;
      m_bounds.X.Max = width;
      m_bounds.Y.Max = height;
      m_orig_bounds = m_bounds;
      const int size = width * height;
      m_pixels.resize(size * 4);
      m_depths.resize(size);

#ifdef VTKH_OPENMP_ENABLED
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
        depth = depth < 0 ? 2.f : depth;
        m_depths[i] =  depth;
      }
    }

    void Init(const unsigned char *color_buffer,
              const float *depth_buffer,
              int width,
              int height,
              int composite_order = -1)
    {
      m_composite_order = composite_order;
      m_bounds.X.Min = 1;
      m_bounds.Y.Min = 1;
      m_bounds.X.Max = width;
      m_bounds.Y.Max = height;
      m_orig_bounds = m_bounds;

      const int size = width * height;
      m_pixels.resize(size * 4);
      m_depths.resize(size);

      std::copy(color_buffer,
                color_buffer + size * 4,
                &m_pixels[0]);

#ifdef vtkh_USE_OPENMP
      #pragma omp parallel for
#endif
      for(int i = 0; i < size; ++i)
      {
        float depth = depth_buffer[i];
        //make sure we can do a single comparison on depth
        depth = depth < 0 ? 2.f : depth;
        m_depths[i] =  depth;
      } // for
    }


    void CompositeBackground(const float *color)
    {

      const int size = static_cast<int>(m_pixels.size() / 4);
      unsigned char bg_color[4];
      for(int i = 0; i < 4; ++i)
      {
        bg_color[i] = static_cast<unsigned char>(color[i] * 255.f);
      }

#ifdef VTKH_OPENMP_ENABLED
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
    //
    // Fill this image with a sub-region of another image
    //
    void SubsetFrom(const Image &image,
                    const vtkm::Bounds &sub_region)
    {
      m_orig_bounds = image.m_orig_bounds;
      m_bounds = sub_region;
      m_orig_rank = image.m_orig_rank;
      m_composite_order = image.m_composite_order;

      assert(sub_region.X.Min >= image.m_bounds.X.Min);
      assert(sub_region.Y.Min >= image.m_bounds.Y.Min);
      assert(sub_region.X.Max <= image.m_bounds.X.Max);
      assert(sub_region.Y.Max <= image.m_bounds.Y.Max);

      const int s_dx  = m_bounds.X.Max - m_bounds.X.Min + 1;
      const int s_dy  = m_bounds.Y.Max - m_bounds.Y.Min + 1;

      const int dx  = image.m_bounds.X.Max - image.m_bounds.X.Min + 1;
      //const int dy  = image.m_bounds.Y.Max - image.m_bounds.Y.Min + 1;

      const int start_x = m_bounds.X.Min - image.m_bounds.X.Min;
      const int start_y = m_bounds.Y.Min - image.m_bounds.Y.Min;
      const int end_y = start_y + s_dy;

      m_pixels.resize(s_dx * s_dy * 4);
      m_depths.resize(s_dx * s_dy);



#ifdef VTKH_OPENMP_ENABLED
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

    void Color(int color)
    {
      unsigned char c[4];
      c[3] = 255;

      c[0] = 0;
      c[1] = 0;
      c[2] = 0;
      int index = color % 3;
      c[index] = 255 - color * 11;;
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
    //
    // Fills the passed in image with the contents of this image
    //
    void SubsetTo(Image &image) const
    {
      image.m_composite_order = m_composite_order;
      assert(m_bounds.X.Min >= image.m_bounds.X.Min);
      assert(m_bounds.Y.Min >= image.m_bounds.Y.Min);
      assert(m_bounds.X.Max <= image.m_bounds.X.Max);
      assert(m_bounds.Y.Max <= image.m_bounds.Y.Max);

      const int s_dx  = m_bounds.X.Max - m_bounds.X.Min + 1;
      const int s_dy  = m_bounds.Y.Max - m_bounds.Y.Min + 1;

      const int dx  = image.m_bounds.X.Max - image.m_bounds.X.Min + 1;
      //const int dy  = image.m_bounds.Y.Max - image.m_bounds.Y.Min + 1;

      const int start_x = m_bounds.X.Min - image.m_bounds.X.Min;
      const int start_y = m_bounds.Y.Min - image.m_bounds.Y.Min;

#ifdef VTKH_OPENMP_ENABLED
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

    void Swap(Image &other)
    {
      vtkm::Bounds orig   = m_orig_bounds;
      vtkm::Bounds bounds = m_bounds;

      m_orig_bounds = other.m_orig_bounds;
      m_bounds      = other.m_bounds;

      other.m_orig_bounds = orig;
      other.m_bounds      = bounds;

      m_pixels.swap(other.m_pixels);
      m_depths.swap(other.m_depths);

    }

    void Clear()
    {
      vtkm::Bounds empty;
      m_orig_bounds = empty;
      m_bounds = empty;
      m_pixels.clear();
      m_depths.clear();
    }

    std::string ToString() const
    {
      std::stringstream ss;
      ss<<"Total size pixels "<< (int) m_pixels.size() / 4;
      ss<<" tile dims: {"<<m_bounds.X.Min<<","<< m_bounds.Y.Min<<"} - ";
      ss<<"{"<<m_bounds.X.Max<<","<<m_bounds.Y.Max<<"}\n";;
      return ss.str();
    }

    void Save(const std::string &name,
              const std::vector<std::string> &comments);
};

struct CompositeOrderSort
{
  inline bool operator()(const Image &lhs, const Image &rhs) const
  {
    return lhs.m_composite_order < rhs.m_composite_order;
  }
};
} //namespace  vtkh
#endif
