#ifndef VTKH_DIY_PAYLOAD_IMAGE_HPP
#define VTKH_DIY_PAYLOAD_IMAGE_HPP

#include <sstream>
#include <vector>
#include <vtkm/Bounds.h>

#include <vtkh/vtkh_exports.h>

namespace vtkh
{

struct VTKH_API PayloadImage
{
    // The image bounds are indicated by a grid starting at
    // 1-width and 1-height. Actual width would be calculated
    // m_bounds.X.Max - m_bounds.X.Min + 1
    // 1024 - 1 + 1 = 1024
    vtkm::Bounds                 m_orig_bounds;
    vtkm::Bounds                 m_bounds;
    std::vector<unsigned char>   m_payloads;
    std::vector<float>           m_depths;
    int                          m_orig_rank;
    int                          m_payload_bytes; // Size of the payload in bytes
    float                        m_default_value;

    PayloadImage()
    {}

    PayloadImage(const vtkm::Bounds &bounds, const int payload_bytes)
      : m_orig_bounds(bounds),
        m_bounds(bounds),
        m_orig_rank(-1),
        m_payload_bytes(payload_bytes)
    {
      m_default_value = vtkm::Nan32();
      const int dx  = bounds.X.Max - bounds.X.Min + 1;
      const int dy  = bounds.Y.Max - bounds.Y.Min + 1;
      m_payloads.resize(dx * dy * m_payload_bytes);
      m_depths.resize(dx * dy);
    }

    void InitOriginal(const PayloadImage &other)
    {
      m_orig_bounds = other.m_orig_bounds;
      m_bounds = other.m_orig_bounds;
      m_payload_bytes = other.m_payload_bytes;
      m_default_value = other.m_default_value;

      const int dx  = m_bounds.X.Max - m_bounds.X.Min + 1;
      const int dy  = m_bounds.Y.Max - m_bounds.Y.Min + 1;
      m_payloads.resize(dx * dy * m_payload_bytes);
      m_depths.resize(dx * dy);

      m_orig_rank = -1;
    }

    int GetNumberOfPixels() const
    {
      return static_cast<int>(m_depths.size());
    }

    void Init(const unsigned char *payload_buffer,
              const float *depth_buffer,
              int width,
              int height)
    {
      m_bounds.X.Min = 1;
      m_bounds.Y.Min = 1;
      m_bounds.X.Max = width;
      m_bounds.Y.Max = height;
      m_orig_bounds = m_bounds;
      const int size = width * height;
      m_payloads.resize(size * m_payload_bytes);
      m_depths.resize(size);

      std::copy(payload_buffer,
                payload_buffer + size * m_payload_bytes,
                &m_payloads[0]);

      std::copy(depth_buffer,
                depth_buffer + size,
                &m_depths[0]);
    }

    //
    // Fill this image with a sub-region of another image
    //
    void SubsetFrom(const PayloadImage &image,
                    const vtkm::Bounds &sub_region)
    {
      m_orig_bounds = image.m_orig_bounds;
      m_bounds = sub_region;
      m_orig_rank = image.m_orig_rank;
      m_payload_bytes = image.m_payload_bytes;

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

      size_t buffer_size = s_dx * s_dy * m_payload_bytes;

      m_payloads.resize(buffer_size);
      m_depths.resize(s_dx * s_dy);


#ifdef VTKH_OPENMP_ENABLED
        #pragma omp parallel for
#endif
      for(int y = start_y; y < end_y; ++y)
      {
        const int copy_to = (y - start_y) * s_dx;
        const int copy_from = y * dx + start_x;

        std::copy(&image.m_payloads[copy_from * m_payload_bytes],
                  &image.m_payloads[copy_from * m_payload_bytes] + s_dx * m_payload_bytes,
                  &m_payloads[copy_to * m_payload_bytes]);
        std::copy(&image.m_depths[copy_from],
                  &image.m_depths[copy_from] + s_dx,
                  &m_depths[copy_to]);
      }

    }

    //
    // Fills the passed in image with the contents of this image
    //
    void SubsetTo(PayloadImage &image) const
    {
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

        std::copy(&m_payloads[copy_from * m_payload_bytes],
                  &m_payloads[copy_from * m_payload_bytes] + s_dx * m_payload_bytes,
                  &image.m_payloads[copy_to * m_payload_bytes]);

        std::copy(&m_depths[copy_from],
                  &m_depths[copy_from] + s_dx,
                  &image.m_depths[copy_to]);
      }
    }

    void Swap(PayloadImage &other)
    {
      vtkm::Bounds orig   = m_orig_bounds;
      vtkm::Bounds bounds = m_bounds;

      m_orig_bounds = other.m_orig_bounds;
      m_bounds      = other.m_bounds;

      other.m_orig_bounds = orig;
      other.m_bounds      = bounds;

      m_payloads.swap(other.m_payloads);
      m_depths.swap(other.m_depths);

    }

    void Clear()
    {
      vtkm::Bounds empty;
      m_orig_bounds = empty;
      m_bounds = empty;
      m_payloads.clear();
      m_depths.clear();
    }

    std::string ToString() const
    {
      std::stringstream ss;
      ss<<"Total size pixels "<< (int) m_depths.size();
      ss<<" tile dims: {"<<m_bounds.X.Min<<","<< m_bounds.Y.Min<<"} - ";
      ss<<"{"<<m_bounds.X.Max<<","<<m_bounds.Y.Max<<"}\n";;
      return ss.str();
    }

    void Save(std::string name);
};

} //namespace  vtkh
#endif
