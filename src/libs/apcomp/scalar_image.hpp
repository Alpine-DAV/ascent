//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_SCALAR_IMAGE_HPP
#define APCOMP_SCALAR_IMAGE_HPP

#include <apcomp/apcomp_config.h>

#include <cmath>
#include <vector>
#include <vector>
#include <apcomp/bounds.hpp>

#include <apcomp/apcomp_exports.h>
#include <assert.h>

namespace apcomp
{

struct APCOMP_API ScalarImage
{
    // The image bounds are indicated by a grid starting at
    // 1-width and 1-height. Actual width would be calculated
    // m_bounds.m_max_x - m_bounds.m_min_x + 1
    // 1024 - 1 + 1 = 1024
    Bounds                       m_orig_bounds;
    Bounds                       m_bounds;
    std::vector<unsigned char>   m_payloads;
    std::vector<float>           m_depths;
    int                          m_orig_rank;
    int                          m_payload_bytes; // Size of the payload in bytes
    float                        m_default_value;

    ScalarImage()
    {}

    ScalarImage(const Bounds &bounds, const int payload_bytes)
      : m_orig_bounds(bounds),
        m_bounds(bounds),
        m_orig_rank(-1),
        m_payload_bytes(payload_bytes)
    {
      m_default_value = std::nanf("");
      const int dx  = bounds.m_max_x - bounds.m_min_x + 1;
      const int dy  = bounds.m_max_y - bounds.m_min_y + 1;
      m_payloads.resize(dx * dy * m_payload_bytes);
      m_depths.resize(dx * dy);
    }

    void InitOriginal(const ScalarImage &other)
    {
      m_orig_bounds = other.m_orig_bounds;
      m_bounds = other.m_orig_bounds;
      m_payload_bytes = other.m_payload_bytes;
      m_default_value = other.m_default_value;

      const int dx  = m_bounds.m_max_x - m_bounds.m_min_x + 1;
      const int dy  = m_bounds.m_max_y - m_bounds.m_min_y + 1;
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
      m_bounds.m_min_x = 1;
      m_bounds.m_min_y = 1;
      m_bounds.m_max_x = width;
      m_bounds.m_max_y = height;
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
    void SubsetFrom(const ScalarImage &image,
                    const Bounds &sub_region)
    {
      m_orig_bounds = image.m_orig_bounds;
      m_bounds = sub_region;
      m_orig_rank = image.m_orig_rank;
      m_payload_bytes = image.m_payload_bytes;

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

      size_t buffer_size = s_dx * s_dy * m_payload_bytes;

      m_payloads.resize(buffer_size);
      m_depths.resize(s_dx * s_dy);


#ifdef APCOMP_OPENMP_ENABLED
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
    void SubsetTo(ScalarImage &image) const
    {
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

#ifdef APCOMP_OPENMP_ENABLED
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

    void Swap(ScalarImage &other)
    {
      Bounds orig   = m_orig_bounds;
      Bounds bounds = m_bounds;

      m_orig_bounds = other.m_orig_bounds;
      m_bounds      = other.m_bounds;

      other.m_orig_bounds = orig;
      other.m_bounds      = bounds;

      m_payloads.swap(other.m_payloads);
      m_depths.swap(other.m_depths);

    }

    void Clear()
    {
      Bounds empty;
      m_orig_bounds = empty;
      m_bounds = empty;
      m_payloads.clear();
      m_depths.clear();
    }

    std::string ToString() const
    {
      std::stringstream ss;
      ss<<"Total size pixels "<< (int) m_depths.size();
      ss<<" tile dims: {"<<m_bounds.m_min_x<<","<< m_bounds.m_min_y<<"} - ";
      ss<<"{"<<m_bounds.m_max_x<<","<<m_bounds.m_max_y<<"}\n";;
      return ss.str();
    }

    void Save(std::string name);
};

} //namespace  apcomp
#endif
