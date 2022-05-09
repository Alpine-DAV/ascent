#ifndef APCOMP_IMAGE_HPP
#define APCOMP_IMAGE_HPP

#include <sstream>
#include <vector>
#include <apcomp/bounds.hpp>

#include <apcomp/apcomp_exports.h>

namespace apcomp
{

struct APCOMP_API Image
{
    // The image bounds are indicated by a grid starting at
    // 1-width and 1-height. Actual width would be calculated
    // m_bounds.m_max_x - m_bounds.m_min_x + 1
    // 1024 - 1 + 1 = 1024
    Bounds                       m_orig_bounds;
    Bounds                       m_bounds;
    std::vector<unsigned char>   m_pixels;
    std::vector<float>           m_depths;
    int                          m_orig_rank;
    bool                         m_has_transparency;
    int                          m_composite_order;
    bool                         m_gl_depth; // expect depth values in (0,1)

    Image();

    Image(const Bounds &bounds);

    // init this image based on the original bounds
    // of the other image
    void InitOriginal(const Image &other);

    int GetNumberOfPixels() const;

    void SetHasTransparency(bool has_transparency);

    bool HasTransparency();

    void Init(const float *color_buffer,
              const float *depth_buffer,
              int width,
              int height,
              bool gl_depth = true,
              int composite_order = -1);

    void Init(const unsigned char *color_buffer,
              const float *depth_buffer,
              int width,
              int height,
              bool gl_depth = true,
              int composite_order = -1);

    void CompositeBackground(const float color[4]);
    //
    // Fill this image with a sub-region of another image
    //
    void SubsetFrom(const Image &image,
                    const Bounds &sub_region);

    // sets alls pixels in this image to a color
    // based on the input integer
    void Color(int color);

    //
    // Fills the passed in image with the contents of this image
    //
    void SubsetTo(Image &image) const;

    //
    // Swap internal storage with another image
    //
    void Swap(Image &other);

    void Clear();

    //
    // Summary of this image
    //
    std::string ToString() const;

    void Save(std::string name);
    void SaveDepth(std::string name);
};

} //namespace  apcomp
#endif
