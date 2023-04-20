//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_COMPOSITOR_BASE_HPP
#define APCOMP_COMPOSITOR_BASE_HPP

#include <apcomp/apcomp_config.h>

#include <sstream>
#include <apcomp/apcomp_exports.h>
#include <apcomp/image.hpp>

namespace apcomp
{

class APCOMP_API Compositor
{
public:
    enum CompositeMode {
                         Z_BUFFER_SURFACE_WORLD,// expect world depth no transparency
                         Z_BUFFER_SURFACE_GL,   // zbuffer composite no transparency
                         Z_BUFFER_BLEND,        // zbuffer composite with transparency
                         VIS_ORDER_BLEND        // blend images in a specific order
                       };
    Compositor();

    virtual ~Compositor();

    void SetCompositeMode(CompositeMode composite_mode);

    void ClearImages();

    void AddImage(const unsigned char *color_buffer,
                  const float *        depth_buffer,
                  const int            width,
                  const int            height);

    void AddImage(const float *color_buffer,
                  const float *depth_buffer,
                  const int    width,
                  const int    height);

    void AddImage(const unsigned char *color_buffer,
                  const float *        depth_buffer,
                  const int            width,
                  const int            height,
                  const int            vis_order);

    void AddImage(const float *color_buffer,
                  const float *depth_buffer,
                  const int    width,
                  const int    height,
                  const int    vis_order);

    Image Composite();

    virtual void         Cleanup();

    std::string          GetLogString();

    unsigned char * ConvertBuffer(const float *buffer, const int size)
    {
        unsigned char *ubytes = new unsigned char[size];

#ifdef APCOMP_OPENMP_ENABLED
        #pragma omp parallel for
#endif
        for(int i = 0; i < size; ++i)
        {
            ubytes[i] = static_cast<unsigned char>(buffer[i] * 255.f);
        }

        return ubytes;
    }

protected:
    virtual void CompositeZBufferSurface();
    virtual void CompositeZBufferBlend();
    virtual void CompositeVisOrder();

    std::stringstream   m_log_stream;
    CompositeMode       m_composite_mode;
    std::vector<Image>  m_images;
};

};

#endif


