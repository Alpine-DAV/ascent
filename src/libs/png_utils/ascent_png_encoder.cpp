//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_png_encoder.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_png_encoder.hpp"

// standard includes
#include <stdlib.h>

// thirdparty includes
#include <conduit.hpp>
#include <lodepng.h>

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
PNGEncoder::PNGEncoder()
:m_buffer(NULL),
 m_buffer_size(0)
{}

//-----------------------------------------------------------------------------
PNGEncoder::~PNGEncoder()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const unsigned char *rgba_in,
                   const int width,
                   const int height)
{
    Cleanup();

    // upside down relative to what lodepng wants
    unsigned char *rgba_flip = new unsigned char[width * height *4];

    for (int y=0; y<height; ++y)
    {
        memcpy(&(rgba_flip[y*width*4]),
               &(rgba_in[(height-y-1)*width*4]),
               width*4);
    }

     unsigned error = lpng::lodepng_encode_memory(&m_buffer,
                                            &m_buffer_size,
                                            &rgba_flip[0],
                                            width,
                                            height,
                                            lpng::LCT_RGBA, // these settings match those for
                                            8);       // lodepng_encode32_file

    delete [] rgba_flip;

    if(error)
    {
        CONDUIT_WARN("lodepng_encode_memory failed")
    }
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const float *rgba_in,
                   const int width,
                   const int height)
{
    Cleanup();

    // upside down relative to what lodepng wants
    unsigned char *rgba_flip = new unsigned char[width * height *4];


    for(int x = 0; x < width; ++x)

#ifdef ASCENT_OPENMP_ENABLED
        #pragma omp parrallel for
#endif
        for (int y = 0; y < height; ++y)
        {
            int inOffset = (y * width + x) * 4;
            int outOffset = ((height - y - 1) * width + x) * 4;
            rgba_flip[outOffset + 0] = (unsigned char)(rgba_in[inOffset + 0] * 255.f);
            rgba_flip[outOffset + 1] = (unsigned char)(rgba_in[inOffset + 1] * 255.f);
            rgba_flip[outOffset + 2] = (unsigned char)(rgba_in[inOffset + 2] * 255.f);
            rgba_flip[outOffset + 3] = (unsigned char)(rgba_in[inOffset + 3] * 255.f);
        }

     unsigned error = lpng::lodepng_encode_memory(&m_buffer,
                                            &m_buffer_size,
                                            &rgba_flip[0],
                                            width,
                                            height,
                                            lpng::LCT_RGBA, // these settings match those for
                                            8);       // lodepng_encode32_file

    delete [] rgba_flip;

    if(error)
    {
        CONDUIT_WARN("lodepng_encode_memory failed")
    }
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Save(const std::string &filename)
{
    if(m_buffer == NULL)
    {
        CONDUIT_WARN("Save must be called after encode()")
        /// we have a problem ...!
        return;
    }

    unsigned error = lpng::lodepng_save_file(m_buffer,
                                       m_buffer_size,
                                       filename.c_str());
    if(error)
    {
        CONDUIT_WARN("Error saving PNG buffer to file: " << filename);
    }
}

//-----------------------------------------------------------------------------
void *
PNGEncoder::PngBuffer()
{
    return (void*)m_buffer;
}

//-----------------------------------------------------------------------------
size_t
PNGEncoder::PngBufferSize()
{
    return m_buffer_size;
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Base64Encode()
{
    if(m_buffer == NULL)
    {
        CONDUIT_WARN("base64_encode must be called after encode()")
        return;
    }

    // base64 encode the raw png data
    m_base64_data.set(DataType::char8_str(m_buffer_size*2));
    utils::base64_encode(m_buffer,
                         m_buffer_size,
                         m_base64_data.data_ptr());
}


//-----------------------------------------------------------------------------
Node &
PNGEncoder::Base64Node()
{
    return m_base64_data;
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Cleanup()
{
    if(m_buffer != NULL)
    {
        //lodepng_free(m_buffer);
        // ^-- Not found even if LODEPNG_COMPILE_ALLOCATORS is defined?
        // simply use "free"
        free(m_buffer);
        m_buffer = NULL;
        m_buffer_size = 0;
    }
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



