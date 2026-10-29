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
// Helper for converting floating point color channels to bytes.
//
// The input value is clamped to [0,1] before scaling and truncation.
// Direct conversion of floating point values outside the representable
// range of unsigned char is undefined behavior in C++, and compilers
// handle it differently (x86 wrap-around modulo 256 vs saturation),
// which silently corrupts output pixels when colors exceed 1.0
// (e.g., additive volume compositing).
//-----------------------------------------------------------------------------
template<typename T>
inline unsigned char
clamp_to_byte(const T &value)
{
    const T clamped = value < T(0) ? T(0) : (value > T(1) ? T(1) : value);
    return static_cast<unsigned char>(clamped * T(255));
}

//-----------------------------------------------------------------------------
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
            rgba_flip[outOffset + 0] = clamp_to_byte(rgba_in[inOffset + 0]);
            rgba_flip[outOffset + 1] = clamp_to_byte(rgba_in[inOffset + 1]);
            rgba_flip[outOffset + 2] = clamp_to_byte(rgba_in[inOffset + 2]);
            rgba_flip[outOffset + 3] = clamp_to_byte(rgba_in[inOffset + 3]);
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
        CONDUIT_WARN("lodepng_encode_memory failed");
    }
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const double *rgba_in,
                   const int width,
                   const int height)
{
  unsigned char *rgba = new unsigned char[width * height *4];


  for(int x = 0; x < width; ++x)

#ifdef ASCENT_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for (int y = 0; y < height; ++y)
    {
      int offset = (y * width + x) * 4;
      rgba[offset + 0] = clamp_to_byte(rgba_in[offset + 0]);
      rgba[offset + 1] = clamp_to_byte(rgba_in[offset + 1]);
      rgba[offset + 2] = clamp_to_byte(rgba_in[offset + 2]);
      rgba[offset + 3] = clamp_to_byte(rgba_in[offset + 3]);
    }

  Encode(rgba, width, height);
  delete[] rgba;
}


//-----------------------------------------------------------------------------
void
PNGEncoder::EncodeChannel(const double *buffer_in,
                          const int width,
                          const int height)
{

  unsigned char *rgba = new unsigned char[width * height *4];

  for(int x = 0; x < width; ++x)

#ifdef ASCENT_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for (int y = 0; y < height; ++y)
    {
      int offset = (y * width + x);
      rgba[offset + 0] = clamp_to_byte(buffer_in[offset]);
      rgba[offset + 1] = clamp_to_byte(buffer_in[offset]);
      rgba[offset + 2] = clamp_to_byte(buffer_in[offset]);
      rgba[offset + 3] = 255;
    }

  Encode(rgba, width, height);
  delete[] rgba;
}

//-----------------------------------------------------------------------------
void
PNGEncoder::EncodeChannel(const float *buffer_in,
                          const int width,
                          const int height)
{

  unsigned char *rgba = new unsigned char[width * height *4];

  for(int x = 0; x < width; ++x)

#ifdef ASCENT_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for (int y = 0; y < height; ++y)
    {
      int offset = (y * width + x);
      rgba[offset + 0] = clamp_to_byte(buffer_in[offset]);
      rgba[offset + 1] = clamp_to_byte(buffer_in[offset]);
      rgba[offset + 2] = clamp_to_byte(buffer_in[offset]);
      rgba[offset + 3] = 255;
    }

  Encode(rgba, width, height);
  delete[] rgba;
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const unsigned char *rgba_in,
                   const int width,
                   const int height,
                   const std::vector<std::string> &comments)
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
 
    lpng::LodePNGState state;
    lpng::lodepng_state_init(&state);
    // use less aggressive compression
    state.encoder.zlibsettings.btype = 2;
    state.encoder.zlibsettings.use_lz77 = 0;
    if(comments.size() % 2 != 0)
    {
        CONDUIT_INFO("PNGEncoder::Encode comments missing value for the last key.\n"
                     << "Ignoring the last key.\n");
    }
    if(comments.size() > 1)
    {
        lpng::lodepng_info_init(&state.info_png);
        // Comments are in pairs with a key and a value, using
        // comments.size()-1 ensures that we don't use the last
        // comment if the length of the vector isn't a multiple of 2.
        for (int i = 0; i < comments.size()-1; i += 2)
            lpng::lodepng_add_text(&state.info_png, comments[i].c_str(),
                                                    comments[i+1].c_str());
    }

    unsigned error = lodepng_encode(&m_buffer,
                                    &m_buffer_size,
                                    &rgba_flip[0],
                                    width,
                                    height,
                                    &state);

    delete [] rgba_flip;

    if(error)
    {
       CONDUIT_WARN("lodepng_encode_memory failed");
    }
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const float *rgba_in,
                   const int width,
                   const int height,
                   const std::vector<std::string> &comments)
{
    Cleanup();

    // upside down relative to what lodepng wants
    unsigned char *rgba_flip = new unsigned char[width * height *4];


    for(int x = 0; x < width; ++x)

#ifdef ASCENT_OPENMP_ENABLED
        #pragma omp parallel for
#endif
        for (int y = 0; y < height; ++y)
        {
            int inOffset = (y * width + x) * 4;
            int outOffset = ((height - y - 1) * width + x) * 4;
            rgba_flip[outOffset + 0] = clamp_to_byte(rgba_in[inOffset + 0]);
            rgba_flip[outOffset + 1] = clamp_to_byte(rgba_in[inOffset + 1]);
            rgba_flip[outOffset + 2] = clamp_to_byte(rgba_in[inOffset + 2]);
            rgba_flip[outOffset + 3] = clamp_to_byte(rgba_in[inOffset + 3]);
        }

    lpng::LodePNGState state;
    lpng::lodepng_state_init(&state);
    // use less aggressive compression
    state.encoder.zlibsettings.btype = 2;
    state.encoder.zlibsettings.use_lz77 = 0;
    if(comments.size() % 2 != 0)
    {
        CONDUIT_INFO("PNGEncoder::Encode comments missing value for the last key.\n"
                     << "Ignoring the last key.\n");
    }
    if(comments.size() > 1)
    {
        vtkh:lodepng_info_init(&state.info_png);
        // Comments are in pairs with a key and a value, using
        // comments.size()-1 ensures that we don't use the last
        // comment if the length of the vector isn't a multiple of 2.
        for (int i = 0; i < comments.size()-1; i += 2)
            lpng::lodepng_add_text(&state.info_png, comments[i].c_str(),
                                                    comments[i+1].c_str());
    }

    unsigned error = lodepng_encode(&m_buffer,
                                    &m_buffer_size,
                                    &rgba_flip[0],
                                    width,
                                    height,
                                    &state);

    delete [] rgba_flip;

    if(error)
    {
        CONDUIT_WARN("lodepng_encode_memory failed");
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



