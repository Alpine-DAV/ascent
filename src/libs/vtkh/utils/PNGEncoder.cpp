#include "PNGEncoder.hpp"

// standard includes
#include <stdlib.h>
#include <iostream>

// thirdparty includes
#include <lodepng.h>

namespace vtkh
{

PNGEncoder::PNGEncoder()
:m_buffer(NULL),
 m_buffer_size(0)
{}

PNGEncoder::~PNGEncoder()
{
    Cleanup();
}

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
 
    vtkh::LodePNGState state;
    vtkh::lodepng_state_init(&state);
    // use less aggressive compression
    state.encoder.zlibsettings.btype = 2;
    state.encoder.zlibsettings.use_lz77 = 0;

    unsigned error = lodepng_encode(&m_buffer,
                                    &m_buffer_size,
                                    &rgba_flip[0],
                                    width,
                                    height,
                                    &state);

    delete [] rgba_flip;

    if(error)
    {
        std::cerr<<"lodepng_encode_memory failed\n";
    }
}

void
PNGEncoder::Encode(const float *rgba_in,
                   const int width,
                   const int height)
{
    Cleanup();

    // upside down relative to what lodepng wants
    unsigned char *rgba_flip = new unsigned char[width * height *4];


    for(int x = 0; x < width; ++x)

#ifdef VTKH_USE_OPENMP
        #pragma omp parallel for
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

    vtkh::LodePNGState state;
    vtkh::lodepng_state_init(&state);
    // use less aggressive compression
    state.encoder.zlibsettings.btype = 2;
    state.encoder.zlibsettings.use_lz77 = 0;

    unsigned error = lodepng_encode(&m_buffer,
                                    &m_buffer_size,
                                    &rgba_flip[0],
                                    width,
                                    height,
                                    &state);

    delete [] rgba_flip;

    if(error)
    {
        std::cerr<<"lodepng_encode_memory failed\n";
    }
}

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
 
    vtkh::LodePNGState state;
    vtkh::lodepng_state_init(&state);
    // use less aggressive compression
    state.encoder.zlibsettings.btype = 2;
    state.encoder.zlibsettings.use_lz77 = 0;
    if(comments.size() % 2 != 0)
    {
        std::cerr<<"PNGEncoder::Encode comments missing value for the last key.\n";
        std::cerr<<"Ignoring the last key.\n";
    }
    if(comments.size() > 1)
    {
        vtkh::lodepng_info_init(&state.info_png);
        // Comments are in pairs with a key and a value, using
        // comments.size()-1 ensures that we don't use the last
        // comment if the length of the vector isn't a multiple of 2.
        for (int i = 0; i < comments.size()-1; i += 2)
            vtkh::lodepng_add_text(&state.info_png, comments[i].c_str(),
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
        std::cerr<<"lodepng_encode_memory failed\n";
    }
}

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

#ifdef VTKH_USE_OPENMP
        #pragma omp parallel for
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

    vtkh::LodePNGState state;
    vtkh::lodepng_state_init(&state);
    // use less aggressive compression
    state.encoder.zlibsettings.btype = 2;
    state.encoder.zlibsettings.use_lz77 = 0;
    if(comments.size() % 2 != 0)
    {
        std::cerr<<"PNGEncoder::Encode comments missing value for the last key.\n";
        std::cerr<<"Ignoring the last key.\n";
    }
    if(comments.size() > 1)
    {
        vtkh:lodepng_info_init(&state.info_png);
        // Comments are in pairs with a key and a value, using
        // comments.size()-1 ensures that we don't use the last
        // comment if the length of the vector isn't a multiple of 2.
        for (int i = 0; i < comments.size()-1; i += 2)
            vtkh::lodepng_add_text(&state.info_png, comments[i].c_str(),
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
        std::cerr<<"lodepng_encode_memory failed\n";
    }
}

void
PNGEncoder::Save(const std::string &filename)
{
    if(m_buffer == NULL)
    {
        std::cerr<<"Save must be called after encode()\n";
        /// we have a problem ...!
        return;
    }

    unsigned error = lodepng_save_file(m_buffer,
                                       m_buffer_size,
                                       filename.c_str());
    if(error)
    {
        std::cerr<<"Error saving PNG buffer to file: " << filename<<"\n";
    }
}

void *
PNGEncoder::PngBuffer()
{
    return (void*)m_buffer;
}

size_t
PNGEncoder::PngBufferSize()
{
    return m_buffer_size;
}

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

};




