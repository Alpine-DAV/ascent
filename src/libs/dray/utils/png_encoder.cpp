#include <dray/utils/png_encoder.hpp>

// standard includes
#include <stdlib.h>
#include <iostream>

// thirdparty includes
#include <lodepng.h>

namespace dray
{

//-----------------------------------------------------------------------------
PNGEncoder::PNGEncoder()
:m_buffer(NULL),
 m_buffer_size(0)
{}

//-----------------------------------------------------------------------------
PNGEncoder::~PNGEncoder()
{
    cleanup();
}

//-----------------------------------------------------------------------------
void
PNGEncoder::encode(const uint8 *rgba_in,
                   const int32 width,
                   const int32 height)
{
  cleanup();

  // upside down relative to what lodepng wants
  uint8 *rgba_flip = new uint8[width * height *4];

  for (int32 y = 0; y < height; ++y)
  {
    memcpy(&(rgba_flip[y*width*4]),
           &(rgba_in[(height-y-1)*width*4]),
           width*4);
  }

   unsigned error = lodepng_encode_memory(&m_buffer,
                                          &m_buffer_size,
                                          &rgba_flip[0],
                                          width,
                                          height,
                                          lpng::LCT_RGBA, // these settings match those for
                                          8);       // lodepng_encode32_file

  delete [] rgba_flip;

  if(error)
  {
    std::cerr<<"lodepng_encode_memory failed\n";
  }
}

//-----------------------------------------------------------------------------
void
PNGEncoder::encode(const float32 *rgba_in,
                   const int32 width,
                   const int32 height)
{
  cleanup();

  // upside down relative to what lodepng wants
  uint8 *rgba_flip = new uint8[width * height *4];


  for(int32 x = 0; x < width; ++x)
  {

#ifdef DRAY_OPENMP_ENABLED
   #pragma omp parallel for
#endif
    for (int32 y = 0; y < height; ++y)
    {
      int32 inOffset = (y * width + x) * 4;
      int32 outOffset = ((height - y - 1) * width + x) * 4;
      rgba_flip[outOffset + 0] = (uint8)(rgba_in[inOffset + 0] * 255.f);
      rgba_flip[outOffset + 1] = (uint8)(rgba_in[inOffset + 1] * 255.f);
      rgba_flip[outOffset + 2] = (uint8)(rgba_in[inOffset + 2] * 255.f);
      rgba_flip[outOffset + 3] = (uint8)(rgba_in[inOffset + 3] * 255.f);
    }
  }
   unsigned error = lodepng_encode_memory(&m_buffer,
                                          &m_buffer_size,
                                          &rgba_flip[0],
                                          width,
                                          height,
                                          lpng::LCT_RGBA, // these settings match those for
                                          8);       // lodepng_encode32_file

  delete [] rgba_flip;

  if(error)
  {
    std::cerr<<"lodepng_encode_memory failed\n";
  }
}

//-----------------------------------------------------------------------------
void
PNGEncoder::save(const std::string &filename)
{
  if(m_buffer == NULL)
  {
    std::cerr<<"Save must be called after encode()\n";
      /// we have a problem ...!
      return;
  }

  unsigned error = lpng::lodepng_save_file(m_buffer,
                                           m_buffer_size,
                                           filename.c_str());
  if(error)
  {
    std::cerr<<"Error saving PNG buffer to file: " << filename<<"\n";
  }
}

//-----------------------------------------------------------------------------
void *
PNGEncoder::png_buffer()
{
  return (void*)m_buffer;
}

//-----------------------------------------------------------------------------
size_t
PNGEncoder::png_buffer_size()
{
  return m_buffer_size;
}

//----------------------------------------------------------------------------
void
PNGEncoder::cleanup()
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



