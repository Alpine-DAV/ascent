//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


// standard includes
#include <stdlib.h>

// rover includes
#include <rover_exceptions.hpp>
#include <utils/png_encoder.hpp>
#include <utils/rover_logging.hpp>
namespace rover {

PNGEncoder::PNGEncoder()
{}

//-----------------------------------------------------------------------------
PNGEncoder::~PNGEncoder()
{
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const unsigned char *rgba_in,
                   const int width,
                   const int height)
{
  m_encoder.Encode(rgba_in, width, height);
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const float *rgba_in,
                   const int width,
                   const int height)
{
  m_encoder.Encode(rgba_in, width, height);
}
//-----------------------------------------------------------------------------
void
PNGEncoder::Encode(const double *rgba_in,
                   const int width,
                   const int height)
{
  unsigned char *rgba = new unsigned char[width * height *4];


  for(int x = 0; x < width; ++x)

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for (int y = 0; y < height; ++y)
    {
      int offset = (y * width + x) * 4;
      rgba[offset + 0] = (unsigned char)(rgba_in[offset + 0] * 255.);
      rgba[offset + 1] = (unsigned char)(rgba_in[offset + 1] * 255.);
      rgba[offset + 2] = (unsigned char)(rgba_in[offset + 2] * 255.);
      rgba[offset + 3] = (unsigned char)(rgba_in[offset + 3] * 255.);
    }

  m_encoder.Encode(rgba, width, height);
  delete[] rgba;
}

void
PNGEncoder::EncodeChannel(const double *buffer_in,
                          const int width,
                          const int height)
{

  unsigned char *rgba = new unsigned char[width * height *4];

  for(int x = 0; x < width; ++x)

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for (int y = 0; y < height; ++y)
    {
      int offset = (y * width + x);
      rgba[offset + 0] = (unsigned char)(buffer_in[offset] * 255.);
      rgba[offset + 1] = (unsigned char)(buffer_in[offset] * 255.);
      rgba[offset + 2] = (unsigned char)(buffer_in[offset] * 255.);
      rgba[offset + 3] = 255;
    }

  m_encoder.Encode(rgba, width, height);
  delete[] rgba;
}
void
PNGEncoder::EncodeChannel(const float *buffer_in,
                          const int width,
                          const int height)
{

  unsigned char *rgba = new unsigned char[width * height *4];

  for(int x = 0; x < width; ++x)

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for (int y = 0; y < height; ++y)
    {
      int offset = (y * width + x);
      rgba[offset + 0] = (unsigned char)(buffer_in[offset] * 255.);
      rgba[offset + 1] = (unsigned char)(buffer_in[offset] * 255.);
      rgba[offset + 2] = (unsigned char)(buffer_in[offset] * 255.);
      rgba[offset + 3] = 255;
    }

  m_encoder.Encode(rgba, width, height);
  delete[] rgba;
}

//-----------------------------------------------------------------------------
void
PNGEncoder::Save(const std::string &filename)
{
  ROVER_INFO("Saved png: "<<filename);
  m_encoder.Save(filename);
}

} // namespace rover
