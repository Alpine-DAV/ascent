//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-749865
//
// All rights reserved.
//
// This file is part of Rover.
//
// Please also read rover/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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

#ifdef ROVER_ENABLE_OPENMP
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

#ifdef ROVER_ENABLE_OPENMP
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

#ifdef ROVER_ENABLE_OPENMP
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
