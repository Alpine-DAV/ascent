//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_png_decoder.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_png_decoder.hpp"

#include "ascent_logging.hpp"

// standard includes
#include <stdlib.h>

// thirdparty includes
#include <lodepng.h>

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
PNGDecoder::PNGDecoder()
{}

//-----------------------------------------------------------------------------
PNGDecoder::~PNGDecoder()
{
}


void
PNGDecoder::Decode(unsigned char *&rgba,
                   int &width,
                   int &height,
                   const std::string &file_name)
{
  unsigned w,h;
  unsigned int res = lpng::lodepng_decode32_file(&rgba, &w, &h, file_name.c_str());

  width = w;
  height = h;

  if(res)
  {
    ASCENT_ERROR("Error decoding png "<<file_name<<"  code "<<res);
  }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



