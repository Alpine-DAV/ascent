#include "png_decoder.hpp"

// standard includes
#include <stdlib.h>
#include <iostream>

// thirdparty includes
#include <lodepng.h>

namespace dray
{

//-----------------------------------------------------------------------------
PNGDecoder::PNGDecoder()
{}

//-----------------------------------------------------------------------------
PNGDecoder::~PNGDecoder()
{
}


void
PNGDecoder::decode(unsigned char *&rgba,
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
    std::cerr<<"Error decoding png "<<file_name<<"  code "<<res<<"\n";
  }
}
void
PNGDecoder::decode_raw(unsigned char *&rgba,
                       int &width,
                       int &height,
                       const unsigned char *raw_png,
                       const size_t raw_size)
{

  unsigned w,h;
  unsigned int res = lpng::lodepng_decode32(&rgba, &w, &h, raw_png, raw_size);
  width = w;
  height = h;

  if(res)
  {
    std::cerr<<"Error decoding raw png code "<<res<<"\n";
  }
}

};
