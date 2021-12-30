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

#include "ascent_png_compare.hpp"
#include "ascent_png_decoder.hpp"
#include "ascent_png_encoder.hpp"

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
PNGCompare::PNGCompare()
  : m_color_tolerance(4)
{}

//-----------------------------------------------------------------------------
PNGCompare::~PNGCompare()
{
}

void PNGCompare::ColorTolerance(int color_tolerance)
{
  if(color_tolerance < 0 || color_tolerance > 255)
  {
    ASCENT_ERROR("Color tolerance must be between 0-255: "<<color_tolerance);
  }
  m_color_tolerance = color_tolerance;
}

void
PNGCompare::DiffImage(const unsigned char *buff_1,
                      const unsigned char *buff_2,
                      const int width,
                      const int height,
                      const std::string out_name)
{
  const int size = width * height;
  std::vector<unsigned char> out_buff;
  out_buff.resize(size*4);

  for(int i = 0; i < size; ++i)
  {
      const int offset = i * 4;
      int r_diff = abs(int(buff_1[offset + 0]) - int(buff_2[offset + 0]));
      int g_diff = abs(int(buff_1[offset + 1]) - int(buff_2[offset + 1]));
      int b_diff = abs(int(buff_1[offset + 2]) - int(buff_2[offset + 2]));
      int a_diff = abs(int(buff_1[offset + 3]) - int(buff_2[offset + 3]));

      if( r_diff > m_color_tolerance ||
          g_diff > m_color_tolerance ||
          b_diff > m_color_tolerance ||
          a_diff > m_color_tolerance)
      {
        out_buff[offset+0] = 255;
        out_buff[offset+1] = 255;
        out_buff[offset+2] = 255;
        out_buff[offset+3] = 255;
      }
      else
      {
        out_buff[offset+0] = 0;
        out_buff[offset+1] = 0;
        out_buff[offset+2] = 0;
        out_buff[offset+3] = 255;
      }
  }

  int res = lpng::lodepng_encode32_file(out_name.c_str(),
                                  &out_buff[0],
                                  width,
                                  height);
  if(res)
  {
    ASCENT_ERROR("Failed to encode and save image diff file "<<out_name);
  }
}

bool
PNGCompare::Compare(const std::string &img1,
                    const std::string &img2,
                    conduit::Node &info,
                    const float tolerance)
{

  bool res = true;

  unsigned char *buff_1 = nullptr, *buff_2 = nullptr;
  int w1, w2, h1, h2;
  PNGDecoder decoder;

  decoder.Decode(buff_1, w1, h1, img1);
  decoder.Decode(buff_2, w2, h2, img2);

  if(w1 != w2 || h1 != h2)
  {
    info["dims_match"] = "false";
    res = false;
  }
  else
  {
    info["dims_match"] = "true";
  }

  if(res)
  {
    int diff = 0;
    const int image_size = w1 * h1;
    for(int i = 0; i < image_size; ++i)
    {
      const int offset = i * 4;
      int r_diff = abs(int(buff_1[offset + 0]) - int(buff_2[offset + 0]));
      int g_diff = abs(int(buff_1[offset + 1]) - int(buff_2[offset + 1]));
      int b_diff = abs(int(buff_1[offset + 2]) - int(buff_2[offset + 2]));
      int a_diff = abs(int(buff_1[offset + 3]) - int(buff_2[offset + 3]));

      if( r_diff > m_color_tolerance ||
          g_diff > m_color_tolerance ||
          b_diff > m_color_tolerance ||
          a_diff > m_color_tolerance)
      {
        diff++;
      }

    }

    float percent_diff = float(diff)/float(image_size);
    info["percent_diff"] = percent_diff;
    info["tolerance"] = tolerance;
    info["pass"] = "true";

    if(percent_diff > tolerance)
    {
      info["pass"] = "false";
      std::string file_name;
      std::string path;

      conduit::utils::rsplit_file_path(img1,
                                       file_name,
                                       path);

      std::string diff_name = conduit::utils::join_file_path(path,"diff_" + file_name);
      info["diff_image"] = diff_name;
      DiffImage(buff_1, buff_2, w1, h1, diff_name);
      res = false;
    }
  }

  free(buff_1);
  free(buff_2);
  return res;
}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



