//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// Please also read ascent/LICENSE
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

#include <apcomp/utils/png_compare.hpp>
#include <apcomp/utils/png_decoder.hpp>
#include <apcomp/utils/png_encoder.hpp>
#include <apcomp/utils/file_utils.hpp>
#include <apcomp/error.hpp>

// standard includes
#include <stdlib.h>
#include <iostream>
#include <vector>

namespace apcomp
{

//-----------------------------------------------------------------------------
PNGCompare::PNGCompare()
{}

//-----------------------------------------------------------------------------
PNGCompare::~PNGCompare()
{
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
      if( buff_1[offset + 0] != buff_2[offset + 0] ||
          buff_1[offset + 1] != buff_2[offset + 1] ||
          buff_1[offset + 2] != buff_2[offset + 2] ||
          buff_1[offset + 3] != buff_2[offset + 3])
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

  PNGEncoder encoder;
  encoder.Encode(&out_buff[0],
                 width,
                 height);
  encoder.Save(out_name);
}

bool
PNGCompare::Compare(const std::string &img1,
                    const std::string &img2,
                    float &difference,
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
    res = false;
  }

  if(res)
  {
    int diff = 0;
    const int image_size = w1 * h1;
    for(int i = 0; i < image_size; ++i)
    {
      const int offset = i * 4;
      if( buff_1[offset + 0] != buff_2[offset + 0] ||
          buff_1[offset + 1] != buff_2[offset + 1] ||
          buff_1[offset + 2] != buff_2[offset + 2] ||
          buff_1[offset + 3] != buff_2[offset + 3])
      {
        diff++;
      }

    }

    difference = float(diff)/float(image_size);

    if(difference > tolerance)
    {
      std::string file_name = img1;
      std::string path;
      rsplit_file_path(img1, file_name, path);

      std::string diff_name = "diff_" + file_name;
      std::string out_name = apcomp::join_file_path(path,diff_name);
      DiffImage(buff_1, buff_2, w1, h1, out_name);
      res = false;
    }
  }

  free(buff_1);
  free(buff_2);
  return res;
}

} // namespace apcomp



