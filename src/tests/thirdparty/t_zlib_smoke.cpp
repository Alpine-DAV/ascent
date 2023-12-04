//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_zlib_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "zlib.h"

//-----------------------------------------------------------------------------
TEST(zlib_smoke, zlib_basic)
{

  // exercise compress and uncompress interface

  Bytef compress_src[32];
  Bytef compress_dest[64];
  Bytef uncompress_dest[64];

  uLongf compress_src_len    = 32;
  uLongf compress_dest_len   = 64;
  uLongf uncompress_dest_len = 64;

  for(int i=0;i<32;i++)
  {
    compress_dest[i] = 0;
    uncompress_dest[i] = 0;
    // some pattern
    compress_dest[i] = i > 4 && i < 28;
  }

  int compress_res = compress(compress_dest,
                              &compress_dest_len,
                              compress_src,
                              compress_src_len);

  EXPECT_EQ(Z_OK,compress_res);


  int uncompress_res = uncompress(uncompress_dest,
                                  &uncompress_dest_len,
                                  compress_dest,
                                  compress_dest_len);
  EXPECT_EQ(Z_OK,uncompress_res);

  for(int i=0;i<32;i++)
  {
    EXPECT_EQ(compress_src[i],uncompress_dest[i]);
  }
}

