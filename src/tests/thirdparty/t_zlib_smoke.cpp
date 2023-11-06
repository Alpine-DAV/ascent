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
  Bytef compress_dest[32];
  Bytef compress_src[32];
  Bytef uncompress_dest[64];

  uLongf compress_dest_len   = 32;
  uLongf compress_src_len    = 32;
  uLongf uncompress_dest_len = 32;

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

  //
  // EXTERN int ZEXPORT compress OF((Bytef *dest,   uLongf *destLen, const Bytef *source, uLong sourceLen));
  // /*
  //          Compresses the source buffer into the destination buffer.  sourceLen is
  //      the byte length of the source buffer.  Upon entry, destLen is the total size
  //      of the destination buffer, which must be at least the value returned by
  //      compressBound(sourceLen).  Upon exit, destLen is the actual size of the
  //      compressed buffer.
  //
  //          compress returns Z_OK if success, Z_MEM_ERROR if there was not
  //      enough memory, Z_BUF_ERROR if there was not enough room in the output
  //      buffer.
  // */
  //
  // ZEXTERN int ZEXPORT uncompress OF((Bytef *dest,   uLongf *destLen, const Bytef *source, uLong sourceLen));
  // /*
  //          Decompresses the source buffer into the destination buffer.  sourceLen is
  //      the byte length of the source buffer.  Upon entry, destLen is the total size
  //      of the destination buffer, which must be large enough to hold the entire
  //      uncompressed data.  (The size of the uncompressed data must have been saved
  //      previously by the compressor and transmitted to the decompressor by some
  //      mechanism outside the scope of this compression library.) Upon exit, destLen
  //      is the actual size of the uncompressed buffer.
  //
  //          uncompress returns Z_OK if success, Z_MEM_ERROR if there was not
  //      enough memory, Z_BUF_ERROR if there was not enough room in the output
  //      buffer, or Z_DATA_ERROR if the input data was corrupted or incomplete.  In
  //      the case where there is not enough room, uncompress() will fill the output
  //      buffer with the uncompressed data up to that point.
  // */
}

