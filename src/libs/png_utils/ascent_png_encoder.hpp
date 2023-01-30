//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_png_encoder.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_PNG_ENCODER_HPP
#define ASCENT_PNG_ENCODER_HPP

#include <png_utils/ascent_png_utils_exports.h>

#include <conduit.hpp>
#include <string>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class ASCENT_API PNGEncoder
{
public:
    PNGEncoder();
    ~PNGEncoder();

    void           Encode(const unsigned char *rgba_in,
                          const int width,
                          const int height);
    void           Encode(const float *rgba_in,
                          const int width,
                          const int height);
    void           Save(const std::string &filename);

    void          *PngBuffer();
    size_t         PngBufferSize();

    void           Base64Encode();
    conduit::Node &Base64Node();

    void           Cleanup();

private:
    unsigned char *m_buffer;
    size_t         m_buffer_size;
    conduit::Node  m_base64_data;
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


