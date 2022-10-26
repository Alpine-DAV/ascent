//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef APCOMP_PNG_ENCODER_HPP
#define APCOMP_PNG_ENCODER_HPP

#include <apcomp/apcomp_config.h>

#include <apcomp/apcomp_exports.h>
#include <string>

namespace apcomp
{

class APCOMP_API PNGEncoder
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

    void           Cleanup();

private:
    unsigned char *m_buffer;
    size_t         m_buffer_size;
};

} // namespace apcomp

#endif


