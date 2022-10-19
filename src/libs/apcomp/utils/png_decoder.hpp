//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: png_decoder.hpp
///
//-----------------------------------------------------------------------------
#ifndef APCOMP_PNG_DECODER_HPP
#define APCOMP_PNG_DECODER_HPP

#include <apcomp/apcomp_config.h>

#include <string>

namespace apcomp
{

class PNGDecoder
{
public:
    PNGDecoder();
    ~PNGDecoder();
    // rgba
    void Decode(unsigned char *&rgba,
                int &width,
                int &height,
                const std::string &file_name);
};

} // namespace acpomp
#endif


