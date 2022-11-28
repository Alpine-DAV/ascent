//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_PNG_COMPARE_HPP
#define APCOMP_PNG_COMPARE_HPP

#include <apcomp/apcomp_config.h>
#include <string>
#include <apcomp/apcomp_exports.h>

namespace apcomp
{

class APCOMP_API PNGCompare
{
public:
    PNGCompare();
    ~PNGCompare();
    bool Compare(const std::string &img1,
                 const std::string &img2,
                 float &difference,
                 const float tolarance = 0.01f);
private:
    void DiffImage(const unsigned char *buff_1,
                   const unsigned char *buff_2,
                   const int width,
                   const int height,
                   const std::string out_name);
};

}

#endif


