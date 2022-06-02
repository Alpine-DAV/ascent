//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_png_compare.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_PNG_COMPARE_HPP
#define ASCENT_PNG_COMPARE_HPP

#include <string>
#include <conduit.hpp>
#include <ascent_exports.h>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class ASCENT_API PNGCompare
{
public:
    PNGCompare();
    ~PNGCompare();
    bool Compare(const std::string &img1,
                 const std::string &img2,
                 conduit::Node &info,
                 const float tolarance = 0.001f); // total pixel  diff tol
    /// int between 0-255
    /// is the tolerance between each rgba component
    void ColorTolerance(int color_tolerance);
private:
    void DiffImage(const unsigned char *buff_1,
                   const unsigned char *buff_2,
                   const int width,
                   const int height,
                   const std::string out_name);

    int m_color_tolerance;
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


