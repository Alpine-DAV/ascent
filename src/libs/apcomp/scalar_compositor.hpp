//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_PAYLOAD_COMPOSITOR_HPP
#define APCOMP_PAYLOAD_COMPOSITOR_HPP

#include <apcomp/apcomp_config.h>
#include <sstream>
#include <apcomp/apcomp_exports.h>
#include <apcomp/scalar_image.hpp>

namespace apcomp
{

class APCOMP_API PayloadCompositor
{
public:
    PayloadCompositor();

    void ClearImages();

    void AddImage(ScalarImage &image);

    ScalarImage Composite();
protected:
    std::vector<ScalarImage>  m_images;
};

};

#endif


