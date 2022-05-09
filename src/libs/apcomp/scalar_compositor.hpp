#ifndef APCOMP_PAYLOAD_COMPOSITOR_HPP
#define APCOMP_PAYLOAD_COMPOSITOR_HPP

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


