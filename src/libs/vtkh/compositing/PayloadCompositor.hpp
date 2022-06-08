#ifndef VTKH_PAYLOAD_COMPOSITOR_HPP
#define VTKH_PAYLOAD_COMPOSITOR_HPP

#include <sstream>
#include <vtkh/vtkh_exports.h>
#include <vtkh/compositing/PayloadImage.hpp>

namespace vtkh
{

class VTKH_API PayloadCompositor
{
public:
    PayloadCompositor();

    void ClearImages();

    void AddImage(PayloadImage &image);

    PayloadImage Composite();
protected:
    std::vector<PayloadImage>  m_images;
};

};

#endif


