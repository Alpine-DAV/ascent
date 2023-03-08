#ifndef VTKH_PNG_ENCODER_HPP
#define VTKH_PNG_ENCODER_HPP

#include <vtkh/vtkh_exports.h>
#include <string>
#include <vector>

namespace vtkh
{

class VTKH_API PNGEncoder
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
    void           Encode(const unsigned char *rgba_in,
                          const int width,
                          const int height,
                          const std::vector<std::string> &comments);
    void           Encode(const float *rgba_in,
                          const int width,
                          const int height,
                          const std::vector<std::string> &comments);
    void           Save(const std::string &filename);

    void          *PngBuffer();
    size_t         PngBufferSize();

    void           Cleanup();

private:
    unsigned char *m_buffer;
    size_t         m_buffer_size;
};

} // namespace vtkh

#endif


