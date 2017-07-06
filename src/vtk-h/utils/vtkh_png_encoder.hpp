#ifndef VTKH_PNG_ENCODER_HPP
#define VTKH_PNG_ENCODER_HPP

#include <string>

namespace vtkh 
{

class PNGEncoder
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

} // namespace vtkh

#endif


