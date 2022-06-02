#ifndef DRAY_PNG_ENCODER_HPP
#define DRAY_PNG_ENCODER_HPP

#include <dray/types.hpp>
#include <string>

namespace dray
{

class PNGEncoder
{
public:
    PNGEncoder();
    ~PNGEncoder();

    void           encode(const uint8 *rgba_in,
                          const int32 width,
                          const int32 height);

    void           encode(const float32 *rgba_in,
                          const int32 width,
                          const int32 height);

    void           save(const std::string &filename);

    void          *png_buffer();
    size_t         png_buffer_size();

    void           cleanup();

private:
    unsigned char *m_buffer;
    size_t         m_buffer_size;
};

};

#endif


