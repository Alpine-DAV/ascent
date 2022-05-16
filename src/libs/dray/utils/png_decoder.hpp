#ifndef DRAY_PNG_DECODER_HPP
#define DRAY_PNG_DECODER_HPP

#include <string>

namespace dray
{

class PNGDecoder
{
public:
    PNGDecoder();
    ~PNGDecoder();
    // creates a buffer and the user is responsible
    // for freeing mem
    void decode(unsigned char *&rgba,
                int &width,
                int &height,
                const std::string &file_name);

    void decode_raw(unsigned char *&rgba,
                    int &width,
                    int &height,
                    const unsigned char *raw_png,
                    const size_t raw_size);
};

};

#endif


