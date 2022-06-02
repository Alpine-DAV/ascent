#ifndef DRAY_PNG_COMPARE_HPP
#define DRAY_PNG_COMPARE_HPP

#include <string>
#include <conduit.hpp>

namespace dray
{

class PNGCompare
{
public:
    PNGCompare();
    ~PNGCompare();
    bool compare(const std::string &img1,
                 const std::string &img2,
                 conduit::Node &info,
                 const float tolarance = 0.001f);
private:
    void diff_image(const unsigned char *buff_1,
                    const unsigned char *buff_2,
                    const int width,
                    const int height,
                    const std::string out_name);
};

};

#endif


