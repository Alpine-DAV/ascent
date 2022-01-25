//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_png_encoder_h
#define rover_png_encoder_h

#include <string>
// thirdparty includes
#include <vtkh/utils/PNGEncoder.hpp>

namespace rover {

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

  void           Encode(const double *rgba_in,
                        const int width,
                        const int height);

  void           EncodeChannel(const float *buffer_in,
                               const int width,
                               const int height);

  void           EncodeChannel(const double *buffer_in,
                               const int width,
                               const int height);

  void           Save(const std::string &filename);

private:
  vtkh::PNGEncoder m_encoder;
};

} // namespace rover

#endif
