//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_image_h
#define rover_image_h

#include <rover_config.h>
#include <vector>
#include <viskores/cont/ArrayHandle.h>
#include <viskores_typedefs.hpp>
#include <partial_image.hpp>

namespace rover
{

template<typename FloatType>
class Image
{
public:
  typedef viskores::cont::ArrayHandle<FloatType> HandleType;

  Image();
  Image(PartialImage<FloatType> &partial);

  HandleType get_intensity(const int &channel_num);
  HandleType get_optical_depth(const int &channel_num);
  int get_num_channels() const;
  bool has_intensity(const int &channel_num) const;
  bool has_optical_depth(const int &channel_num) const;
  void normalize_intensity(const int &channel_num);
  void normalize_intensity(const int &channel_num,
                           const float min_val,
                           const float max_val,
                           const bool log_scale);

  void normalize_optical_depth(const int &channel_num);
  void operator=(PartialImage<FloatType> partial);
  template<typename O> void operator=(Image<O> &other);
  HandleType flatten_intensity_values();
  HandleType flatten_optical_depth_values();
  template<typename T,
           typename O> friend void init_from_image(Image<T> &left,
                                                   Image<O> &right);

protected:
  std::vector<HandleType> m_intensity_values;
  std::vector<HandleType> m_optical_depth_values;

  void init_from_partial(PartialImage<FloatType> &);
  void normalize_handle(HandleType &, bool);
  void normalize_handle(HandleType &, bool, float, float, bool);
};
} // namespace rover
#endif
