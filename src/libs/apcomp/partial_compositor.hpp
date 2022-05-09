//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#define APCOMP_PARTIAL_COMPOSITOR_HPP
#define APCOMP_PARTIAL_COMPOSITOR_HPP

#include <vector>
#include <iostream>
#include <apcomp/apcomp_exports.h>
#include <apcomp/absorption_partial.hpp>
#include <apcomp/emission_partial.hpp>
#include <apcomp/volume_partial.hpp>


namespace apcomp {

template<typename PartialType>
class APCOMP_API PartialCompositor
{
public:
  PartialCompositor();
  ~PartialCompositor();
  void
  composite(std::vector<std::vector<PartialType>> &partial_images,
            std::vector<PartialType> &output_partials);
  void set_background(std::vector<float> &background_values);
  void set_background(std::vector<double> &background_values);
protected:
  void merge(const std::vector<std::vector<PartialType>> &in_partials,
             std::vector<PartialType> &partials,
             int &global_min_pixel,
             int &global_max_pixel);

  void composite_partials(std::vector<PartialType> &partials,
                          std::vector<PartialType> &output_partials);

  std::vector<typename PartialType::ValueType> m_background_values;
};

}; // namespace apcomp
#endif
