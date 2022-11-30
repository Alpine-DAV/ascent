#ifndef APCOMP_DIY_RADIX_K_HPP
#define APCOMP_DIY_RADIX_K_HPP

#include <apcomp/apcomp_config.h>
#include <apcomp/image.hpp>
#include <apcomp/scalar_image.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace apcomp
{

class RadixKCompositor
{
public:
  RadixKCompositor();
  ~RadixKCompositor();
  void CompositeSurface(apcompdiy::mpi::communicator &diy_comm, Image &image);
  void CompositeSurface(apcompdiy::mpi::communicator &diy_comm, ScalarImage &image);

  template<typename ImageType>
  void CompositeImpl(apcompdiy::mpi::communicator &diy_comm, ImageType &image);

  std::string GetTimingString();
private:
  std::stringstream m_timing_log;
};

} // namspace apcomp

#endif
