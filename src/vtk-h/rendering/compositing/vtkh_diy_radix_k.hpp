#ifndef VTKH_DIY_RADIX_K_HPP
#define VTKH_DIY_RADIX_K_HPP

#include <rendering/vtkh_image.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace vtkh
{

class RadixKCompositor
{
public:
  RadixKCompositor();
  ~RadixKCompositor();
  void CompositeSurface(diy::mpi::communicator &diy_comm, Image &image); 
  std::string GetTimingString();
private:
  std::stringstream m_timing_log;
};

} // namspace vtkh

#endif
