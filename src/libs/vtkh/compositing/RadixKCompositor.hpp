#ifndef VTKH_DIY_RADIX_K_HPP
#define VTKH_DIY_RADIX_K_HPP

#include <vtkh/compositing/Image.hpp>
#include <vtkh/compositing/PayloadImage.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace vtkh
{

class RadixKCompositor
{
public:
  RadixKCompositor();
  ~RadixKCompositor();
  void CompositeSurface(vtkhdiy::mpi::communicator &diy_comm, Image &image);
  void CompositeSurface(vtkhdiy::mpi::communicator &diy_comm, PayloadImage &image);

  template<typename ImageType>
  void CompositeImpl(vtkhdiy::mpi::communicator &diy_comm, ImageType &image);

  std::string GetTimingString();
private:
  std::stringstream m_timing_log;
};

} // namspace vtkh

#endif
