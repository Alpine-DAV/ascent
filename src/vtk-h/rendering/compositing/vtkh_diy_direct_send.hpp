#ifndef VTKH_DIY_DIRECT_SEND_HPP
#define VTKH_DIY_DIRECT_SEND_HPP

#include <rendering/vtkh_image.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace vtkh 
{

class DirectSendCompositor
{
public:
  DirectSendCompositor();
  ~DirectSendCompositor();
  void CompositeVolume(diy::mpi::communicator &diy_comm, 
                       Image                  &image, 
                       const int *             vis_order,
                       const float *           bg_color); 
  std::string GetTimingString();
private:
  std::stringstream m_timing_log;
};

} // namespace vtkh
#endif
