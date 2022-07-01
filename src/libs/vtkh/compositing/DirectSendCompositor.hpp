#ifndef VTKH_DIY_DIRECT_SEND_HPP
#define VTKH_DIY_DIRECT_SEND_HPP

#include <vtkh/compositing/Image.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace vtkh
{

class DirectSendCompositor
{
public:
  DirectSendCompositor();
  ~DirectSendCompositor();
  void CompositeVolume(vtkhdiy::mpi::communicator &diy_comm,
                       std::vector<Image>     &images);
  std::string GetTimingString();
private:
  std::stringstream m_timing_log;
};

} // namespace vtkh
#endif
