#ifndef APCOMP_DIY_DIRECT_SEND_HPP
#define APCOMP_DIY_DIRECT_SEND_HPP

#include <apcomp/image.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace apcomp
{

class DirectSendCompositor
{
public:
  DirectSendCompositor();
  ~DirectSendCompositor();
  void CompositeVolume(apcompdiy::mpi::communicator &diy_comm,
                       std::vector<Image>     &images);
  std::string GetTimingString();
private:
  std::stringstream m_timing_log;
};

} // namespace apcomp
#endif
