//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_DIY_DIRECT_SEND_HPP
#define APCOMP_DIY_DIRECT_SEND_HPP

#include <apcomp/apcomp_config.h>
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
