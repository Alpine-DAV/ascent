#ifndef DIY_COMMUNICATOR_HPP
#define DIY_COMMUNICATOR_HPP

#warning "diy::Communicator (in diy/communicator.hpp) is deprecated, use diy::mpi::communicator directly"

#include "mpi.hpp"

namespace vtkhdiy
{
  typedef mpi::communicator         Communicator;
}

#endif
