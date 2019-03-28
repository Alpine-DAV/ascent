//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_conduit_reductions.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_conduit_reductions.hpp"

#include <ascent_logging.hpp>

#include <limits>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{

template<typename Function>
conduit::Node
type_dispatch(const conduit::Node &values, const Function &func)
{
  conduit::Node res;
  const int num_vals = values.dtype().number_of_elements();
  if(values.dtype().is_float32())
  {
    const conduit::float32 *ptr =  values.as_float32_ptr();
    res = func(ptr, num_vals);
  }
  else if(values.dtype().is_float64())
  {
    const conduit::float64 *ptr =  values.as_float64_ptr();
    res = func(ptr, num_vals);
  }
  else if(values.dtype().is_int32())
  {
    const conduit::int32 *ptr =  values.as_int32_ptr();
    res = func(ptr, num_vals);
  }
  else if(values.dtype().is_int64())
  {
    const conduit::int64 *ptr =  values.as_int64_ptr();
    res = func(ptr, num_vals);
  }
  else
  {
    ASCENT_ERROR("Type dispatch: unsupported array type");
  }
  return res;
}

struct MaxCompare
{
  double value;
  int index;
};

#ifdef ASCENT_USE_OPENMP
    #pragma omp declare reduction(maximum: struct MaxCompare : \
        omp_out = omp_in.value > omp_out.value ? omp_in : omp_out)
#endif

struct MaxFunctor
{
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    MaxCompare mcomp;

    mcomp.value = std::numeric_limits<double>::min();
    mcomp.index = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parrallel for reduction(maximum:mcomp)
#endif
    for(int v = 0; v < size; ++v)
    {
      double val = static_cast<double>(values[v]);
      if(val > mcomp.value)
      {
        mcomp.value = val;
        mcomp.index = v;
      }
    }
    conduit::Node res;
    res["value"] = mcomp.value;
    res["index"] = mcomp.index;
    return res;
  }
};
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail--
//-----------------------------------------------------------------------------

conduit::Node
array_max(const conduit::Node &values)
{
  return detail::type_dispatch(values, detail::MaxFunctor());
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





