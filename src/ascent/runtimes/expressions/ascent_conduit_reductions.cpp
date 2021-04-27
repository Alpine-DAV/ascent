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
#include "ascent_memory_manager.hpp"
#include "ascent_memory_interface.hpp"
#include "ascent_array.hpp"
#include "ascent_raja_policies.hpp"
#include "ascent_execution.hpp"
#include "ascent_math.hpp"

#include <ascent_logging.hpp>

#include <cstring>
#include <cmath>
#include <limits>

#pragma diag_suppress 2527
#pragma diag_suppress 2529
#pragma diag_suppress 2651
#pragma diag_suppress 2653
#pragma diag_suppress 2668
#pragma diag_suppress 2669
#pragma diag_suppress 2670
#pragma diag_suppress 2671
#pragma diag_suppress 2735
#pragma diag_suppress 2737
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

bool field_is_float32(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_float32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_float32();
  }
}

bool field_is_float64(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_float64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_float64();
  }
}

bool field_is_int32(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_int32();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_int32();
  }
}

bool field_is_int64(const conduit::Node &field)
{
  const int children = field["values"].number_of_children();
  if(children == 0)
  {
    return field["values"].dtype().is_int64();
  }
  else
  {
    // there has to be one or more children so ask the first
    return field["values"].child(0).dtype().is_int64();
  }
}

template<typename Function, typename Exec>
conduit::Node dispatch_memory(const conduit::Node &field,
                              std::string component,
                              const Function &func,
                              const Exec &exec)
{
  const std::string mem_space = Exec::memory_space;

  conduit::Node res;
  if(field_is_float32(field))
  {
    MemoryInterface<conduit::float32> farray(field);
    MemoryAccessor<conduit::float32> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else if(field_is_float64(field))
  {
    MemoryInterface<conduit::float64> farray(field);
    MemoryAccessor<conduit::float64> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else if(field_is_int32(field))
  {
    MemoryInterface<conduit::int32> farray(field);
    MemoryAccessor<conduit::int32> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else if(field_is_int64(field))
  {
    MemoryInterface<conduit::int64> farray(field);
    MemoryAccessor<conduit::int64> accessor = farray.accessor(mem_space,component);
    res = func(accessor, exec);
  }
  else
  {
    ASCENT_ERROR("Type dispatch: unsupported array type "<<
                  field.schema().to_string());
  }
  return res;
}

template<typename Function>
conduit::Node
exec_dispatch(const conduit::Node &field, std::string component, const Function &func)
{

  conduit::Node res;
  const std::string exec_policy = ExecutionManager::execution();
  std::cout<<"Exec policy "<<exec_policy<<"\n";
  if(exec_policy == "serial")
  {
    SerialExec exec;
    res = dispatch_memory(field, component, func, exec);
  }
#if defined(ASCENT_USE_OPENMP)
  else if(exec_policy == "openmp")
  {
    OpenMPExec exec;
    res = dispatch_memory(field, component, func, exec);
  }
#endif
#ifdef ASCENT_USE_CUDA
  else if(exec_policy == "cuda")
  {
    CudaExec exec;
    res = dispatch_memory(field, component, func, exec);
  }
#endif
  else
  {
    ASCENT_ERROR("Execution dispatch: unsupported execution policy "<<
                  exec_policy);
  }
  return res;
}

template<typename Function>
conduit::Node
field_dispatch(const conduit::Node &field, const Function &func)
{
  // check for single component scalar
  int num_children = field["values"].number_of_children();
  if(num_children > 1)
  {
    ASCENT_ERROR("Field Dispatch internal error: expected scalar array.");
  }
  conduit::Node res;

  if(field_is_float32(field))
  {
    MemoryInterface<conduit::float32> farray(field);
    res = func(farray.ptr_const(), farray.size(0));
  }
  else if(field_is_float64(field))
  {
    MemoryInterface<conduit::float64> farray(field);
    res = func(farray.ptr_const(), farray.size(0));
  }
  else if(field_is_int32(field))
  {
    MemoryInterface<conduit::int32> farray(field);
    res = func(farray.ptr_const(), farray.size(0));
  }
  else if(field_is_int64(field))
  {
    MemoryInterface<conduit::int64> farray(field);
    res = func(farray.ptr_const(), farray.size(0));
  }
  else
  {
    ASCENT_ERROR("Type dispatch: unsupported array type "<<
                  field.schema().to_string());
  }
  return res;
}

struct IndexLoc
{
  RAJA::Index_type idx;
  constexpr IndexLoc() : idx(-1) {}
  constexpr __host__ __device__ IndexLoc(RAJA::Index_type idx) : idx(idx) {}
};

struct MaxFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const MemoryAccessor<T> accessor,
                           const Exec &) const
  {
    T identity = std::numeric_limits<T>::lowest();
    const int size = accessor.m_size;

    using fp = typename Exec::for_policy;
    using rp = typename Exec::reduce_policy;

    RAJA::ReduceMaxLoc<rp, T, IndexLoc> reducer(identity, IndexLoc());

    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
    {
      const T val = accessor[i];
      reducer.maxloc(val,i);
    });
    ASCENT_ERROR_CHECK();

    conduit::Node res;
    res["value"] = reducer.get();
    res["index"] = reducer.getLoc().idx;
    return res;
  }
};

struct MinFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const MemoryAccessor<T> accessor,
                           const Exec &) const
  {
    T identity = std::numeric_limits<T>::max();
    const int size = accessor.m_size;

    using fp = typename Exec::for_policy;
    using rp = typename Exec::reduce_policy;

    RAJA::ReduceMinLoc<rp, T, IndexLoc> reducer(identity, IndexLoc());

    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const T val = accessor[i];
      reducer.minloc(val,i);
    });
    ASCENT_ERROR_CHECK();

    conduit::Node res;
    res["value"] = reducer.get();
    res["index"] = reducer.getLoc().idx;
    return res;
  }
};

struct SumFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const MemoryAccessor<T> accessor,
                           const Exec &) const
  {
    const int size = accessor.m_size;
    using fp = typename Exec::for_policy;
    using rp = typename Exec::reduce_policy;

    RAJA::ReduceSum<rp, T> sum(static_cast<T>(0));
    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
    {
      const T val = accessor[i];
      sum += val;
    });
    ASCENT_ERROR_CHECK();

    conduit::Node res;
    res["value"] = sum.get();
    res["count"] = size;
    return res;
  }
};

struct NanFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const MemoryAccessor<T> accessor,
                           const Exec &) const
  {
    const int size = accessor.m_size;
    using fp = typename Exec::for_policy;
    using rp = typename Exec::reduce_policy;

    RAJA::ReduceSum<rp, RAJA::Index_type> count(0);
    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const T value = accessor[i];
      RAJA::Index_type is_nan = 0;
      if(value != value)
      {
        is_nan = 1;
      }
      count += is_nan;
    });
    ASCENT_ERROR_CHECK();

    conduit::Node res;
    res["value"] = count.get();
    res["count"] = size;
    return res;
  }

};

struct InfFunctor
{
  // default template for non floating point types
  template<typename T, typename Exec>
  conduit::Node operator()(const MemoryAccessor<T> accessor, Exec &) const
  {
    const int size = accessor.m_size;
    T sum = 0;
    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
    return res;
  }

  template<typename T, typename Exec>
  conduit::Node impl(const MemoryAccessor<T> accessor, Exec &) const
  {
    using fp = typename Exec::for_policy;
    using rp = typename Exec::reduce_policy;
    const int size = accessor.m_size;

    RAJA::ReduceSum<rp, RAJA::Index_type> count(0);

    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const T value = accessor[i];
      RAJA::Index_type is = 0;
      if(is_inf(value))
      {
        is = 1;
      }
      count += is;
    });
    ASCENT_ERROR_CHECK();

    conduit::Node res;
    res["value"] = count.get();
    res["count"] = size;
    return res;
  }

  template<typename Exec>
  conduit::Node operator()(const MemoryAccessor<float> accessor, Exec &exec) const
  {
    return impl(accessor, exec);
  }

  template<typename Exec>
  conduit::Node operator()(const MemoryAccessor<double> accessor, Exec &exec) const
  {
    return impl(accessor, exec);
  }
};

struct HistogramFunctor
{
  double m_min_val;
  double m_max_val;
  int m_num_bins;
  HistogramFunctor(const double &min_val,
                   const double &max_val,
                   const int &num_bins)
    : m_min_val(min_val),
      m_max_val(max_val),
      m_num_bins(num_bins)
  {}

  template<typename T, typename Exec>
  conduit::Node operator()(const MemoryAccessor<T> accessor,
                           const Exec &) const
  {
    const int size = accessor.m_size;
    const double inv_delta = double(m_num_bins) / (m_max_val - m_min_val);
    // need to avoid capturing 'this'
    const int num_bins = m_num_bins;
    const double min_val = m_min_val;

    // conduit zero initializes this array
    conduit::Node res;
    res["value"].set(conduit::DataType::float64(num_bins));
    double *nb = res["value"].value();

    Array<double> bins(nb, num_bins);

    double *bins_ptr = bins.ptr(Exec::memory_space);

    using fp = typename Exec::for_policy;
    using ap = typename Exec::atomic_policy;

    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
    {
      double val = static_cast<double>(accessor[i]);
      int bin_index = static_cast<int>((val - min_val) * inv_delta);
      // clamp for now
      // another option is not to count data outside the range
      bin_index = max(0, min(bin_index, num_bins - 1));
      int old = RAJA::atomicAdd<ap> (&(bins_ptr[bin_index]), 1.);

    });
    ASCENT_ERROR_CHECK();

    // synch the values back to the host
    (void)  bins.host_ptr();
    res["bin_size"] = (m_max_val - m_min_val) / double(m_num_bins);

    return res;
  }
};
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail--
//-----------------------------------------------------------------------------

conduit::Node
field_reduction_max(const conduit::Node &field, std::string component)
{
  return detail::exec_dispatch(field, component, detail::MaxFunctor());
}

conduit::Node
field_reduction_min(const conduit::Node &field, const std::string component)
{
  return detail::exec_dispatch(field, component, detail::MinFunctor());
}

conduit::Node
field_reduction_sum(const conduit::Node &field, const std::string component)
{
  return detail::exec_dispatch(field, component, detail::SumFunctor());
}

conduit::Node
field_reduction_nan_count(const conduit::Node &field, std::string component)
{
  return detail::exec_dispatch(field, component, detail::NanFunctor());
}

conduit::Node
field_reduction_inf_count(const conduit::Node &field, std::string component)
{
  return detail::exec_dispatch(field, component, detail::InfFunctor());
}

conduit::Node
field_reduction_histogram(const conduit::Node &field,
                const double &min_value,
                const double &max_value,
                const int &num_bins,
                std::string component)
{
  detail::HistogramFunctor histogram(min_value, max_value, num_bins);
  return detail::exec_dispatch(field, component, histogram);
}

conduit::Node
array_max(const conduit::Node &array, const std::string exec_loc, std::string component)
{
  // keep the original so we can set it back
  const std::string orig = ExecutionManager::execution();
  ExecutionManager::execution(exec_loc);

  conduit::Node fake_field;
  fake_field["values"].set_external(array);

  conduit::Node res = field_reduction_max(fake_field, component);
  // restore the original exectution env
  ExecutionManager::execution(orig);
  return res;
}

conduit::Node
array_min(const conduit::Node &array, const std::string exec_loc, std::string component)
{
  // keep the original so we can set it back
  const std::string orig = ExecutionManager::execution();
  ExecutionManager::execution(exec_loc);

  conduit::Node fake_field;
  fake_field["values"].set_external(array);

  conduit::Node res = field_reduction_min(fake_field, component);
  // restore the original exectution env
  ExecutionManager::execution(orig);
  return res;
}

conduit::Node
array_sum(const conduit::Node &array, const std::string exec_loc, std::string component)
{
  // keep the original so we can set it back
  const std::string orig = ExecutionManager::execution();
  ExecutionManager::execution(exec_loc);

  conduit::Node fake_field;
  fake_field["values"].set_external(array);

  conduit::Node res = field_reduction_sum(fake_field, component);
  // restore the original exectution env
  ExecutionManager::execution(orig);
  return res;
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
