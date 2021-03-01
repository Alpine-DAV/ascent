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
#include "ascent_field_array.hpp"
#include "ascent_raja_policies.hpp"
#include "ascent_math.hpp"

#include <ascent_logging.hpp>

#include <cstring>
#include <cmath>
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
field_dispatch(const conduit::Node &field, const Function &func)
{
  // check for single component scalar
  int num_children = field["values"].number_of_children();
  if(num_children > 1)
  {
    ASCENT_ERROR("Field Dispatch internal error: expected scalar array.");
  }
  conduit::Node res;

  // NO this is not the way
  if(field["values"].dtype().is_float32())
  {
    FieldArray<conduit::float32> farray(field);
    res = func(farray.ptr_const(), farray.size());
  }
  else if(field["values"].dtype().is_float64())
  {
    FieldArray<conduit::float64> farray(field);
    res = func(farray.ptr_const(), farray.size());
  }
  //}
  //else if(vals.dtype().is_int32())
  //{
  //  const conduit::int32 *ptr =  vals.as_int32_ptr();
  //  res = func(ptr, num_vals);
  //}
  //else if(vals.dtype().is_int64())
  //{
  //  const conduit::int64 *ptr =  vals.as_int64_ptr();
  //  res = func(ptr, num_vals);
  //}
  //else
  //{
  //  ASCENT_ERROR("Type dispatch: unsupported array type "<<
  //                values.schema().to_string());
  //}
  return res;
}

template<typename Function>
conduit::Node
type_dispatch(const conduit::Node &values, const Function &func)
{
  // check for single component scalar
  int num_children = values.number_of_children();
  if(num_children > 1)
  {
    ASCENT_ERROR("Internal error: expected scalar array.");
  }

  const conduit::Node &vals = num_children == 0 ? values : values.child(0);

  conduit::Node res;
  const int num_vals = vals.dtype().number_of_elements();

  if(vals.dtype().is_float32())
  {
    const conduit::float32 *ptr = MemoryAccessor::float32_ptr_const(vals);
    res = func(ptr, num_vals);
  }
  else if(vals.dtype().is_float64())
  {
    const conduit::float64 *ptr = MemoryAccessor::float64_ptr_const(vals);
    res = func(ptr, num_vals);
  }
  else if(vals.dtype().is_int32())
  {
    const conduit::int32 *ptr =  vals.as_int32_ptr();
    res = func(ptr, num_vals);
  }
  else if(vals.dtype().is_int64())
  {
    const conduit::int64 *ptr =  vals.as_int64_ptr();
    res = func(ptr, num_vals);
  }
  else
  {
    ASCENT_ERROR("Type dispatch: unsupported array type "<<
                  values.schema().to_string());
  }
  return res;
}

struct IndexLoc
{
  RAJA::Index_type idx;
  constexpr IndexLoc() : idx(-1) {}
  constexpr IndexLoc(RAJA::Index_type idx) : idx(idx) {}
};

struct MaxFunctor
{
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    T identity = std::numeric_limits<T>::lowest();


    RAJA::ReduceMaxLoc<reduce_policy, T, IndexLoc> reducer(identity, IndexLoc());

    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const T val = values[i];
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
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    T identity = std::numeric_limits<T>::max();

    RAJA::ReduceMinLoc<reduce_policy, T, IndexLoc> reducer(identity, IndexLoc());

    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const T val = values[i];
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
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    RAJA::ReduceSum<reduce_policy, T> sum(static_cast<T>(0));
    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {
      const T val = values[i];
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
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    RAJA::ReduceSum<reduce_policy, RAJA::Index_type> count(0);
    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const T value = values[i];
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
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    T sum = 0;
    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
    return res;
  }

  conduit::Node operator()(const float* values, const int &size) const
  {
    RAJA::ReduceSum<reduce_policy, RAJA::Index_type> count(0);
    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const float value = values[i];
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

  conduit::Node operator()(const double* values, const int &size) const
  {
    RAJA::ReduceSum<reduce_policy, RAJA::Index_type> count(0);
    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i) {

      const double value = values[i];
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

  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    const double inv_delta = double(m_num_bins) / (m_max_val - m_min_val);

    double *bins = new double[m_num_bins];
    memset(bins, 0, sizeof(double) * m_num_bins);

    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
    {
      double val = static_cast<double>(values[i]);
      int bin_index = static_cast<int>((val - m_min_val) * inv_delta);
      // clamp for now
      // another option is not to count data outside the range
      bin_index = max(0, min(bin_index, m_num_bins - 1));
      int old = RAJA::atomicAdd<atomic_policy> (&(bins[bin_index]), 1.);

    });
    ASCENT_ERROR_CHECK();
    conduit::Node res;
    res["value"].set(bins, m_num_bins);
    res["bin_size"] = (m_max_val - m_min_val) / double(m_num_bins);

    delete[] bins;
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

conduit::Node
field_array_max(const conduit::Node &field)
{
  return detail::field_dispatch(field, detail::MaxFunctor());
}

conduit::Node
array_min(const conduit::Node &values)
{
  return detail::type_dispatch(values, detail::MinFunctor());
}

conduit::Node
array_sum(const conduit::Node &values)
{
  return detail::type_dispatch(values, detail::SumFunctor());
}

conduit::Node
array_nan_count(const conduit::Node &values)
{
  return detail::type_dispatch(values, detail::NanFunctor());
}

conduit::Node
array_inf_count(const conduit::Node &values)
{
  return detail::type_dispatch(values, detail::InfFunctor());
}

conduit::Node
array_histogram(const conduit::Node &values,
                const double &min_value,
                const double &max_value,
                const int &num_bins)
{
  detail::HistogramFunctor histogram(min_value, max_value, num_bins);
  return detail::type_dispatch(values, histogram);
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





