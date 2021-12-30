//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_conduit_reductions.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_conduit_reductions.hpp"

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
    const conduit::float32 *ptr =  vals.as_float32_ptr();
    res = func(ptr, num_vals);
  }
  else if(vals.dtype().is_float64())
  {
    const conduit::float64 *ptr =  vals.as_float64_ptr();
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

////////////////////////////////////////////////////////////////////////////////////

template<typename Function>
conduit::Node
type_dispatch(const conduit::Node &values0, const conduit::Node &values1, const bool is_list, const Function &func)
{
  // check for single component scalar
  int num_children0 = values0.number_of_children();
  int num_children1 = values1.number_of_children();
  if(num_children0 > 1 || num_children1 > 1)
  {
    ASCENT_ERROR("Internal error: expected scalar array.");
  }
  const conduit::Node &vals0 = num_children0 == 0 ? values0 : values0.child(0);
  const conduit::Node &vals1 = num_children1 == 0 ? values1 : values1.child(0);

  conduit::Node res;
  const int num_vals0 = vals0.dtype().number_of_elements();
  const int num_vals1 = vals1.dtype().number_of_elements();

  if(vals0.dtype().is_float32())
  {
    const conduit::float32 *ptr0 =  vals0.as_float32_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64() || is_list) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("Type dispatch: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else if(vals0.dtype().is_float64())
  {
    const conduit::float64 *ptr0 =  vals0.as_float64_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64() || is_list) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("Type dispatch: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else if(vals0.dtype().is_int32())
  {
    const conduit::int32 *ptr0 =  vals0.as_int32_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64() || is_list) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("Type dispatch: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else if(vals0.dtype().is_int64())
  {
    const conduit::int64 *ptr0 =  vals0.as_int64_ptr();
    if(vals1.dtype().is_float32()) {
      const conduit::float32 *ptr1 =  vals1.as_float32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_float64() || is_list) {
      const conduit::float64 *ptr1 =  vals1.as_float64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int32()) {
      const conduit::int32 *ptr1 =  vals1.as_int32_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else if(vals1.dtype().is_int64()) {
      const conduit::int64 *ptr1 =  vals1.as_int64_ptr();
      res = func(ptr0, ptr1, num_vals0, num_vals1);
    }
    else {
      ASCENT_ERROR("Type dispatch: unsupported array type for array1: "<< values1.schema().to_string());
    }
  }
  else
  {
    ASCENT_ERROR("Type dispatch: unsupported array type for array0: "<<
                  values0.schema().to_string());
  }
  return res;
}

struct GradientFunctor
{
  template<typename T, typename T2>
  conduit::Node operator()(const T* y_values, const T2* dx_values, const int &size_y_values, const int &size_dx_values) const
  {
    bool single_dx = (size_dx_values == 1);

    if(!single_dx && size_dx_values < (size_y_values-1)) {
        ASCENT_ERROR("Must either supply a single uniform delta_x value, or provide at least len(y_values)-1 delta_x values (indicating the delta_x from each y value to the next).");                
    }

    int num_gradients = size_y_values-1;
    double *gradients = new double[num_gradients];

    if(single_dx) {    
    #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for
    #endif
        for(int v = 0; v < num_gradients; ++v)
        {
            gradients[v] = ( (y_values[v+1] - y_values[v]) / (double) *dx_values);
        }
    }
    else {
    #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for
    #endif
        for(int v = 0; v < num_gradients; ++v)
        {
            gradients[v] = ( (y_values[v+1] - y_values[v]) / (double) dx_values[v]);
        }
    }

    conduit::Node res;
    res["value"].set(gradients, num_gradients);
    res["count"] = num_gradients;
    return res;
  }
};

////////////////////////////////////////////////////////////////////////////////////


struct MaxCompare
{
  double value;
  int index;
};

#ifdef ASCENT_USE_OPENMP
    #pragma omp declare reduction(maximum: struct MaxCompare : \
        omp_out = omp_in.value > omp_out.value ? omp_in : omp_out) \
        initializer(omp_priv={std::numeric_limits<double>::lowest(),0})
#endif

struct MaxFunctor
{
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    MaxCompare mcomp;

    mcomp.value = std::numeric_limits<double>::lowest();
    mcomp.index = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(maximum:mcomp)
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

struct MinCompare
{
  double value;
  int index;
};

#ifdef ASCENT_USE_OPENMP
    #pragma omp declare reduction(minimum: struct MinCompare : \
        omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) \
        initializer(omp_priv={std::numeric_limits<double>::max(),0})
#endif

struct MinFunctor
{
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    MinCompare mcomp;

    mcomp.value = std::numeric_limits<double>::max();
    mcomp.index = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(minimum:mcomp)
#endif
    for(int v = 0; v < size; ++v)
    {
      double val = static_cast<double>(values[v]);
      if(val < mcomp.value)
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

struct SumFunctor
{
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    T sum = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:sum)
#endif
    for(int v = 0; v < size; ++v)
    {
      double val = static_cast<double>(values[v]);
      sum += val;
    }
    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
    return res;
  }
};

struct NanFunctor
{
  template<typename T>
  conduit::Node operator()(const T* values, const int &size) const
  {
    T sum = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:sum)
#endif
    for(int v = 0; v < size; ++v)
    {
      T is_nan = T(0);
      const T value = values[v];
      if(value != value)
      {
        is_nan = T(1);
      }
      sum += is_nan;
    }

    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
    return res;
  }
};

struct InfFunctor
{
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
    double sum = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:sum)
#endif
    for(int v = 0; v < size; ++v)
    {
      sum += std::isinf(values[v]) ? 1. : 0.;
    }

    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
    return res;
  }

  conduit::Node operator()(const double* values, const int &size) const
  {
    double sum = 0;
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:sum)
#endif
    for(int v = 0; v < size; ++v)
    {
      sum += std::isinf(values[v]) ? 1. : 0.;
    }

    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
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
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for
#endif
    for(int v = 0; v < size; ++v)
    {
      double val = static_cast<double>(values[v]);
      int bin_index = static_cast<int>((val - m_min_val) * inv_delta);
      // clamp for now
      // another option is not to count data outside the range
      bin_index = std::max(0, std::min(bin_index, m_num_bins - 1));
#ifdef ASCENT_USE_OPENMP
      #pragma omp atomic
#endif
      bins[bin_index]++;

    }
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
array_gradient(const conduit::Node &y_values, const conduit::Node &dx_values, const bool is_list)
{
  return detail::type_dispatch(y_values, dx_values, is_list, detail::GradientFunctor());
}

conduit::Node
array_max(const conduit::Node &values)
{
  return detail::type_dispatch(values, detail::MaxFunctor());
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






