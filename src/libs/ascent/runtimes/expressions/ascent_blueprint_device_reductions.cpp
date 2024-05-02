//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_blueprint_device_reductions.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_blueprint_device_reductions.hpp"
#include "ascent_memory_manager.hpp"
#include "ascent_blueprint_device_dispatch.hpp"
#include "ascent_blueprint_device_mesh_objects.hpp"
#include "ascent_array.hpp"
#include "ascent_execution_policies.hpp"
#include "ascent_execution_manager.hpp"
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

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::expressions::detail --
//-----------------------------------------------------------------------------
namespace detail
{

struct MaxFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const DeviceAccessor<T> accessor,
                           const Exec &) const
  {
    T identity = std::numeric_limits<T>::lowest();
    const int size = accessor.m_size;

    using for_policy    = typename Exec::for_policy;
    using reduce_policy = typename Exec::reduce_policy;

    ascent::ReduceMaxLoc<reduce_policy,T> reducer(identity,-1);
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
      const T val = accessor[i];
      reducer.maxloc(val,i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    conduit::Node res;
    res["value"] = reducer.get();
    res["index"] = reducer.getLoc();
    return res;
  }
};

struct MinFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const DeviceAccessor<T> accessor,
                           const Exec &) const
  {
    T identity = std::numeric_limits<T>::max();
    const int size = accessor.m_size;

    using for_policy = typename Exec::for_policy;
    using reduce_policy = typename Exec::reduce_policy;

    ascent::ReduceMinLoc<reduce_policy,T> reducer(identity,-1);
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {

      const T val = accessor[i];
      reducer.minloc(val,i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    conduit::Node res;
    res["value"] = reducer.get();
    res["index"] = reducer.getLoc();
    return res;
  }
};

struct SumFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const DeviceAccessor<T> accessor,
                           const Exec &) const
  {
    const int size = accessor.m_size;
    using for_policy = typename Exec::for_policy;
    using reduce_policy = typename Exec::reduce_policy;

    ascent::ReduceSum<reduce_policy,T> sum(static_cast<T>(0));
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
      const T val = accessor[i];
      sum += val;

    });
    ASCENT_DEVICE_ERROR_CHECK();

    conduit::Node res;
    res["value"] = sum.get();
    res["count"] = size;
    return res;
  }
};

struct DFAddFunctor
{
    template<typename T, typename Exec>
    conduit::Node operator()(const DeviceAccessor<T> l_accessor,
                             const DeviceAccessor<T> r_accessor,
                             const Exec &) const
    {

    const int l_size = l_accessor.m_size;
    const int r_size = r_accessor.m_size;

    bool diff_sizes = false;
    int size; 
    int max_size;

    size = max_size = l_size; 
    if(l_size != r_size)
    {
        size = min(l_size, r_size);
        max_size = max(l_size, r_size);
        diff_sizes = true;
    }


    // conduit zero initializes this array
    conduit::Node res;
    res["values"].set(conduit::DataType::float64(max_size));
    double *res_array = res["values"].value();

    Array<double> field_sums(res_array, max_size);
    double *sums_ptr = field_sums.get_ptr(Exec::memory_space);

    using for_policy = typename Exec::for_policy;
    using atomic_policy = typename Exec::atomic_policy;

    // init device array
    ascent::forall<for_policy>(0, max_size, [=] ASCENT_LAMBDA(index_t i)
    {
        sums_ptr[i]=0.0;
    });
    ASCENT_DEVICE_ERROR_CHECK();

    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
      const double val = l_accessor[i] + r_accessor[i];
      //sums_ptr[i] = val;
      int old = ascent::atomic_add<atomic_policy>(&(sums_ptr[i]), val);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    if(diff_sizes)
    {
      if(l_size > r_size)
      {
        ascent::forall<for_policy>(size, l_size, [=] ASCENT_LAMBDA(index_t i)
        {
          const T val = l_accessor[i];
          sums_ptr[i] = val;
        });
        ASCENT_DEVICE_ERROR_CHECK();
      }
      else
      {
        ascent::forall<for_policy>(size, r_size, [=] ASCENT_LAMBDA(index_t i)
        {
          const T val = r_accessor[i];
          sums_ptr[i] = val;
        });
        ASCENT_DEVICE_ERROR_CHECK();
      }
    }

    // synch the values back to the host
    (void) field_sums.get_host_ptr();

    return res;
    }
};

struct DFPowerFunctor
{
    template<typename T, typename Exec>
    conduit::Node operator()(const DeviceAccessor<T> f_accessor,
                             const double &exponent,
                             const Exec &) const
    {

    const int size = f_accessor.m_size;

    // conduit zero initializes this array
    conduit::Node res;
    res["values"].set(conduit::DataType::float64(size));
    double *res_array = res["values"].value();

    Array<double> power_of_field(res_array, size);
    double *powers_ptr = power_of_field.get_ptr(Exec::memory_space);

    using for_policy = typename Exec::for_policy;
    using atomic_policy = typename Exec::atomic_policy;

    // init device array
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        powers_ptr[i]=0.0;
    });
    ASCENT_DEVICE_ERROR_CHECK();

    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
      const double val = std::pow(f_accessor[i], exponent);
      //powers_ptr[i] = val;
      int old = ascent::atomic_add<atomic_policy>(&(powers_ptr[i]), val);
      //if(f_accessor[i] < 0)
      //{
      //  std::cerr << "field val: " << f_accessor[i] << "^" << exponent << " = " << val << std::endl;
      //  std::cerr << "field val: " << f_accessor[i] << "^" << exponent << " = " << powers_ptr[i] << std::endl;
      //}
    });
    ASCENT_DEVICE_ERROR_CHECK();

    // synch the values back to the host
    (void) power_of_field.get_host_ptr();

    return res;
    }
};

struct NanFunctor
{
  template<typename T, typename Exec>
  conduit::Node operator()(const DeviceAccessor<T> accessor,
                           const Exec &) const
  {
    const int size = accessor.m_size;
    using for_policy = typename Exec::for_policy;
    using reduce_policy = typename Exec::reduce_policy;

    ascent::ReduceSum<reduce_policy,index_t> count(0);
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {

      const T value = accessor[i];
      index_t is_nan = 0;
      if(value != value)
      {
        is_nan = 1;
      }
      count += is_nan;
    });
    ASCENT_DEVICE_ERROR_CHECK();

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
  conduit::Node operator()(const DeviceAccessor<T> accessor, Exec &) const
  {
    const int size = accessor.m_size;
    T sum = 0;
    conduit::Node res;
    res["value"] = sum;
    res["count"] = (int)size;
    return res;
  }

  template<typename T, typename Exec>
  conduit::Node impl(const DeviceAccessor<T> accessor, Exec &) const
  {
    using for_policy = typename Exec::for_policy;
    using reduce_policy = typename Exec::reduce_policy;
    const int size = accessor.m_size;

    ascent::ReduceSum<reduce_policy,index_t> count(0);
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {

      const T value = accessor[i];
      index_t is = 0;
      if(is_inf(value))
      {
        is = 1;
      }
      count += is;

    });
    ASCENT_DEVICE_ERROR_CHECK();

    conduit::Node res;
    res["value"] = count.get();
    res["count"] = size;
    return res;
  }

  template<typename Exec>
  conduit::Node operator()(const DeviceAccessor<float> accessor, Exec &exec) const
  {
    return impl(accessor, exec);
  }

  template<typename Exec>
  conduit::Node operator()(const DeviceAccessor<double> accessor, Exec &exec) const
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
  conduit::Node operator()(const DeviceAccessor<T> accessor,
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

    double *bins_ptr = bins.get_ptr(Exec::memory_space);

    using for_policy    = typename Exec::for_policy;
    using atomic_policy = typename Exec::atomic_policy;

    // init device array
    ascent::forall<for_policy>(0, num_bins, [=] ASCENT_LAMBDA(index_t i)
    {
      bins_ptr[i]=0.0;
    });
    ASCENT_DEVICE_ERROR_CHECK();

    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
      double val = static_cast<double>(accessor[i]);
      int bin_index = static_cast<int>((val - min_val) * inv_delta);
      // clamp for now
      // another option is not to count data outside the range
      bin_index = max(0, min(bin_index, num_bins - 1));
      int old = ascent::atomic_add<atomic_policy>(&(bins_ptr[bin_index]), 1.);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    // synch the values back to the host
    (void)  bins.get_host_ptr();
    res["bin_size"] = (m_max_val - m_min_val) / double(m_num_bins);

    return res;
  }
};

////////////////////////////////////////////////////////////////////////////////////
// TODO THIS NEEDS TO BE RAJAFIED
struct HistoryGradientRangeFunctor
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
    #ifdef ASCENT_OPENMP_ENABLED
        #pragma omp parallel for
    #endif
        for(int v = 0; v < num_gradients; ++v)
        {
            gradients[v] = ( (y_values[v+1] - y_values[v]) / (double) *dx_values);
        }
    }
    else {
    #ifdef ASCENT_OPENMP_ENABLED
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


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions::detail--
//-----------------------------------------------------------------------------

conduit::Node
history_gradient_range(const conduit::Node &y_values,
                       const conduit::Node &dx_values)
{
  // TODO THIS NEEDS TO BE PORTED TO RAJA ?
  return exec_dispatch_two_leaves(y_values, dx_values, detail::HistoryGradientRangeFunctor());
}


conduit::Node
field_reduction_max(const conduit::Node &field, const std::string &component)
{
  return exec_dispatch_mcarray_component(field["values"], component, detail::MaxFunctor());
}

conduit::Node
field_reduction_min(const conduit::Node &field, const std::string &component)
{
  return exec_dispatch_mcarray_component(field["values"], component, detail::MinFunctor());
}

conduit::Node
field_reduction_sum(const conduit::Node &field, const std::string &component)
{
  return exec_dispatch_mcarray_component(field["values"], component, detail::SumFunctor());
}

conduit::Node
field_reduction_nan_count(const conduit::Node &field, const std::string &component)
{
  return exec_dispatch_mcarray_component(field["values"], component, detail::NanFunctor());
}

conduit::Node
field_reduction_inf_count(const conduit::Node &field, const std::string &component)
{
  return exec_dispatch_mcarray_component(field["values"], component, detail::InfFunctor());
}

conduit::Node
field_reduction_histogram(const conduit::Node &field,
                          const double &min_value,
                          const double &max_value,
                          const int &num_bins,
                          const std::string &component)
{
  detail::HistogramFunctor histogram(min_value, max_value, num_bins);
  return exec_dispatch_mcarray_component(field["values"], component, histogram);
}

conduit::Node
array_max(const conduit::Node &array,
          const std::string &exec_loc,
          const std::string &component)
{
  // keep the original so we can set it back
  const std::string orig = ExecutionManager::execution_policy();
  ExecutionManager::set_execution_policy(exec_loc);

  conduit::Node fake_field;
  fake_field["values"].set_external(array);

  conduit::Node res = field_reduction_max(fake_field, component);
  // restore the original execution env
  ExecutionManager::set_execution_policy(orig);

  return res;
}

conduit::Node
array_min(const conduit::Node &array,
          const std::string &exec_loc,
          const std::string &component)
{
  // keep the original so we can set it back
  const std::string orig = ExecutionManager::execution_policy();
  ExecutionManager::set_execution_policy(exec_loc);

  conduit::Node fake_field;
  fake_field["values"].set_external(array);

  conduit::Node res = field_reduction_min(fake_field, component);
  // restore the original execution env
  ExecutionManager::set_execution_policy(orig);
  return res;
}

conduit::Node
array_sum(const conduit::Node &array,
          const std::string &exec_loc,
          const std::string &component)
{
  // keep the original so we can set it back
  const std::string orig = ExecutionManager::execution_policy();
  ExecutionManager::set_execution_policy(exec_loc);

  conduit::Node fake_field;
  fake_field["values"].set_external(array);

  conduit::Node res = field_reduction_sum(fake_field, component);
  // restore the original execution env
  ExecutionManager::set_execution_policy(orig);

  return res;
}

conduit::Node
derived_field_binary_add(const conduit::Node &l_field,
                         const conduit::Node &r_field,
                         const std::string &component)
{
  return exec_dispatch_binary_df(l_field,
                                 r_field,
                                 component,
                                 detail::DFAddFunctor());
}

conduit::Node
derived_field_power(const conduit::Node &field,
                    const double &exponent,
                    const std::string &component)
{
  return exec_dispatch_unary_df(field,
                                exponent,
                                component,
                                detail::DFPowerFunctor());
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

