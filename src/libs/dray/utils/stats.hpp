#ifndef DRAY_STATS_HPP
#define DRAY_STATS_HPP

#include <dray/types.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>
#include <RAJA/RAJA.hpp>

namespace dray
{


template <typename T, int32 mult>
struct _MultiReduceSum : public _MultiReduceSum<T,mult-1>
{
  RAJA::ReduceSum<reduce_policy,T> m_count = RAJA::ReduceSum<reduce_policy,T>(0);

  DRAY_EXEC const RAJA::ReduceSum<reduce_policy,T> & operator[] (int ii) const
  {
    return *(&m_count - mult + 1 + ii);
  }

  DRAY_EXEC RAJA::ReduceSum<reduce_policy,T> & operator[] (int ii)
  {
    return *(&m_count - mult + 1 + ii);
  }

  void get(T sums[])
  {
    for (int ii = 0; ii < mult; ii++)
    {
      sums[ii] = static_cast<T>(operator[](ii).get());
    }
  }
};
template <typename T>
struct _MultiReduceSum<T,0> {};

template <typename T, int32 nbins>
struct HistogramSmall
{
  const T *m_sep;  //[nbins-1];   // User must initialize the separator values and this pointer to array T[nbins-1].
  _MultiReduceSum<int32,nbins> m_count;

  DRAY_EXEC void datum(T x, int32 w = 1) const
  {
    // Iterate over all bins until find the bin for x.
    int32 b = 0;
    while (b < nbins-1 && x > m_sep[b])
    {
      b++;
    }

    // Increment bin for x.
    m_count[b] += w;
  }

  void get(T sums[]) { m_count.get(sums); }

  void operator+= (const HistogramSmall &other)
  {
    for (int32 b = 0; b < nbins; b++)
    {
      m_count[b] += other.m_count[b].get();
    }
  }

  static void log(const T *sep, const int32 *counts)
  {
    DRAY_LOG_OPEN("histogram_small");
    int32 ii;
    for (ii = 0; ii < nbins - 1; ii++)
    {
      DRAY_LOG_ENTRY("hist_bin_count", counts[ii]);
      DRAY_LOG_ENTRY("hist_bin_r_sep", sep[ii]);
    }
    DRAY_LOG_ENTRY("hist_bin_count", counts[ii]);
    DRAY_LOG_CLOSE();
  }
};

template <int32 base, int32 power>
struct _NewtonSolveHistogram
{
  Array<int32> m_solves_sep;
  Array<int32> m_steps_sep;

  HistogramSmall<int32, power+3> m_num_solves;                    // Number of candidates needed for intersection. {key==-1} means never intersect.
  HistogramSmall<int32, power+3> m_num_useful_steps_success;      // Steps used on the successful intersecting candidate.
  HistogramSmall<int32, power+3> m_num_wasted_steps_success;      // Steps used on predecing candidates (but eventually intersect).
  HistogramSmall<int32, power+3> m_num_wasted_steps_failure;      // Total steps when never intersect.

  static _NewtonSolveHistogram factory()
  {
    // Fill separator arrays.
    int32 solves_sep[power+2];
    int32 steps_sep[power+2];
    solves_sep[0] = -1;
    steps_sep[0] = 0;
    int32 v = 1;
    for (int32 b = 1; b < power+2; b++)
    {
      solves_sep[b] = v;
      steps_sep[b] = v;
      v *= base;
    }

    _NewtonSolveHistogram ret;

    // Copy separator arrays to dray::Arrays.
    ret.m_solves_sep = Array<int32>(solves_sep, power+2);
    ret.m_steps_sep = Array<int32>(steps_sep, power+2);

    // Initialize pointers.
    ret.m_num_solves.m_sep = ret.m_solves_sep.get_device_ptr_const();
    ret.m_num_useful_steps_success.m_sep = ret.m_steps_sep.get_device_ptr_const();
    ret.m_num_wasted_steps_success.m_sep = ret.m_steps_sep.get_device_ptr_const();
    ret.m_num_wasted_steps_failure.m_sep = ret.m_steps_sep.get_device_ptr_const();

    return ret;
  }
};

using NewtonSolveHistogram = _NewtonSolveHistogram<2,5>;  // Rightmost finite bin is 2^5 == 32.


template <int32 base, int32 power>
struct _NewtonSolveCounter
{
  void add_candidate_steps(int32 steps)
  {
    m_accum_steps += m_last_candidate_steps;
    m_last_candidate_steps = steps;
    m_num_candidates++;
  }

  ////void finalize_search(bool did_succeed, const _NewtonSolveHistogram<base,power> &nsh)   // A success is counted as a "final" success.
  ////{
  ////  if (did_succeed)
  ////  {
  ////    nsh.m_num_solves.datum( m_num_candidates );
  ////    nsh.m_num_useful_steps_success.datum( m_last_candidate_steps );
  ////    nsh.m_num_wasted_steps_success.datum( m_accum_steps );
  ////  }
  ////  else
  ////  {
  ////    nsh.m_num_solves.datum( -1 );
  ////    nsh.m_num_wasted_steps_failure.datum( m_last_candidate_steps + m_accum_steps );
  ////  }
  ////}

  void finalize_search(bool did_succeed, int32 &r_wasted_steps, int32 &r_total_steps)   // A success is counted as a "final" success.
  {
    r_wasted_steps += m_accum_steps;
    if (!did_succeed)
    {
      r_wasted_steps += m_last_candidate_steps;
    }
    r_total_steps += m_accum_steps + m_last_candidate_steps;
  }

protected:
  int32 m_last_candidate_steps = 0;     // Internal states used for counting useful/wasted steps.
  int32 m_accum_steps = 0;              //
  int32 m_num_candidates = 0;
};

using NewtonSolveCounter = _NewtonSolveCounter<2,5>;  // Rightmost finite bin is 2^5 == 32.


////// //
////// // Stats - Class to accumulate statistical counters using RAJA.
////// //
////// template <typename T>   // T must have comparison operators defined.
////// struct Stats
////// {
//////   // abilities:
//////   // - min
//////   // - max
//////   //
//////   // - small_histogram     // Static arrays in registers -> multiple reduce
//////   // - mid_histogram       // External memory -> sweep and atomic
//////   // - large_histogram     // External memory -> binary search and atomic
//////
//////   // The first and last bins extend to their respective infinities.
//////   // The set of bins is defined by bin separators. The bin to the left of any
//////   // separator contains <= the separator value; the bin to the right of any
//////   // separator contains > the separator value.
//////   //
//////   // The separators array is read-only to this class (TODO take advantage of this for optimizations).
//////   // The hist array must be writable by this class.
//////
//////   static Stats factory_small_histogram(T min_val, T max_val,
//////
////// };

}  // namespace dray

#endif // DRAY_STATS_HPP
