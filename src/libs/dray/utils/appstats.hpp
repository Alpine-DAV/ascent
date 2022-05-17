#ifndef DRAY_APPSTATS_HPP
#define DRAY_APPSTATS_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>
#include <dray/ray.hpp>
#include <dray/types.hpp>

#include <iostream>
#include <vector>
#include <utility>

namespace dray
{
namespace stats
{

struct Stats
{
#ifdef DRAY_STATS
  int32 m_newton_iters; // total newton iterations
  int32 m_candidates;   // number of candidates testes
  int32 m_found;        // found (1) or not (0)

  void DRAY_EXEC construct()
  {
    m_newton_iters = 0;
    m_candidates = 0;
    m_found = 0;
  }

  void DRAY_EXEC acc_iters(const int32 &iters)
  {
    m_newton_iters += iters;
  }

  void DRAY_EXEC acc_candidates(const int32 &candidates)
  {
    m_candidates += candidates;
  }

  void DRAY_EXEC found()
  {
    m_found = true;
  }

  int32 DRAY_EXEC iters()
  {
    return m_newton_iters;
  }

#else
  // we do nothing
  void DRAY_EXEC construct() { }
  void DRAY_EXEC acc_iters(const int32&) { }
  void DRAY_EXEC acc_candidates(const int32&) { }
  void DRAY_EXEC found() { }
  int32 DRAY_EXEC iters() { return 0; }
#endif

  friend std::ostream& operator<<(std::ostream &os, const Stats &stats_struct);

};
// When stats are not enabled, calls to this class are no-ops
class StatStore
{
protected:
  static std::vector<std::vector<std::pair<int32,Stats>>> m_ray_stats;
  static std::vector<std::vector<std::pair<Vec<float32,3>,Stats>>> m_point_stats;
public:
  static void add_ray_stats(const Array<Ray> &rays, Array<Stats> &stats);

  static void add_point_stats(Array<Vec<Float,3>> &points, Array<Stats> &stats);

  static void write_ray_stats(const int32 width,const int32 height);
  static void write_point_stats(const std::string name);
  static void clear();

};

} // namespace stats
} // namespace dray

#endif//DRAY_APPSTATS_HPP
