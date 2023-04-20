#ifndef DRAY_EXTRACT_THREE_SLICE_HPP
#define DRAY_EXTRACT_THREE_SLICE_HPP

#include <dray/data_model/collection.hpp>
#include <dray/vec.hpp>

#include <utility>

namespace dray
{

class ExtractThreeSlice
{
protected:
  using VecType = Vec<Float, 3>;
  static const VecType N0;
  static const VecType N1;
  static const VecType N2;
  VecType m_point;
public:
  ExtractThreeSlice();
  ~ExtractThreeSlice();

  void set_point(VecType point);

  std::pair<Collection, Collection> execute(Collection &collection);
};

}//namespace dray

#endif
