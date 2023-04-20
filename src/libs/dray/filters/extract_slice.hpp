#ifndef DRAY_EXTRACT_SLICE_HPP
#define DRAY_EXTRACT_SLICE_HPP

#include <dray/data_model/collection.hpp>
#include <dray/vec.hpp>

#include <vector>
#include <utility>

namespace dray
{

class ExtractSlice
{
protected:
  std::vector<Vec<Float, 3>> m_points;
  std::vector<Vec<Float, 3>> m_normals;
public:
  ExtractSlice();
  ~ExtractSlice();

  void add_plane(Vec<Float, 3> point, Vec<Float, 3> normal);
  void clear();

  std::pair<Collection, Collection> execute(Collection &collection);
};

}//namespace dray

#endif
