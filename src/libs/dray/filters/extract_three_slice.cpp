#include "extract_three_slice.hpp"

#include <dray/filters/extract_slice.hpp>

namespace dray
{

const ExtractThreeSlice::VecType ExtractThreeSlice::N0 = {1.f, 0.f, 0.f};
const ExtractThreeSlice::VecType ExtractThreeSlice::N1 = {0.f, 1.f, 0.f};
const ExtractThreeSlice::VecType ExtractThreeSlice::N2 = {0.f, 0.f, 1.f};

ExtractThreeSlice::ExtractThreeSlice()
  : m_point({0.f, 0.f, 0.f})
{
}

ExtractThreeSlice::~ExtractThreeSlice()
{
}

void
ExtractThreeSlice::set_point(Vec<Float, 3> point)
{
  m_point = point;
}

std::pair<Collection, Collection>
ExtractThreeSlice::execute(Collection &collection)
{
  ExtractSlice slice;
  slice.add_plane(m_point, N0);
  slice.add_plane(m_point, N1);
  slice.add_plane(m_point, N2);
  return slice.execute(collection);
}

}//namespace dray
