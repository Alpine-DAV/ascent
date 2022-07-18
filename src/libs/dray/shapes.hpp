#ifndef DRAY_SHAPES_HPP
#define DRAY_SHAPES_HPP

#include<dray/types.hpp>

namespace dray
{
// Shape Tags for specializations
// m_dims is the topological dimensions of the shape
// also known as 'ref_dim` in some circles
struct Hex
{
  constexpr int32 m_dims = 3;
};

struct Quad
{
  constexpr int32 m_dims = 2;
};

struct Tet
{
  constexpr int32 m_dims = 3;
};

struct Tri
{
  constexpr int32 m_dims = 2;
};

} // namespace dray
#endif
