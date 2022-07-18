#ifndef DRAY_REFLECT_HPP
#define DRAY_REFLECT_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class Reflect
{
protected:
  Vec<float32,3> m_point;
  Vec<float32,3> m_normal;
public:
  Reflect();
  void plane(const Vec<float32,3> &point, const Vec<float32,3> &normal);
  Collection execute(Collection &collection);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP
