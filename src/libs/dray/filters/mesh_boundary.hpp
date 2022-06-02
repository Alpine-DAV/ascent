#ifndef DRAY_MESH_BOUNDARY_HPP
#define DRAY_MESH_BOUNDARY_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class MeshBoundary
{
protected:
public:
  /**
   * @brief Extracts the boundary (surface) from a topologically 3D dataset,
   *        returing a topologically 2D dataset.
   * @param data_set The topologically 3D dataset.
   * @return A topologically 2D dataset. The Element types will be preserved
   *         except for the dimension.
   *
   * Assume that ElemT::get_dim()==3, so we will return NDElem with dim 2.
   */
  Collection execute(Collection &collection);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP
