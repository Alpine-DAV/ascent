#ifndef DRAY_GET_SHADING_CONTEXT_HPP
#define DRAY_GET_SHADING_CONTEXT_HPP

#include <dray/data_set.hpp>
#include <dray/ref_point.hpp>
#include <dray/fragment.hpp>
#include <dray/ray.hpp>

// Ultimately, this file will not be installed with dray
// PRIVATE in cmake
namespace dray
{
namespace internal
{

template <class MeshElem, class FieldElem>
Array<Fragment>
get_fragments(Array<Ray> &rays,
                    Range scalar_range,
                    Field<FieldElem> &field,
                    Mesh<MeshElem> &mesh,
                    Array<RayHit> &hits);

}; // namespace internal

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP
