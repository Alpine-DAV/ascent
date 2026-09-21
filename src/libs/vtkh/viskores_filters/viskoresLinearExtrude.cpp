#include "viskoresLinearExtrude.hpp"

#include <viskores/CellShape.h>
#include <viskores/Math.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleTransform.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/ErrorFilterExecution.h>
#include <viskores/cont/UnknownCellSet.h>
#include <viskores/filter/geometry_refinement/ExtrusionLinear.h>

namespace vtkh
{

namespace
{

struct BinaryAnd
{
  VISKORES_EXEC_CONT
  bool operator()(bool u, bool v) const { return u && v; }
};

struct IsTriangleShape
{
  VISKORES_EXEC_CONT
  bool operator()(viskores::UInt8 shape) const { return shape == viskores::CellShapeTagTriangle::Id; }
};

struct IsThreeIndices
{
  template <typename ValueType>
  VISKORES_EXEC_CONT bool operator()(ValueType count) const
  {
    return count == 3;
  }
};

VISKORES_CONT inline bool
NeedsTriangulateForExtrusion(const viskores::cont::UnknownCellSet &unknownCellSet)
{
  if(unknownCellSet.CanConvert<viskores::cont::CellSetSingleType<>>())
  {
    const auto cellSet = unknownCellSet.AsCellSet<viskores::cont::CellSetSingleType<>>();
    return cellSet.GetCellShapeAsId() != viskores::CellShapeTagTriangle::Id;
  }

  if(unknownCellSet.CanConvert<viskores::cont::CellSetExplicit<>>())
  {
    const auto cellSet = unknownCellSet.AsCellSet<viskores::cont::CellSetExplicit<>>();

    const auto shapes = cellSet.GetShapesArray(viskores::TopologyElementTagCell{},
                                               viskores::TopologyElementTagPoint{});
    const bool allTriangleShapes = viskores::cont::Algorithm::Reduce(viskores::cont::make_ArrayHandleTransform(shapes, IsTriangleShape{}),
                                                                     true,
                                                                     BinaryAnd{});

    const auto numIndices = cellSet.GetNumIndicesArray(viskores::TopologyElementTagCell{},
                                                       viskores::TopologyElementTagPoint{});
    const bool allTriangleCounts = viskores::cont::Algorithm::Reduce(viskores::cont::make_ArrayHandleTransform(numIndices, IsThreeIndices{}),
                                                                     true,
                                                                     BinaryAnd{});

    return !allTriangleShapes || !allTriangleCounts;
  }

  // Unknown cell set type: allow Viskores to attempt a triangulation internally.
  return true;
}

} // namespace

// Linearly extrude an input mesh along a vector for a fixed number of steps.
viskores::cont::DataSet
viskoresLinearExtrude::Run(viskores::cont::DataSet &input,
                           const viskores::Vec<viskores::Float64,3> &vector,
                           const viskores::Int32 steps,
                           viskores::filter::FieldSelection map_fields)
{
  if (steps <= 0)
  {
    throw viskores::cont::ErrorFilterExecution("vtkh::LinearExtrude requires 'steps' > 0");
  }

  const viskores::Float64 perStepDistance = viskores::Magnitude(vector) / static_cast<viskores::Float64>(steps);

  viskores::filter::geometry_refinement::ExtrusionLinear extruder;
  extruder.SetFieldsToPass(map_fields);
  extruder.SetTriangulateInput(NeedsTriangulateForExtrusion(input.GetCellSet()));
  extruder.SetCompactOutput(false);
  extruder.SetNumberOfPlanes(static_cast<viskores::Id>(steps) + 1);
  extruder.SetDirection(viskores::Vec3f(static_cast<viskores::FloatDefault>(vector[0]),
                                        static_cast<viskores::FloatDefault>(vector[1]),
                                        static_cast<viskores::FloatDefault>(vector[2])));
  extruder.SetDistance(static_cast<viskores::FloatDefault>(perStepDistance));

  return extruder.Execute(input);
}

} // namespace vtkh
