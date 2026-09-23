#include "viskoresRevolve.hpp"

#include <viskores/Math.h>
#include <viskores/Matrix.h>
#include <viskores/cont/ErrorFilterExecution.h>
#include <viskores/filter/field_transform/PointTransform.h>
#include <viskores/filter/geometry_refinement/ExtrusionRotational.h>

namespace vtkh
{

namespace
{

VISKORES_CONT inline viskores::Matrix<viskores::Float64,4,4>
make_axis_rotation_matrix(const viskores::Vec3f_64 &origin,
                          const viskores::Vec3f_64 &axis,
                          const viskores::Float64 angle_radians)
{
  const viskores::Float64 axisMagnitude2 = viskores::MagnitudeSquared(axis);
  if (axisMagnitude2 <= 0.0)
  {
    throw viskores::cont::ErrorFilterExecution("vtkh::Revolve requires a non-zero 'axis'");
  }

  const viskores::Float64 invAxisMagnitude = viskores::RSqrt(axisMagnitude2);
  const viskores::Vec3f_64 u = axis * invAxisMagnitude;

  const viskores::Float64 c = viskores::Cos(angle_radians);
  const viskores::Float64 s = viskores::Sin(angle_radians);
  const viskores::Float64 oneMinusC = 1.0 - c;

  const viskores::Float64 ux = u[0];
  const viskores::Float64 uy = u[1];
  const viskores::Float64 uz = u[2];

  // Rodrigues rotation matrix.
  const viskores::Float64 r00 = c + ux * ux * oneMinusC;
  const viskores::Float64 r01 = ux * uy * oneMinusC - uz * s;
  const viskores::Float64 r02 = ux * uz * oneMinusC + uy * s;

  const viskores::Float64 r10 = uy * ux * oneMinusC + uz * s;
  const viskores::Float64 r11 = c + uy * uy * oneMinusC;
  const viskores::Float64 r12 = uy * uz * oneMinusC - ux * s;

  const viskores::Float64 r20 = uz * ux * oneMinusC - uy * s;
  const viskores::Float64 r21 = uz * uy * oneMinusC + ux * s;
  const viskores::Float64 r22 = c + uz * uz * oneMinusC;

  const viskores::Float64 ox = origin[0];
  const viskores::Float64 oy = origin[1];
  const viskores::Float64 oz = origin[2];

  // Rotate about an axis through `origin` (translate to origin, rotate, translate back):
  // p' = R(p - o) + o = Rp + (o - Ro)
  const viskores::Float64 tx = ox - (r00 * ox + r01 * oy + r02 * oz);
  const viskores::Float64 ty = oy - (r10 * ox + r11 * oy + r12 * oz);
  const viskores::Float64 tz = oz - (r20 * ox + r21 * oy + r22 * oz);

  viskores::Matrix<viskores::Float64,4,4> m;
  m(0,0) = r00; m(0,1) = r01; m(0,2) = r02; m(0,3) = tx;
  m(1,0) = r10; m(1,1) = r11; m(1,2) = r12; m(1,3) = ty;
  m(2,0) = r20; m(2,1) = r21; m(2,2) = r22; m(2,3) = tz;
  m(3,0) = 0.0; m(3,1) = 0.0; m(3,2) = 0.0; m(3,3) = 1.0;
  return m;
}

} // namespace

// Revolve an input mesh around an axis to form a swept dataset.
viskores::cont::DataSet
viskoresRevolve::Run(viskores::cont::DataSet &input,
                     const viskores::Vec<viskores::Float64,3> &point,
                     const viskores::Vec<viskores::Float64,3> &axis,
                     const viskores::Float64 start_angle_degrees,
                     const viskores::Float64 sweep_angle_degrees,
                     const viskores::Int32 steps,
                     const bool periodic,
                     const bool triangulate_input,
                     viskores::filter::FieldSelection map_fields)
{
  if (steps <= 0)
  {
    throw viskores::cont::ErrorFilterExecution("vtkh::Revolve requires 'steps' > 0");
  }

  if (periodic && steps < 3)
  {
    throw viskores::cont::ErrorFilterExecution("vtkh::Revolve with 'periodic' requires 'steps' >= 3");
  }

  const viskores::Float64 start_radians = viskores::Pi() * start_angle_degrees / 180.0;
  const viskores::Float64 sweep_radians = viskores::Pi() * sweep_angle_degrees / 180.0;

  viskores::cont::DataSet workingInput = input;
  if (start_radians != 0.0)
  {
    viskores::filter::field_transform::PointTransform transform;
    transform.SetChangeCoordinateSystem(true);
    const auto rotation = make_axis_rotation_matrix(viskores::Vec3f_64{ point },
                                                    viskores::Vec3f_64{ axis },
                                                    start_radians);
    transform.SetTransform(rotation);
    workingInput = transform.Execute(input);
  }

  viskores::filter::geometry_refinement::ExtrusionRotational revolver;
  revolver.SetFieldsToPass(map_fields);
  revolver.SetTriangulateInput(triangulate_input);
  revolver.SetCompactOutput(false);
  revolver.SetAxis(viskores::Vec3f_64{ axis });
  revolver.SetCenter(viskores::Vec3f_64{ point });
  revolver.SetCloseSweep(periodic);
  revolver.SetSweepAngle(static_cast<viskores::FloatDefault>(sweep_radians));
  revolver.SetNumberOfPlanes(periodic ? static_cast<viskores::Id>(steps)
                                      : (static_cast<viskores::Id>(steps) + 1));

  return revolver.Execute(workingInput);
}

} // namespace vtkh
