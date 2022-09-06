// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef DRAY_TRANSFORM_3D_HPP
#define DRAY_TRANSFORM_3D_HPP

// This header file contains a collection of math functions useful in the
// linear transformation of homogeneous points for rendering in 3D.

#include <dray/matrix.hpp>

namespace dray
{

/// \brief Transform a 3D point by a transformation matrix with perspective.
///
/// Given a 4x4 transformation matrix and a 3D point, returns the point
/// transformed by the given matrix in homogeneous coordinates.
///
/// Unlike Transform3DPoint, this method honors the fourth component of the
/// transformed homogeneous coordinate. This makes it applicable for perspective
/// transformations, but requires some more computations.
///
template <typename T>
DRAY_EXEC Vec<T, 3>
transform_point (const Matrix<T, 4, 4> &matrix, const Vec<T, 3> &point)
{
  Vec<T, 4> homogeneousPoint{ point[0], point[1], point[2], T (1) };
  T inverseW = 1 / dot (matrix.get_row (3), homogeneousPoint);
  return Vec<T, 3>{ dot (matrix.get_row (0), homogeneousPoint) * inverseW,
                          dot (matrix.get_row (1), homogeneousPoint) * inverseW,
                          dot (matrix.get_row (2), homogeneousPoint) * inverseW };
}

/// \brief Transform a 3D vector by a transformation matrix.
///
/// Given a 4x4 transformation matrix and a 3D vector, returns the vector
/// transformed by the given matrix in homogeneous coordinates. Unlike points,
/// vectors do not get translated.
///
template <typename T>
DRAY_EXEC Vec<T, 3>
transform_vector (const Matrix<T, 4, 4> &matrix, const Vec<T, 3> &vector)
{
  Vec<T, 4> homogeneousVector{ vector[0], vector[1], vector[2], T (0) };
  homogeneousVector = matrix * homogeneousVector;
  return Vec<T, 3>{ homogeneousVector[0], homogeneousVector[1],
                          homogeneousVector[2] };
}

/// \brief Returns a scale matrix.
///
/// Given a scale factor for the x, y, and z directions, returns a
/// transformation matrix for those scales.
///
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> scale (const T &scaleX, const T &scaleY, const T &scaleZ)
{
  Matrix<T, 4, 4> scaleMatrix (T (0));
  scaleMatrix (0, 0) = scaleX;
  scaleMatrix (1, 1) = scaleY;
  scaleMatrix (2, 2) = scaleZ;
  scaleMatrix (3, 3) = T (1);
  return scaleMatrix;
}

/// \brief Returns a scale matrix.
///
/// Given a scale factor for the x, y, and z directions (defined in a Vec),
/// returns a transformation matrix for those scales.
///
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> scale (const Vec<T, 3> &scaleVec)
{
  return scale (scaleVec[0], scaleVec[1], scaleVec[2]);
}

/// \brief Returns a scale matrix.
///
/// Given a uniform scale factor, returns a transformation matrix for those
/// scales.
///
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> Transform3DScale (const T &scale)
{
  return scale (scale, scale, scale);
}

/// \brief Returns a translation matrix.
///
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> translate (const T &x, const T &y, const T &z)
{
  Matrix<T, 4, 4> translateMatrix;
  translateMatrix.identity ();
  translateMatrix (0, 3) = x;
  translateMatrix (1, 3) = y;
  translateMatrix (2, 3) = z;
  return translateMatrix;
}
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> translate (const Vec<T, 3> &v)
{
  return translate (v[0], v[1], v[2]);
}

/// \brief Returns a rotation matrix.
///
/// Given an angle (in degrees) and an axis of rotation, returns a
/// transformation matrix that rotates around the given axis. The rotation
/// follows the right-hand rule, so if the vector points toward the user, the
/// rotation will be counterclockwise.
///
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> rotate (T angleDegrees, const Vec<T, 3> &axisOfRotation)
{
  T angleRadians = pi_180f () * angleDegrees;
  Vec<T, 3> normAxis = axisOfRotation;
  normAxis.normalize ();
  T sinAngle = sin (angleRadians);
  T cosAngle = cos (angleRadians);

  Matrix<T, 4, 4> matrix;

  matrix (0, 0) = normAxis[0] * normAxis[0] * (1 - cosAngle) + cosAngle;
  matrix (0, 1) = normAxis[0] * normAxis[1] * (1 - cosAngle) - normAxis[2] * sinAngle;
  matrix (0, 2) = normAxis[0] * normAxis[2] * (1 - cosAngle) + normAxis[1] * sinAngle;
  matrix (0, 3) = T (0);

  matrix (1, 0) = normAxis[1] * normAxis[0] * (1 - cosAngle) + normAxis[2] * sinAngle;
  matrix (1, 1) = normAxis[1] * normAxis[1] * (1 - cosAngle) + cosAngle;
  matrix (1, 2) = normAxis[1] * normAxis[2] * (1 - cosAngle) - normAxis[0] * sinAngle;
  matrix (1, 3) = T (0);

  matrix (2, 0) = normAxis[2] * normAxis[0] * (1 - cosAngle) - normAxis[1] * sinAngle;
  matrix (2, 1) = normAxis[2] * normAxis[1] * (1 - cosAngle) + normAxis[0] * sinAngle;
  matrix (2, 2) = normAxis[2] * normAxis[2] * (1 - cosAngle) + cosAngle;
  matrix (2, 3) = T (0);

  matrix (3, 0) = T (0);
  matrix (3, 1) = T (0);
  matrix (3, 2) = T (0);
  matrix (3, 3) = T (1);

  return matrix;
}
template <typename T>
DRAY_EXEC Matrix<T, 4, 4> rotate (T angleDegrees, T x, T y, T z)
{
  return rotate (angleDegrees, Vec<T, 3> {{x, y, z}});
}

/// \brief Returns a rotation matrix.
///
/// Returns a transformation matrix that rotates around the x axis.
///
template <typename T> DRAY_EXEC Matrix<T, 4, 4> rotate_x (T angleDegrees)
{
  return rotate (angleDegrees, T (1), T (0), T (0));
}

/// \brief Returns a rotation matrix.
///
/// Returns a transformation matrix that rotates around the y axis.
///
template <typename T> DRAY_EXEC Matrix<T, 4, 4> rotate_y (T angleDegrees)
{
  return rotate (angleDegrees, T (0), T (1), T (0));
}

/// \brief Returns a rotation matrix.
///
/// Returns a transformation matrix that rotates around the z axis.
///
template <typename T> DRAY_EXEC Matrix<T, 4, 4> rotate_z (T angleDegrees)
{
  return rotate (angleDegrees, T (0), T (0), T (1));
}

static DRAY_EXEC Matrix<float32, 4, 4>
trackball_matrix (float32 p1x, float32 p1y, float32 p2x, float32 p2y)
{
  const float32 RADIUS = 0.80f; // z value lookAt x = y = 0.0
  const float32 COMPRESSION = 3.5f; // multipliers for x and y.
  const float32 AR3 = RADIUS * RADIUS * RADIUS;

  Matrix<float32, 4, 4> matrix;

  matrix.identity ();

  if (p1x == p2x && p1y == p2y)
  {
    return matrix;
  }

  Vec<float32, 3> p1{ p1x, p1y, AR3 / ((p1x * p1x + p1y * p1y) * COMPRESSION + AR3) };
  Vec<float32, 3> p2{ p2x, p2y, AR3 / ((p2x * p2x + p2y * p2y) * COMPRESSION + AR3) };
  Vec<float32, 3> axis = cross (p2, p1);
  axis.normalize ();

  Vec<float32, 3> p2_p1{ p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
  float32 t = p2_p1.magnitude ();
  t = min (max (t, -1.0f), 1.0f);
  float32 phi = static_cast<float32> (-2.0f * asin (t / (2.0f * RADIUS)));
  float32 val = static_cast<float32> (sin (phi / 2.0f));
  axis[0] *= val;
  axis[1] *= val;
  axis[2] *= val;

  // quaternion
  float32 q[4] = { axis[0], axis[1], axis[2], static_cast<float32> (cos (phi / 2.0f)) };

  // normalize quaternion to unit magnitude
  t = 1.0f / static_cast<float32> (
             sqrt (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]));
  q[0] *= t;
  q[1] *= t;
  q[2] *= t;
  q[3] *= t;

  matrix (0, 0) = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
  matrix (0, 1) = 2 * (q[0] * q[1] + q[2] * q[3]);
  matrix (0, 2) = (2 * (q[2] * q[0] - q[1] * q[3]));

  matrix (1, 0) = 2 * (q[0] * q[1] - q[2] * q[3]);
  matrix (1, 1) = 1 - 2 * (q[2] * q[2] + q[0] * q[0]);
  matrix (1, 2) = (2 * (q[1] * q[2] + q[0] * q[3]));

  matrix (2, 0) = (2 * (q[2] * q[0] + q[1] * q[3]));
  matrix (2, 1) = (2 * (q[1] * q[2] - q[0] * q[3]));
  matrix (2, 2) = (1 - 2 * (q[1] * q[1] + q[0] * q[0]));

  return matrix;
}

} // namespace dray

#endif
