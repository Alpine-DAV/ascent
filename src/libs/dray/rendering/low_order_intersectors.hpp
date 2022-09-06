// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LOW_ORDER_INTERSECTORS_HPP
#define DRAY_LOW_ORDER_INTERSECTORS_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/error.hpp>


namespace dray
{

template<typename T>
DRAY_EXEC
void quad_ref_point(const Vec<T,3> &v00,
                    const Vec<T,3> &v11,
                    const T &alpha,
                    const T &beta,
                    const Vec<T,3> &e01,
                    const Vec<T,3> &e03,
                    T &u,
                    T &v)
{

  constexpr T epsilon = 0.00001f;
  // Compute the barycentric coordinates of V11
  T alpha_11, beta_11;
  Vec<T,3> e02 = v11 - v00;
  Vec<T,3> n = cross(e01, e02);

  if ((abs(n[0]) >= abs(n[1])) && (abs(n[0]) >= abs(n[2])))
  {
    alpha_11 = ((e02[1] * e03[2]) - (e02[2] * e03[1])) / n[0];
    beta_11 = ((e01[1] * e02[2]) - (e01[2] * e02[1])) / n[0];
  }
  else if ((abs(n[1]) >= abs(n[0])) && (abs(n[1]) >= abs(n[2])))
  {
    alpha_11 = ((e02[2] * e03[0]) - (e02[0] * e03[2])) / n[1];
    beta_11 = ((e01[2] * e02[0]) - (e01[0] * e02[2])) / n[1];
  }
  else
  {
    alpha_11 = ((e02[0] * e03[1]) - (e02[1] * e03[0])) / n[2];
    beta_11 = ((e01[0] * e02[1]) - (e01[1] * e02[0])) / n[2];
  }

  // Compute the bilinear coordinates of the intersection point.
  if (abs(alpha_11 - 1.0f) < epsilon)
  {

    u = alpha;
    if (abs(beta_11 - 1.0f) < epsilon)
      v = beta;
    else
      v = beta / ((u * (beta_11 - 1.0f)) + 1.0f);
  }
  else if (abs(beta_11 - 1.0) < epsilon)
  {

    v = beta;
    u = alpha / ((v * (alpha_11 - 1.0f)) + 1.0f);
  }
  else
  {

    T a = 1.0f - beta_11;
    T b = (alpha * (beta_11 - 1.0f)) - (beta * (alpha_11 - 1.0f)) - 1.0f;
    T c = alpha;
    T d = (b * b) - (4.0f * a * c);
    T qq = -0.5f * (b + ((b < 0.0f ? -1.0f : 1.0f) * sqrt(d)));
    u = qq / a;
    if ((u < 0.0f) || (u > 1.0f))
    {
      u = c / qq;
    }
    v = beta / ((u * (beta_11 - 1.0f)) + 1.0f);
  }
}

template<typename T>
DRAY_EXEC
T intersect_quad(const Vec<T,3> &v00,
                 const Vec<T,3> &v10,
                 const Vec<T,3> &v11,
                 const Vec<T,3> &v01,
                 const Vec<T,3> &origin,
                 const Vec<T,3> &dir,
                 T &alpha,
                 T &beta,
                 Vec<T,3> &e01,
                 Vec<T,3> &e03)
{
  constexpr T epsilon = 0.00001f;
  T distance = infinity<T>();
  /* An Eﬃcient Ray-Quadrilateral Intersection Test
     Ares Lagae Philip Dutr´e
     http://graphics.cs.kuleuven.be/publications/LD05ERQIT/index.html

  v01 *------------ * v11
      |\           |
      |  \         |
      |    \       |
      |      \     |
      |        \   |
      |          \ |
  v00 *------------* v10
  */
  // Rejects rays that are parallel to Q, and rays that intersect the plane of
  // Q either on the left of the line V00V01 or on the right of the line V00V10.

  e03 = v01 - v00;
  Vec<T,3> p = cross(dir, e03);
  e01 = v10 - v00;
  T det = dot(e01, p);
  bool hit = true;

  const T rel_epsilon = e03.magnitude() * epsilon;

  if (abs(det) < rel_epsilon)
  {
    hit = false;
  }
  T inv_det = 1.0f / det;
  Vec<T,3> t = origin - v00;
  alpha = dot(t, p) * inv_det;
  if (alpha < 0.0)
  {
    hit = false;
  }
  Vec<T,3> q = cross(t, e01);
  beta = dot(dir, q) * inv_det;
  if (beta < 0.0)
  {
    hit = false;
  }

  if ((alpha + beta) > 1.0f)
  {

    // Rejects rays that intersect the plane of Q either on the
    // left of the line V11V10 or on the right of the line V11V01.

    Vec<T,3> e23 = v01 - v11;
    Vec<T,3> e21 = v10 - v11;
    Vec<T,3> p_prime = cross(dir, e21);
    T det_prime = dot(e23, p_prime);
    if (abs(det_prime) < rel_epsilon)
    {
      hit = false;
    }
    T inv_det_prime = 1.0f / det_prime;
    Vec<T,3> t_prime = origin - v11;
    T alpha_prime = dot(t_prime, p_prime) * inv_det_prime;
    if (alpha_prime < 0.0f)
    {
      hit = false;
    }
    Vec<T,3> q_prime = cross(t_prime, e23);
    T beta_prime = dot(dir, q_prime) * inv_det_prime;
    if (beta_prime < 0.0f)
    {
      hit = false;
    }
  }

  // Compute the ray parameter of the intersection point, and
  // reject the ray if it does not hit Q.

  if(hit)
  {
    distance = dot(e03, q) * inv_det;
  }

  return distance;
}

template<typename T>
DRAY_EXEC
T intersect_quad(const Vec<T,3> &v00,
                 const Vec<T,3> &v10,
                 const Vec<T,3> &v01,
                 const Vec<T,3> &v11,
                 const Vec<T,3> &origin,
                 const Vec<T,3> &dir,
                 T &u,
                 T &v)
{
  T alpha;
  T beta;
  Vec<T,3> e01;
  Vec<T,3> e03;
  T distance;
  distance = intersect_quad(v00, v10, v11, v01, origin, dir, alpha, beta, e01, e03);

  if(distance != infinity32())
  {
    quad_ref_point(v00, v11, alpha, beta, e01, e03, u, v);
  }

  return distance;
}

template<typename T>
DRAY_EXEC
T intersect_quad(const Vec<T,3> &v00,
                       const Vec<T,3> &v10,
                       const Vec<T,3> &v01,
                       const Vec<T,3> &v11,
                       const Vec<T,3> &origin,
                       const Vec<T,3> &dir)
{
  T alpha;
  T beta;
  Vec<T,3> e01;
  Vec<T,3> e03;
  T distance;
  distance = intersect_quad(v00, v10, v11, v01, origin, dir, alpha, beta, e01, e03);

  return distance;
}


template<typename T>
DRAY_EXEC
T intersect_tri(const Vec<T,3> &a,
                const Vec<T,3> &b,
                const Vec<T,3> &c,
                const Vec<T,3> &origin,
                const Vec<T,3> &dir,
                T &u,
                T &v)
{
  const T EPSILON2 = 0.0001f;
  T distance = infinity<T>();

  Vec<T, 3> e1 = b - a;
  Vec<T, 3> e2 = c - a;

  Vec<T, 3> p;
  p[0] = dir[1] * e2[2] - dir[2] * e2[1];
  p[1] = dir[2] * e2[0] - dir[0] * e2[2];
  p[2] = dir[0] * e2[1] - dir[1] * e2[0];
  T dot = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (dot != 0.f)
  {
    dot = 1.f / dot;
    Vec<T, 3> t;
    t = origin - a;

    u = (t[0] * p[0] + t[1] * p[1] + t[2] * p[2]) * dot;
    if (u >= (0.f - EPSILON2) && u <= (1.f + EPSILON2))
    {

      Vec<T, 3> q; // = t % e1;
      q[0] = t[1] * e1[2] - t[2] * e1[1];
      q[1] = t[2] * e1[0] - t[0] * e1[2];
      q[2] = t[0] * e1[1] - t[1] * e1[0];

      v = (dir[0] * q[0] +
           dir[1] * q[1] +
           dir[2] * q[2]) * dot;

      if (v >= (0.f - EPSILON2) && v <= (1.f + EPSILON2) && !(u + v > 1.f))
      {
        distance = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * dot;
      }
    }
  }
  return distance;
}

template<typename T>
DRAY_EXEC
T intersect_tri(const Vec<T,3> &a,
                const Vec<T,3> &b,
                const Vec<T,3> &c,
                const Vec<T,3> &origin,
                const Vec<T,3> &dir)
{
  T u,v;
  T distance = intersect_tri(a,b,c,origin,dir,u,v);
  (void) u;
  (void) v;
  return distance;
}

template<typename T>
DRAY_EXEC
T intersect_sphere(const Vec<T,3> &center,
                   const T &radius,
                   const Vec<T,3> &origin,
                   const Vec<T,3> &dir)
{
  T dist = infinity<T>();

  Vec<T, 3> l = center - origin;

  T dot1 = dot(l, dir);
  if (dot1 >= 0)
  {
    T d = dot(l, l) - dot1 * dot1;
    T r2 = radius * radius;
    if (d <= r2)
    {
      T tch = sqrt(r2 - d);
      dist = dot1 - tch;
    }
  }
  return dist;
}

} // namespace dray
#endif
