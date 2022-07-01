// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/camera.hpp>

#include <dray/array_utils.hpp>
#include <dray/halton.hpp>
#include <dray/policies.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/transform_3d.hpp>

#include <random>
#include <sstream>

namespace dray
{

namespace detail
{

void init_random (Array<int32> &random)
{
  const int size = random.size ();
  std::random_device rd;
  std::mt19937 gen (rd ());
  std::uniform_int_distribution<> dis (0, size * 2);

  int32 *random_ptr = random.get_host_ptr ();

  for (int32 i = 0; i < size; ++i)
  {
    random_ptr[i] = dis (gen);
  }
}

} // namespace detail

Camera::Camera ()
{
  m_height = 500;
  m_width = 500;
  m_subset_width = 500;
  m_subset_height = 500;
  m_subset_min_x = 0;
  m_subset_min_y = 0;
  m_fov_y = 30.f;
  m_fov_x = 30.f;
  m_look[0] = 0.f;
  m_look[1] = 0.f;
  m_look[2] = -1.f;
  m_look_at[0] = 0.f;
  m_look_at[1] = 0.f;
  m_look_at[2] = -1.f;
  m_up[0] = 0.f;
  m_up[1] = 1.f;
  m_up[2] = 0.f;
  m_position[0] = 0.f;
  m_position[1] = 0.f;
  m_position[2] = 0.f;
  m_sample = 0;
  m_zoom = 1.f;
  m_ray_differential_x = {{0.f,0.f,0.f}};
  m_ray_differential_y = {{0.f,0.f,0.f}};
}

Camera::~Camera ()
{
}

void Camera::set_height (const int32 &height)
{
  if (height <= 0)
  {
    DRAY_ERROR("Camera height must be greater than zero.");
  }
  if (m_height != height)
  {
    m_height = height;
    m_subset_height = height;
    set_fov (m_fov_y);
  }
}

int32 Camera::get_height () const
{
  return m_height;
}

void Camera::set_zoom(const float32 zoom)
{
  if (zoom <= 0)
  {
    DRAY_ERROR("Zoom must be greater than 0");
  }
  m_zoom = zoom;
}


void Camera::set_width (const int32 &width)
{
  if (width <= 0)
  {
    DRAY_ERROR("Camera width must be greater than zero.");
  }

  m_width = width;
  m_subset_width = width;
  set_fov (m_fov_y);
}


int32 Camera::get_width () const
{
  return m_width;
}


int32 Camera::get_subset_width () const
{
  return m_subset_width;
}


int32 Camera::get_subset_height () const
{
  return m_subset_height;
}

void Camera::set_fov (const float32 &degrees)
{
  if (degrees <= 0)
  {
    DRAY_ERROR("Camera feild of view must be greater than zero.");
  }
  if (degrees > 180)
  {
    DRAY_ERROR("Camera feild of view must be less than 180.");
  }

  float32 new_fov_y = degrees;
  float32 new_fov_x;

  float32 fov_y_rad = (new_fov_y * static_cast<float32> (pi ())) / 180.0f;

  // Use the tan function to find the distance from the center of the image to the top (or
  // bottom). (Actually, we are finding the ratio of this distance to the near plane distance,
  // but since we scale everything by the near plane distance, we can use this ratio as a scaled
  // proxy of the distances we need.)
  float32 vertical_distance = tan (0.5f * fov_y_rad);

  // Scale the vertical distance by the aspect ratio to get the horizontal distance.
  float32 aspect = float32 (m_width) / float32 (m_height);
  float32 horizontal_distance = aspect * vertical_distance;

  // Now use the arctan function to get the proper field of view in the x direction.
  float32 fov_x_rad = 2.0f * atan (horizontal_distance);
  new_fov_x = 180.0f * (fov_x_rad / static_cast<float32> (pi ()));
  m_fov_x = new_fov_x;
  m_fov_y = new_fov_y;
}


float32 Camera::get_fov () const
{
  return m_fov_y;
}


void Camera::set_up (const Vec<float32, 3> &up)
{
  m_up = up;
  m_up.normalize ();
}


Vec<float32, 3> Camera::get_up () const
{
  return m_up;
}


void Camera::set_look_at (const Vec<float32, 3> &look_at)
{
  m_look_at = look_at;
}


Vec<float32, 3> Camera::get_look_at () const
{
  return m_look_at;
}


void Camera::set_pos (const Vec<float32, 3> &position)
{
  m_position = position;
}


Vec<float32, 3> Camera::get_pos () const
{
  return m_position;
}

void Camera::create_rays (Array<Ray> &rays, AABB<> bounds)
{
  create_rays_imp (rays, bounds);
}


void Camera::create_rays_jitter (Array<Ray> &rays, AABB<> bounds)
{
  create_rays_jitter_imp (rays, bounds);
}

void Camera::create_rays_imp (Array<Ray> &rays, AABB<> bounds)
{
  int32 num_rays = m_width * m_height;
  // TODO: find subset
  // for now just set
  m_subset_width = m_width;
  m_subset_height = m_height;
  m_subset_min_x = 0;
  m_subset_min_y = 0;

  rays.resize (num_rays);

  Vec<Float, 3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  m_look = m_look_at - m_position;

  // TODO Why don't we set rays.m_dist to the same 0.0 as m_near?

  gen_perspective (rays);

  // rays.m_active_rays = array_counting(rays.size(),0,1);
}

Vec<float32,3> Camera::ray_differential_x() const
{
  return m_ray_differential_x;
}
Vec<float32,3> Camera::ray_differential_y() const
{
  return m_ray_differential_y;
}

void Camera::create_rays_jitter_imp (Array<Ray> &rays, AABB<> bounds)
{
  int32 num_rays = m_width * m_height;
  // TODO: find subset
  // for now just set
  m_subset_width = m_width;
  m_subset_height = m_height;
  m_subset_min_x = 0;
  m_subset_min_y = 0;

  rays.resize (num_rays);

  Vec<Float, 3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  m_look = m_look_at - m_position;

  // TODO Why don't we set rays.m_dist to the same 0.0 as m_near?

  gen_perspective_jitter (rays);

  // rays.m_active_rays = array_counting(rays.size(),0,1);
}


std::string Camera::print () const
{
  std::stringstream sstream;
  sstream << "------------------------------------------------------------\n";
  sstream << "Position : [" << m_position[0] << ",";
  sstream << m_position[1] << ",";
  sstream << m_position[2] << "]\n";
  sstream << "LookAt   : [" << m_look_at[0] << ",";
  sstream << m_look_at[1] << ",";
  sstream << m_look_at[2] << "]\n";
  sstream << "FOV_X    : " << m_fov_x << "\n";
  sstream << "Up       : [" << m_up[0] << ",";
  sstream << m_up[1] << ",";
  sstream << m_up[2] << "]\n";
  sstream << "Width    : " << m_width << "\n";
  sstream << "Height   : " << m_height << "\n";
  sstream << "Subset W : " << m_subset_width << "\n";
  sstream << "Subset H : " << m_subset_height << "\n";
  sstream << "------------------------------------------------------------\n";
  return sstream.str ();
}


void Camera::gen_perspective_jitter (Array<Ray> &rays)
{
  Vec<Float, 3> nlook;
  Vec<Float, 3> delta_x;
  Vec<Float, 3> delta_y;

  Float thx = tanf ((m_fov_x * Float (pi ()) / 180.f) * .5f);
  Float thy = tanf ((m_fov_y * Float (pi ()) / 180.f) * .5f);
  Vec<float32, 3> ruf = cross (m_look, m_up);
  Vec<Float, 3> ru;
  ru[0] = ruf[0];
  ru[1] = ruf[1];
  ru[2] = ruf[2];

  ru.normalize ();

  Vec<float32, 3> rvf = cross (ruf, m_look);
  Vec<Float, 3> rv;
  rv[0] = rvf[0];
  rv[1] = rvf[1];
  rv[2] = rvf[2];

  rv.normalize ();
  delta_x = ru * (2 * thx / (Float)m_width);
  delta_y = rv * (2 * thy / (Float)m_height);

  if (m_zoom > 0)
  {
    delta_x = delta_x / m_zoom;
    delta_y = delta_y / m_zoom;
  }

  nlook[0] = m_look[0];
  nlook[1] = m_look[1];
  nlook[2] = m_look[2];
  nlook.normalize ();

  Vec<Float, 3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  const int size = rays.size ();
  if (m_random.size () != size)
  {
    m_random.resize (size);
    detail::init_random (m_random);
  }

  int32 sample = m_sample;

  int32 *random_ptr = m_random.get_device_ptr ();
  Ray *rays_ptr = rays.get_device_ptr ();
  // Vec<T, 3> *dir_ptr = rays.m_dir.get_device_ptr();
  // int32 *pid_ptr = rays.m_pixel_id.get_device_ptr();
  // something weird is happening with the
  // lambda capture
  const int32 w = m_width;
  const int32 sub_min_x = m_subset_min_x;
  const int32 sub_min_y = m_subset_min_y;
  const int32 sub_w = m_subset_width;
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 idx) {
    Ray ray;
    // init stuff
    ray.m_orig = pos;
    ray.m_near = Float (0.f);
    ray.m_far = infinity<Float> ();

    Vec<Float, 2> xy;
    int32 sample_index = sample + random_ptr[idx];
    Halton2D<Float, 3> (sample_index, xy);
    xy[0] -= 0.5f;
    xy[1] -= 0.5f;

    int32 i = int32 (idx) % sub_w;
    int32 j = int32 (idx) / sub_w;
    i += sub_min_x;
    j += sub_min_y;
    // Write out the global pixelId
    ray.m_pixel_id = static_cast<int32> (j * w + i);
    ray.m_dir = nlook + delta_x * ((2.f * (Float (i) + xy[0]) - Float (w)) / 2.0f) +
                delta_y * ((2.f * (Float (j) + xy[1]) - Float (w)) / 2.0f);
    // avoid some numerical issues
    for (int32 d = 0; d < 3; ++d)
    {
      if (ray.m_dir[d] == 0.f) ray.m_dir[d] += 0.0000001f;
    }

    ray.m_dir.normalize ();

    // printf("Ray dir %f %f %f\n", ray.m_dir[0], ray.m_dir[1], ray.m_dir[2]);
    rays_ptr[idx] = ray;
  });
  DRAY_ERROR_CHECK();

  m_sample += 1;
}

void Camera::gen_perspective (Array<Ray> &rays)
{
  Vec<Float, 3> nlook;
  Vec<Float, 3> delta_x;
  Vec<Float, 3> delta_y;

  Float thx = tanf ((m_fov_x * Float (pi ()) / 180.f) * .5f);
  Float thy = tanf ((m_fov_y * Float (pi ()) / 180.f) * .5f);
  Vec<float32, 3> ruf = cross (m_look, m_up);
  Vec<Float, 3> ru;
  ru[0] = ruf[0];
  ru[1] = ruf[1];
  ru[2] = ruf[2];

  ru.normalize ();

  Vec<float32, 3> rvf = cross (ruf, m_look);
  Vec<Float, 3> rv;
  rv[0] = rvf[0];
  rv[1] = rvf[1];
  rv[2] = rvf[2];

  rv.normalize ();
  delta_x = ru * (2 * thx / (Float)m_width);
  delta_y = rv * (2 * thy / (Float)m_height);

  m_ray_differential_x = delta_x;
  m_ray_differential_y = delta_y;

  if (m_zoom > 0)
  {
    delta_x = delta_x / m_zoom;
    delta_y = delta_y / m_zoom;
  }

  nlook[0] = m_look[0];
  nlook[1] = m_look[1];
  nlook[2] = m_look[2];
  nlook.normalize ();

  Vec<Float, 3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  const int size = rays.size ();
  Ray *rays_ptr = rays.get_device_ptr ();
  // Vec<T, 3> *dir_ptr = rays.m_dir.get_device_ptr();
  // int32 *pid_ptr = rays.m_pixel_id.get_device_ptr();
  // something weird is happening with the
  // lambda capture
  const int32 w = m_width;
  const int32 sub_min_x = m_subset_min_x;
  const int32 sub_min_y = m_subset_min_y;
  const int32 sub_w = m_subset_width;
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 idx) {
    Ray ray;
    // init stuff
    ray.m_orig = pos;
    ray.m_near = Float (0.f);
    ray.m_far = infinity<Float> ();
    int32 i = int32 (idx) % sub_w;
    int32 j = int32 (idx) / sub_w;
    i += sub_min_x;
    j += sub_min_y;
    // Write out the global pixelId
    ray.m_pixel_id = static_cast<int32> (j * w + i);
    ray.m_dir = nlook + delta_x * ((2.f * Float (i) - Float (w)) / 2.0f) +
                delta_y * ((2.f * Float (j) - Float (w)) / 2.0f);
    // avoid some numerical issues
    for (int32 d = 0; d < 3; ++d)
    {
      if (ray.m_dir[d] == 0.f) ray.m_dir[d] += 0.0000001f;
    }

    ray.m_dir.normalize ();

    // printf("Ray dir %f %f %f\n", ray.m_dir[0], ray.m_dir[1], ray.m_dir[2]);
    rays_ptr[idx] = ray;
  });
  DRAY_ERROR_CHECK();
}

void Camera::reset_to_bounds (const AABB<> bounds, const float64 xpad, const float64 ypad, const float64 zpad)
{
  AABB<> db;

  float64 pad = xpad * (bounds.m_ranges[0].max () - bounds.m_ranges[0].min ());
  db.m_ranges[0].include (bounds.m_ranges[0].max () + pad);
  db.m_ranges[0].include (bounds.m_ranges[0].min () - pad);

  pad = ypad * (bounds.m_ranges[1].max () - bounds.m_ranges[1].min ());
  db.m_ranges[1].include (bounds.m_ranges[1].max () + pad);
  db.m_ranges[1].include (bounds.m_ranges[1].min () - pad);

  pad = zpad * (bounds.m_ranges[2].max () - bounds.m_ranges[2].min ());
  db.m_ranges[2].include (bounds.m_ranges[2].max () + pad);
  db.m_ranges[2].include (bounds.m_ranges[2].min () - pad);

  Vec3f proj_dir = m_position - m_look_at;
  proj_dir.normalize ();

  Vec3f center = db.center ();
  m_look_at = center;

  Vec3f extent;
  extent[0] = float32 (db.m_ranges[0].length ());
  extent[1] = float32 (db.m_ranges[1].length ());
  extent[2] = float32 (db.m_ranges[2].length ());
  float32 diagonal = extent.magnitude ();
  m_position = center + proj_dir * diagonal * 1.0f;
  set_fov (60.0f);
}

void Camera::elevate (const float32 degrees)
{
  Vec3f look = m_look_at - m_position; // Tail: Position. Tip: Looking target.
  float32 distance = look.magnitude ();
  // make sure we have good basis
  Vec<float32, 3> right = cross (look, m_up);
  Vec<float32, 3> up = cross (right, look);

  look.normalize ();
  right.normalize ();
  up.normalize ();

  Matrix<float32, 4, 4> rotation = rotate (-degrees, right);

  look = transform_vector (rotation, look);

  /// m_up = transform_vector(rotation, up);  // What if I don't want relative up
  m_position = m_look_at - look * distance;
}

void Camera::azimuth (const float32 degrees)
{
  Vec3f look = m_look_at - m_position;
  float32 distance = look.magnitude ();
  // make sure we have good basis
  Vec<float32, 3> right = cross (look, m_up);
  Vec<float32, 3> up = cross (right, look);

  look.normalize ();
  right.normalize ();
  up.normalize ();

  Matrix<float32, 4, 4> rotation = rotate (degrees, up);

  look = transform_vector (rotation, look);

  /// m_up = transform_vector(rotation, up);
  /// m_up = up;
  m_position = m_look_at - look * distance;
}

void Camera::trackball_rotate (float32 startX, float32 startY, float32 endX, float32 endY)
{
  Matrix<float32, 4, 4> rotate = trackball_matrix (startX, startY, endX, endY);

  Matrix<float32, 4, 4> trans = translate (-m_look_at);
  Matrix<float32, 4, 4> inv_trans = translate (m_look_at);


  Matrix<float32, 4, 4> view = view_matrix ();
  view (0, 3) = 0;
  view (1, 3) = 0;
  view (2, 3) = 0;

  Matrix<float32, 4, 4> inverseView = view.transpose ();

  Matrix<float32, 4, 4> full_transform = inv_trans * inverseView * rotate * view * trans;

  m_position = transform_point (full_transform, m_position);
  m_look_at = transform_point (full_transform, m_look_at);
  m_up = transform_vector (full_transform, m_up);
}

Matrix<float32, 4, 4> Camera::view_matrix () const
{
  Vec<float32, 3> viewDir = m_position - m_look_at;
  Vec<float32, 3> right = cross (m_up, viewDir);
  Vec<float32, 3> ru = cross (viewDir, right);

  viewDir.normalize ();
  right.normalize ();
  ru.normalize ();

  Matrix<float32, 4, 4> matrix;
  matrix.identity ();

  matrix (0, 0) = right[0];
  matrix (0, 1) = right[1];
  matrix (0, 2) = right[2];
  matrix (1, 0) = ru[0];
  matrix (1, 1) = ru[1];
  matrix (1, 2) = ru[2];
  matrix (2, 0) = viewDir[0];
  matrix (2, 1) = viewDir[1];
  matrix (2, 2) = viewDir[2];

  matrix (0, 3) = -dray::dot (right, m_position);
  matrix (1, 3) = -dray::dot (ru, m_position);
  matrix (2, 3) = -dray::dot (viewDir, m_position);
  return matrix;
}

Matrix<float32, 4, 4> Camera::projection_matrix (const float32 near, const float32 far) const
{
  Matrix<float32, 4, 4> matrix;
  matrix.identity ();

  float32 aspect_ratio = float32 (m_width) / float32 (m_height);
  float32 fov_x = m_fov_x / m_zoom;
  float32 fov_rad = fov_x * pi_180f();
  fov_rad = tan (fov_rad * 0.5f);
  float32 size = near * fov_rad;
  float32 left = -size * aspect_ratio;
  float32 right = size * aspect_ratio;
  float32 bottom = -size;
  float32 top = size;

  matrix (0, 0) = 2.f * near / (right - left);
  matrix (1, 1) = 2.f * near / (top - bottom);
  matrix (0, 2) = (right + left) / (right - left);
  matrix (1, 2) = (top + bottom) / (top - bottom);
  matrix (2, 2) = -(far + near) / (far - near);
  matrix (3, 2) = -1.f;
  matrix (2, 3) = -(2.f * far * near) / (far - near);
  matrix (3, 3) = 0.f;

  return matrix;
}

// given 3D bounds in world space, returns the projection matrix by determining
// logical near and far values.
Matrix<float32, 4, 4> Camera::projection_matrix (const AABB<3> bounds) const
{
  float minx, miny, minz, maxx, maxy, maxz;
  minx = bounds.m_ranges[0].min();
  miny = bounds.m_ranges[1].min();
  minz = bounds.m_ranges[2].min();
  maxx = bounds.m_ranges[0].max();
  maxy = bounds.m_ranges[1].max();
  maxz = bounds.m_ranges[2].max();

  Matrix<float32, 4, 4> V = this->view_matrix();

  Vec<float32, 3> o,i,j,k,ij,ik,jk,ijk;
  o = transform_point(V, ((Vec<float32,3>) {{minx, miny, minz}}));
  i = transform_point(V, ((Vec<float32,3>) {{maxx, miny, minz}}));
  j = transform_point(V, ((Vec<float32,3>) {{minx, maxy, minz}}));
  k = transform_point(V, ((Vec<float32,3>) {{minx, miny, maxz}}));
  ij = transform_point(V, ((Vec<float32,3>) {{maxx, maxy, minz}}));
  ik = transform_point(V, ((Vec<float32,3>) {{maxx, miny, maxz}}));
  jk = transform_point(V, ((Vec<float32,3>) {{minx, maxy, maxz}}));
  ijk = transform_point(V, ((Vec<float32,3>) {{maxx, maxy, maxz}}));

  float near, far;
  float z_values[] = {o[2], i[2], j[2], k[2], ij[2], ik[2], jk[2], ijk[2]};
  near = z_values[0];
  far = z_values[0];
  for (int i = 1; i < 8; i ++)
  {
    if (z_values[i] < near)
    {
      near = z_values[i];
    }
    if (z_values[i] > far)
    {
      far = z_values[i];
    }
  }

  near = abs(near);
  far = abs(far);

  if (near > far)
  {
    float temp = far;
    far = near;
    near = temp;
  }

  float diff = far - near;
  float fudge = diff * 1e-5;

  return this->projection_matrix(std::max(1e-5f, near - fudge), far + fudge);
}

Array<float32> Camera::gl_depth(const Array<float32> &world_depth,
                                const float32 near,
                                const float32 far)
{
  int32 image_size = this->get_width() * this->get_height();

  Array<float32> dbuffer;
  dbuffer.resize(image_size);
  array_memset(dbuffer, 1.0001f);

  float32 *d_ptr = dbuffer.get_device_ptr();
  const float32 *in_ptr = world_depth.get_device_ptr_const();
  Matrix<float32,4,4> view_proj = this->projection_matrix(near, far) * this->view_matrix();

  // we have to recreate the rays to get the world position
  // of the depth. This could technically go away if the parallel
  // compositing could handle any depth range, which is doesn't
  // for important reasons long forgotten. Shame on my past self,
  // he did future self no favors.
  Vec<float32, 3> nlook;
  Vec<float32, 3> delta_x;
  Vec<float32, 3> delta_y;

  Float thx = tanf ((m_fov_x *  float32(pi ()) / 180.f) * .5f);
  Float thy = tanf ((m_fov_y *  float32(pi ()) / 180.f) * .5f);
  Vec<float32, 3> ruf = cross (m_look, m_up);
  Vec<float32, 3> ru;
  ru[0] = ruf[0];
  ru[1] = ruf[1];
  ru[2] = ruf[2];

  ru.normalize ();

  Vec<float32, 3> rv = cross (ruf, m_look);
  rv.normalize ();

  delta_x = ru * (2 * thx / (float32)m_width);
  delta_y = rv * (2 * thy / (float32)m_height);

  nlook[0] = m_look[0];
  nlook[1] = m_look[1];
  nlook[2] = m_look[2];
  nlook.normalize ();

  Vec<float32, 3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  const int32 width = this->get_width();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, image_size), [=] DRAY_LAMBDA (int32 i)
  {
    float32 depth = in_ptr[i];
    if(depth != infinity32())
    {
      const int32 x = int32 (i) % width;
      const int32 y = int32 (i) / width;

     Vec<float32,3> dir = nlook + delta_x * ((2.f * float32(x) - float32(width)) / 2.0f) +
                delta_y * ((2.f * float32(y) - float32(width)) / 2.0f);
      Vec<float32,3> hit = pos + dir * depth;
      Vec<float32,3> transformed = transform_point(view_proj, hit);
      depth = 0.5f * transformed[2] + 0.5f;
    }
    d_ptr[i] = depth;
  });

  return dbuffer;
}

void Camera::gl_to_world_depth(Array<float32> &gl_depth,
                               const float32 near,
                               const float32 far)
{
  int32 image_size = this->get_width() * this->get_height();


  float32 *depth_ptr = gl_depth.get_device_ptr();

  Matrix<float32,4,4> view_proj = this->projection_matrix(near, far) * this->view_matrix();
  bool valid;
  MatrixInverse<float32,4> inverse(view_proj, valid);

  const int32 width = this->get_width();
  const int32 height = this->get_height();
  float32 double_inv_height = 2.f / static_cast<float32>(height);
  float32 double_inv_width = 2.f / static_cast<float32>(width);

  Vec<float32,3> origin = this->get_pos();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, image_size), [=] DRAY_LAMBDA (int32 i)
  {

    const int32 x = int32 (i) % width;
    const int32 y = int32 (i) / width;
    Vec<float32,4> pos;
    pos[0] = static_cast<float32>(x);
    pos[1] = static_cast<float32>(y);
    pos[2] = depth_ptr[i];
    pos[3] = 1.f;

    // transform into normalized device coords (-1,1)
    pos[0] = pos[0] * double_inv_height - 1.f;
    pos[1] = pos[1] * double_inv_width - 1.f;

    // offset a little
    pos[2] -= 0.00001f;

    pos = inverse * pos;

    Vec<float32,3> world_pos;
    world_pos[0] = pos[0]/pos[3];
    world_pos[1] = pos[1]/pos[3];
    world_pos[2] = pos[2]/pos[3];

    depth_ptr[i] = (world_pos - origin).magnitude();
  });
}

int32 Camera::subset_size(AABB<3> bounds)
{
  int32 res = 0;
  // we need a clipping range to create a perspective projection,
  // so just construct one that wont clip anything
  int32 max_comp = bounds.max_dim();
  float32 max_dim = bounds.m_ranges[max_comp].length();
  max_dim *= 100.f;

  Matrix<float32,4,4> view_proj =
    this->projection_matrix(0.001f, max_dim) * this->view_matrix();

  float32 x[2], y[2], z[2];
  x[0] = static_cast<float32>(bounds.m_ranges[0].min());
  x[1] = static_cast<float32>(bounds.m_ranges[0].max());
  y[0] = static_cast<float32>(bounds.m_ranges[1].min());
  y[1] = static_cast<float32>(bounds.m_ranges[1].max());
  z[0] = static_cast<float32>(bounds.m_ranges[2].min());
  z[1] = static_cast<float32>(bounds.m_ranges[2].max());

  Vec<Float, 3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  //Inside the data bounds
  if (bounds.contains(pos))
  {
    return m_width * m_height;
  }

  float32 xmin, ymin, xmax, ymax, zmin, zmax;
  xmin = infinity32();
  ymin = infinity32();
  zmin = infinity32();
  xmax = neg_infinity32();
  ymax = neg_infinity32();
  zmax = neg_infinity32();
  Vec<float32,4> extent_point;
  for (int32 i = 0; i < 2; ++i)
    for (int32 j = 0; j < 2; ++j)
      for (int32 k = 0; k < 2; ++k)
      {
        extent_point[0] = x[i];
        extent_point[1] = y[j];
        extent_point[2] = z[k];
        extent_point[3] = 1.f;
        Vec<float32,4>  transformed = view_proj * extent_point;
        // perform the perspective divide
        for (int32 a = 0; a < 3; ++a)
        {
          transformed[a] = transformed[a] / transformed[3];
        }

        transformed[0] = (transformed[0] * 0.5f + 0.5f) * static_cast<float32>(m_width);
        transformed[1] = (transformed[1] * 0.5f + 0.5f) * static_cast<float32>(m_height);
        transformed[2] = (transformed[2] * 0.5f + 0.5f);
        zmin = std::min(zmin, transformed[2]);
        zmax = std::max(zmax, transformed[2]);
        if (transformed[2] < 0 || transformed[2] > 1)
        {
          continue;
        }
        xmin = std::min(xmin, transformed[0]);
        ymin = std::min(ymin, transformed[1]);
        xmax = std::max(xmax, transformed[0]);
        ymax = std::max(ymax, transformed[1]);
      }

  xmin -= .001f;
  xmax += .001f;
  ymin -= .001f;
  ymax += .001f;
  xmin = std::floor(std::min(std::max(0.f, xmin), float32(m_width)));
  xmax = std::ceil(std::min(std::max(0.f, xmax), float32(m_width)));
  ymin = std::floor(std::min(std::max(0.f, ymin), float32(m_height)));
  ymax = std::ceil(std::min(std::max(0.f, ymax), float32(m_height)));

  int32 dx = int32(xmax) - int32(xmin);
  int32 dy = int32(ymax) - int32(ymin);
  //
  //  scene is behind the camera
  //
  if (zmax < 0 || xmin >= xmax || ymin >= ymax)
  {
    res = 0;
  }
  else
  {
    res = dx * dy;
  }

  return res;
}

} // namespace dray
