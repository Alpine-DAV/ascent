// Copyright 2022 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dispatcher.hpp>
#include <dray/filters/clipfield.hpp>
#include <dray/filters/clip.hpp>
#include <dray/policies.hpp>
#include <RAJA/RAJA.hpp>

namespace dray
{

// Make the sphere distance field.
class SphereDistance
{
public:
  SphereDistance(const Float center[3], const Float radius) : m_output()
  {
    m_center[0] = center[0];
    m_center[1] = center[1];
    m_center[2] = center[2];
    m_radius = radius;
  }

  template <class MeshType>
  void operator()(MeshType &mesh)
  {
    // Inputs
    const GridFunction<3> &mesh_gf = mesh.get_dof_data();
    DeviceGridFunctionConst<3> mesh_dgf(mesh_gf);
    auto ndofs = mesh_gf.m_values.size();

    // Outputs
    GridFunction<1> gf;
    gf.m_el_dofs = mesh_gf.m_el_dofs;
    gf.m_size_el = mesh_gf.m_size_el;
    gf.m_size_ctrl = mesh_gf.m_size_ctrl;
    gf.m_ctrl_idx = mesh_gf.m_ctrl_idx;
    gf.m_values.resize(ndofs);
    DeviceGridFunction<1> dgf(gf);

    // Execute
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, ndofs),
      [=] DRAY_LAMBDA (int32 id)
    {
      // Get accessor for mesh's coord at this dof "id".
      const ReadDofPtr<Vec<Float, 3>> mesh_rdp = mesh_dgf.get_rdp(id);
      // Get accessor for output gf at this dof "id".
      WriteDofPtr<Vec<Float, 1>> gf_wdp = dgf.get_wdp(id);

      Float dx = m_center[0] - mesh_rdp[0][0];
      Float dy = m_center[1] - mesh_rdp[0][1];
      Float dz = m_center[2] - mesh_rdp[0][2];
      Float dist = sqrt(dx*dx + dy*dy + dz*dz) - m_radius;
      gf_wdp[0] = dist;
    });

    // Wrap the new GridFunction as a Field.
    using MeshElemType = typename MeshType::ElementType;
    using FieldElemType = Element<MeshElemType::get_dim(),
                                  1,
                                  MeshElemType::get_etype(),
                                  MeshElemType::get_P()>;
    m_output = std::make_shared<UnstructuredField<FieldElemType>>(
                 gf, mesh.order());
  }

  std::shared_ptr<Field> m_output;
  Float m_center[3];
  Float m_radius;
};

class Clip::InternalsType
{
public:
  InternalsType() : boxbounds()
  {
  }

  ~InternalsType()
  {
  }

  void SetBoxClip(const AABB<3> &bounds)
  {
    clip_mode = 0;
    boxbounds = bounds;
  }

  void SetSphereClip(const Float center[3], const Float radius)
  {
    clip_mode = 1;
    sphere_center[0] = center[0];
    sphere_center[1] = center[1];
    sphere_center[2] = center[2];
    sphere_radius = radius;
  }

  void SetPlaneClip(const Float origin[3], const Float normal[3])
  {
    clip_mode = 2;
    plane_origin[0][0] = origin[0];
    plane_origin[0][1] = origin[1];
    plane_origin[0][2] = origin[2];
    plane_normal[0][0] = normal[0];
    plane_normal[0][1] = normal[1];
    plane_normal[0][2] = normal[2];
  }

  void Set2PlaneClip(const Float origin1[3],
                     const Float normal1[3],
                     const Float origin2[3],
                     const Float normal2[3])
  {
    clip_mode = 3;
    plane_origin[0][0] = origin1[0];
    plane_origin[0][1] = origin1[1];
    plane_origin[0][2] = origin1[2];
    plane_normal[0][0] = normal1[0];
    plane_normal[0][1] = normal1[1];
    plane_normal[0][2] = normal1[2];

    plane_origin[1][0] = origin2[0];
    plane_origin[1][1] = origin2[1];
    plane_origin[1][2] = origin2[2];
    plane_normal[1][0] = normal2[0];
    plane_normal[1][1] = normal2[1];
    plane_normal[1][2] = normal2[2];
  }

  void Set3PlaneClip(const Float origin1[3],
                     const Float normal1[3],
                     const Float origin2[3],
                     const Float normal2[3],
                     const Float origin3[3],
                     const Float normal3[3])
  {
    clip_mode = 4;
    plane_origin[0][0] = origin1[0];
    plane_origin[0][1] = origin1[1];
    plane_origin[0][2] = origin1[2];
    plane_normal[0][0] = normal1[0];
    plane_normal[0][1] = normal1[1];
    plane_normal[0][2] = normal1[2];

    plane_origin[1][0] = origin2[0];
    plane_origin[1][1] = origin2[1];
    plane_origin[1][2] = origin2[2];
    plane_normal[1][0] = normal2[0];
    plane_normal[1][1] = normal2[1];
    plane_normal[1][2] = normal2[2];

    plane_origin[2][0] = origin3[0];
    plane_origin[2][1] = origin3[1];
    plane_origin[2][2] = origin3[2];
    plane_normal[2][0] = normal3[0];
    plane_normal[2][1] = normal3[1];
    plane_normal[2][2] = normal3[2];
  }

  int
  num_passes(bool multipass) const
  {
    int passes = 1;
    if(clip_mode == 3 && multipass) // 2 planes
      passes = 2;
    else if(clip_mode == 4 && multipass) // 3 planes
      passes = 3;
    return passes;
  }

  std::shared_ptr<Field>
  make_box_distances(DataSet domain, bool invert) const
  {
    // TODO: Use the AABB<3> box to make distance field.
    std::shared_ptr<Field> retval;

    return retval;
  }

  std::shared_ptr<Field>
  make_sphere_distances(DataSet domain, bool invert) const
  {
    SphereDistance distcalc(sphere_center, sphere_radius);
    dispatch_3d(domain.mesh(), distcalc);
    std::shared_ptr<Field> retval = distcalc.m_output;
    return retval;
  }

  std::shared_ptr<Field>
  make_plane_distances(DataSet domain, bool invert, size_t plane_index) const
  {
    // TODO: Make a distance field for the plane surface.
    std::shared_ptr<Field> retval;
    return retval;
  }

  std::shared_ptr<Field>
  make_multi_plane_distances(DataSet domain, bool invert) const
  {
    // TODO: Make a field that combines plane distances for all planes,
    //       depending on clip_mode.
    std::shared_ptr<Field> retval;
    return retval;
  }

  std::shared_ptr<Field>
  make_distances(DataSet domain, bool invert, bool multipass, size_t pass = 0) const
  {
    std::shared_ptr<Field> f;
    if(clip_mode == 0)
      f = make_box_distances(domain, invert);
    else if(clip_mode == 1)
      f = make_sphere_distances(domain, invert);
    else if(clip_mode == 2)
      f = make_plane_distances(domain, invert, 0);
    else if(clip_mode == 3 || clip_mode == 4) // 2 or 3 planes
    {
      if(multipass)
        f = make_plane_distances(domain, invert, pass);
      else
        f = make_multi_plane_distances(domain, invert);
    }
    return f;
  }

public:
  AABB<3> boxbounds;
  Float sphere_center[3] = {0., 0., 0.};
  Float sphere_radius = 1.;
  Float plane_origin[3][3] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
  Float plane_normal[3][3] = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
  int clip_mode = 2;
};


Clip::Clip() : m_internals(std::make_shared<Clip::InternalsType>()),
  m_invert(false), m_do_multi_plane(false)
{
}

Clip::~Clip()
{
}

void
Clip::SetBoxClip(const AABB<3> &bounds)
{
  m_internals->SetBoxClip(bounds);
}

void
Clip::SetSphereClip(const Float center[3], const Float radius)
{
  m_internals->SetSphereClip(center, radius);
}

void
Clip::SetPlaneClip(const Float origin[3], const Float normal[3])
{
  m_internals->SetPlaneClip(origin, normal);
}

void
Clip::Set2PlaneClip(const Float origin1[3],
                    const Float normal1[3],
                    const Float origin2[3],
                    const Float normal2[3])
{
  m_internals->Set2PlaneClip(origin1, normal1, origin2, normal2);
}

void
Clip::Set3PlaneClip(const Float origin1[3],
                    const Float normal1[3],
                    const Float origin2[3],
                    const Float normal2[3],
                    const Float origin3[3],
                    const Float normal3[3])
{
  m_internals->Set3PlaneClip(origin1, normal1, origin2, normal2, origin3, normal3);
}

void
Clip::SetInvertClip(bool invert)
{
  m_invert = invert;
}

Collection
Clip::execute(Collection &collection)
{
  Collection res;

  // Compute distance field for all of the domains. 
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dom = collection.domain(i);
    if(dom.mesh() != nullptr)
    {
      std::string field_name("__dray_clip_field__");
      size_t npasses = m_internals->num_passes(m_do_multi_plane);
      DataSet input = dom, output;
      for(size_t pass = 0; pass < npasses; pass++)
      {
        // Make the clipping field and add it to the dataset.
        auto f = m_internals->make_distances(dom, m_do_multi_plane, pass);
        f->mesh_name(dom.mesh()->name());
        f->name(field_name);
        input.add_field(f);

        // Do the clipping pass on this single domain.
        ClipField clipper;
        clipper.set_clip_value(0.);
        clipper.set_field(field_name);
        clipper.set_invert_clip(m_invert);
        output = clipper.execute(input);

        // Remove the new field from the output and the input.
        if(input.has_field(field_name))
          input.remove_field(field_name);
        if(output.has_field(field_name))
          output.remove_field(field_name);

        // Prepare for the next pass.
        input = output;
      }

      // Add the clipped output to the collection.
      res.add_domain(output);       
    }
  }

  return res;
}

};//namespace dray
