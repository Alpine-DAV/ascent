#include "Clip.hpp"

#include <vtkh/filters/CleanGrid.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/vtkm_filters/vtkmClip.hpp>
#include <vtkm/ImplicitFunction.h>

#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>

namespace vtkh
{

namespace detail
{

class MultiPlane : public vtkm::internal::ImplicitFunctionBase<MultiPlane>
{
public:
  MultiPlane() = default;

  VTKM_EXEC_CONT MultiPlane(const Vector points[3],
                            const Vector normals[3],
                            const int num_planes)
  {
    this->SetPlanes(points, normals);
    this->m_num_planes = num_planes;
  }

  VTKM_EXEC void SetPlanes(const Vector points[6], const Vector normals[6])
  {
    for (vtkm::Id index : { 0, 1, 2})
    {
      this->Points[index] = points[index];
    }
    for (vtkm::Id index : { 0, 1, 2})
    {
      this->Normals[index] = normals[index];
    }
  }

  VTKM_EXEC void SetPlane(int idx, const Vector& point, const Vector& normal)
  {
    VTKM_ASSERT((idx >= 0) && (idx < 3));
    this->Points[idx] = point;
    this->Normals[idx] = normal;
  }

  VTKM_EXEC_CONT void SetNumPlanes(const int &num)
  {
    this->m_num_planes = num;
  }

  VTKM_EXEC_CONT void GetPlanes(Vector points[3], Vector normals[3]) const
  {
    for (vtkm::Id index : { 0, 1, 2})
    {
      points[index] = this->Points[index];
    }
    for (vtkm::Id index : { 0, 1, 2})
    {
      normals[index] = this->Normals[index];
    }
  }

  VTKM_EXEC_CONT const Vector* GetPoints() const { return this->Points; }

  VTKM_EXEC_CONT const Vector* GetNormals() const { return this->Normals; }

  VTKM_EXEC_CONT Scalar Value(const Vector& point) const
  {
    Scalar maxVal = vtkm::NegativeInfinity<Scalar>();
    for (vtkm::Id index = 0; index < this->m_num_planes; ++index)
    {
      const Vector& p = this->Points[index];
      const Vector& n = this->Normals[index];
      const Scalar val = vtkm::Dot(point - p, n);
      maxVal = vtkm::Max(maxVal, val);
    }
    return maxVal;
  }

  VTKM_EXEC_CONT Vector Gradient(const Vector& point) const
  {
    Scalar maxVal = vtkm::NegativeInfinity<Scalar>();
    vtkm::Id maxValIdx = 0;
    for (vtkm::Id index = 0; index < this->m_num_planes; ++index)
    {
      const Vector& p = this->Points[index];
      const Vector& n = this->Normals[index];
      Scalar val = vtkm::Dot(point - p, n);
      if (val > maxVal)
      {
        maxVal = val;
        maxValIdx = index;
      }
    }
    return this->Normals[maxValIdx];
  }

private:
  Vector Points[6] = { { -0.0f, 0.0f, 0.0f },
                       { 0.0f, 0.0f, 0.0f },
                       { 0.0f, -0.0f, 0.0f }};
  Vector Normals[6] = { { -1.0f, 0.0f, 0.0f },
                        { 1.0f, 0.0f, 0.0f },
                        { 0.0f, 0.0f, 0.0f } };
  int m_num_planes = 3;
};

class MultiPlaneField : public vtkm::worklet::WorkletMapField
{
protected:
  MultiPlane m_multi_plane;
public:
  VTKM_CONT
  MultiPlaneField(MultiPlane &multi_plane)
    : m_multi_plane(multi_plane)
  {
  }

  typedef void ControlSignature(FieldIn, FieldOut);
  typedef void ExecutionSignature(_1, _2);

  template<typename T>
  VTKM_EXEC
  void operator()(const vtkm::Vec<T,3> &point, vtkm::Float32& distance) const
  {
    distance = m_multi_plane.Value(point);
  }
}; //class SliceField

}// namespace detail

struct Clip::InternalsType
{
  vtkm::ImplicitFunctionGeneral m_func;
  detail::MultiPlane m_multi_plane;
  InternalsType()
  {}
};

Clip::Clip()
  : m_internals(new InternalsType),
    m_invert(false),
    m_do_multi_plane(false)
{

}

Clip::~Clip()
{

}

void
Clip::SetInvertClip(bool invert)
{
  m_invert = invert;
}

void
Clip::SetBoxClip(const vtkm::Bounds &clipping_bounds)
{
  m_do_multi_plane = false;
  auto box = vtkm::Box({ clipping_bounds.X.Min,
                         clipping_bounds.Y.Min,
                         clipping_bounds.Z.Min},
                       { clipping_bounds.X.Max,
                         clipping_bounds.Y.Max,
                         clipping_bounds.Z.Max});


  m_internals->m_func = box;
}

void
Clip::SetSphereClip(const double center[3], const double radius)
{
  m_do_multi_plane = false;
  vtkm::Vec<vtkm::FloatDefault,3> vec_center;
  vec_center[0] = center[0];
  vec_center[1] = center[1];
  vec_center[2] = center[2];
  vtkm::FloatDefault r = radius;

  auto sphere = vtkm::Sphere(vec_center, r);
  m_internals->m_func = sphere;
}

void
Clip::SetPlaneClip(const double origin[3], const double normal[3])
{
  m_do_multi_plane = false;
  vtkm::Vec<vtkm::FloatDefault,3> vec_origin;
  vec_origin[0] = origin[0];
  vec_origin[1] = origin[1];
  vec_origin[2] = origin[2];

  vtkm::Vec<vtkm::FloatDefault,3> vec_normal;
  vec_normal[0] = normal[0];
  vec_normal[1] = normal[1];
  vec_normal[2] = normal[2];

  auto plane = vtkm::Plane(vec_origin, vec_normal);
  m_internals->m_func = plane;
}

void
Clip::Set2PlaneClip(const double origin1[3],
                    const double normal1[3],
                    const double origin2[3],
                    const double normal2[3])
{
  m_do_multi_plane = true;
  vtkm::Vec3f plane_points[3];
  plane_points[0][0] = float(origin1[0]);
  plane_points[0][1] = float(origin1[1]);
  plane_points[0][2] = float(origin1[2]);

  plane_points[1][0] = float(origin2[0]);
  plane_points[1][1] = float(origin2[1]);
  plane_points[1][2] = float(origin2[2]);

  plane_points[2][0] = 0.f;
  plane_points[2][1] = 0.f;
  plane_points[2][2] = 0.f;

  vtkm::Vec3f plane_normals[3];
  plane_normals[0][0] = float(normal1[0]);
  plane_normals[0][1] = float(normal1[1]);
  plane_normals[0][2] = float(normal1[2]);

  plane_normals[1][0] = float(normal2[0]);
  plane_normals[1][1] = float(normal2[1]);
  plane_normals[1][2] = float(normal2[2]);

  plane_normals[2][0] = 0.f;
  plane_normals[2][1] = 0.f;
  plane_normals[2][2] = 0.f;

  vtkm::Normalize(plane_normals[0]);
  vtkm::Normalize(plane_normals[1]);

  auto planes
    = detail::MultiPlane(plane_points, plane_normals, 2);
  m_internals->m_multi_plane = planes;
}

void
Clip::Set3PlaneClip(const double origin1[3],
                    const double normal1[3],
                    const double origin2[3],
                    const double normal2[3],
                    const double origin3[3],
                    const double normal3[3])
{
  m_do_multi_plane = true;
  vtkm::Vec3f plane_points[3];
  plane_points[0][0] = float(origin1[0]);
  plane_points[0][1] = float(origin1[1]);
  plane_points[0][2] = float(origin1[2]);

  plane_points[1][0] = float(origin2[0]);
  plane_points[1][1] = float(origin2[1]);
  plane_points[1][2] = float(origin2[2]);

  plane_points[2][0] = float(origin3[0]);
  plane_points[2][1] = float(origin3[1]);
  plane_points[2][2] = float(origin3[2]);

  vtkm::Vec3f plane_normals[3];
  plane_normals[0][0] = float(normal1[0]);
  plane_normals[0][1] = float(normal1[1]);
  plane_normals[0][2] = float(normal1[2]);

  plane_normals[1][0] = float(normal2[0]);
  plane_normals[1][1] = float(normal2[1]);
  plane_normals[1][2] = float(normal2[2]);

  plane_normals[2][0] = float(normal3[0]);
  plane_normals[2][1] = float(normal3[1]);
  plane_normals[2][2] = float(normal3[2]);

  vtkm::Normalize(plane_normals[0]);
  vtkm::Normalize(plane_normals[1]);
  vtkm::Normalize(plane_normals[2]);

  auto planes
    = detail::MultiPlane(plane_points, plane_normals, 3);
  m_internals->m_multi_plane = planes;
}

void Clip::PreExecute()
{
  Filter::PreExecute();
}

void Clip::PostExecute()
{
  Filter::PostExecute();
}

void Clip::DoExecute()
{

  DataSet data_set;
  const int global_domains = this->m_input->GetGlobalNumberOfDomains();
  if(global_domains == 0)
  {
    // if the number of domains zero there is no work to do,
    // additionally, a multiplane clip will fail since it will
    // check if 'mclip_field' will exist, which it wont.
    DataSet *output = new DataSet();
    *output = *(this->m_input);
    this->m_output = output;
    return;
  }
  const int num_domains = this->m_input->GetNumberOfDomains();
  // we now have to work around this since
  // vtkm dropped support for new implicit functions
  if(m_do_multi_plane)
  {

    const std::string fname = "mclip_field";
    // shallow copy the input so we don't propagate the field
    // to the input data set, since it might be used in other places
    vtkh::DataSet temp_ds = *(this->m_input);
    for(int i = 0; i < num_domains; ++i)
    {
      vtkm::cont::DataSet &dom = temp_ds.GetDomain(i);

      vtkm::cont::ArrayHandle<vtkm::Float32> clip_field;
      vtkm::worklet::DispatcherMapField<detail::MultiPlaneField>(detail::MultiPlaneField(m_internals->m_multi_plane))
        .Invoke(dom.GetCoordinateSystem().GetData(), clip_field);

      dom.AddField(vtkm::cont::Field(fname,
                                     vtkm::cont::Field::Association::POINTS,
                                     clip_field));
    } // each domain

    vtkm::Range range;
    range.Include(0.);
    if(m_invert)
    {
      range.Include(vtkm::NegativeInfinity64());
    }
    else
    {
      range.Include(vtkm::Infinity64());
    }
    vtkh::IsoVolume isovolume;
    isovolume.SetInput(&temp_ds);
    isovolume.SetRange(range);
    isovolume.SetField(fname);
    isovolume.Update();
    vtkh::DataSet *temp_out = isovolume.GetOutput();
    data_set = *temp_out;
    delete temp_out;

  }
  else
  {
    for(int i = 0; i < num_domains; ++i)
    {
      vtkm::Id domain_id;
      vtkm::cont::DataSet dom;
      this->m_input->GetDomain(i, dom, domain_id);

      vtkh::vtkmClip clipper;

      auto dataset = clipper.Run(dom,
                                 m_internals->m_func,
                                 m_invert,
                                 this->GetFieldSelection());

      data_set.AddDomain(dataset, domain_id);
    }
  }

  CleanGrid cleaner;
  cleaner.SetInput(&data_set);
  cleaner.Update();
  this->m_output = cleaner.GetOutput();
}

std::string
Clip::GetName() const
{
  return "vtkh::Clip";
}

} //  namespace vtkh
