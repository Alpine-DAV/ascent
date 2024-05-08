#include "Threshold.hpp"
#include <vtkh/Error.hpp>
#include <vtkm/filter/entity_extraction/Threshold.h>
#include <vtkm/filter/entity_extraction/ExtractGeometry.h>
#include <vtkh/filters/CleanGrid.hpp>
#include <vtkm/ImplicitFunction.h>


//---------------------------------------------------------------------------//
namespace vtkh
{



//---------------------------------------------------------------------------//
struct
Threshold::Internals
{

  typedef enum
  {
      UNKNOWN,
      FIELD,
      BOX,
      PLANE,
      CYLINDER,
      SPHERE
  } Mode;

  int  m_mode;
  bool m_return_all_in_range;
  bool m_invert;
  bool m_boundary;

  // field case
  vtkm::Range m_field_range;
  std::string m_field_name;

  vtkm::ImplicitFunctionGeneral m_thresh_func;

  Internals():
  m_mode(Threshold::Internals::Mode::UNKNOWN),
  m_return_all_in_range(false),
  m_invert(false),
  m_boundary(false)
  {}
};

//---------------------------------------------------------------------------//
Threshold::Threshold()
: m_internals(new Internals)
{
}

//---------------------------------------------------------------------------//
Threshold::~Threshold()
{

}

//---------------------------------------------------------------------------//
std::string
Threshold::GetThresholdMode() const
{
  switch(m_internals->m_mode)
  {
    case Threshold::Internals::Mode::UNKNOWN:
      return "unknown";
    case Threshold::Internals::Mode::FIELD:
      return "field";
    case Threshold::Internals::Mode::BOX:
      return "box";
    case Threshold::Internals::Mode::PLANE:
      return "plane";
    case Threshold::Internals::Mode::CYLINDER:
      return "cylinder";
    case Threshold::Internals::Mode::SPHERE:
      return "sphere";
    default:
      return "unknown";
  }
}

//---------------------------------------------------------------------------//
void
Threshold::SetAllInRange(const bool &value)
{
  m_internals->m_return_all_in_range = value;
}


//---------------------------------------------------------------------------//
bool
Threshold::GetAllInRange() const
{
  return m_internals->m_return_all_in_range;
}

//---------------------------------------------------------------------------//
// threshold by field
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Threshold::SetFieldUpperThreshold(const double &value)
{
  m_internals->m_mode = Threshold::Internals::Mode::FIELD;
  m_internals->m_field_range.Max = value;
}

//---------------------------------------------------------------------------//
void
Threshold::SetFieldLowerThreshold(const double &value)
{
  m_internals->m_mode = Threshold::Internals::Mode::FIELD;
  m_internals->m_field_range.Min = value;
}

//---------------------------------------------------------------------------//
void
Threshold::SetField(const std::string &field_name)
{
  m_internals->m_mode = Threshold::Internals::Mode::FIELD;
  m_internals->m_field_name = field_name;
}

//---------------------------------------------------------------------------//
// invert/boundary
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// invert
//---------------------------------------------------------------------------//
void
Threshold::SetInvertThreshold(bool invert)
{
  m_internals->m_invert = invert;
}

//---------------------------------------------------------------------------//
// boundary
//---------------------------------------------------------------------------//
void
Threshold::SetBoundaryThreshold(bool boundary)
{
  m_internals->m_boundary = boundary;
}


//---------------------------------------------------------------------------//
// threshold by box
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Threshold::SetBoxThreshold(const vtkm::Bounds &box_bounds)
{
  m_internals->m_mode = Threshold::Internals::Mode::BOX;
  auto box = vtkm::Box({ box_bounds.X.Min,
                         box_bounds.Y.Min,
                         box_bounds.Z.Min},
                       { box_bounds.X.Max,
                         box_bounds.Y.Max,
                         box_bounds.Z.Max});

  m_internals->m_thresh_func = box;
}


//---------------------------------------------------------------------------//
// threshold by plane
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Threshold::SetPlaneThreshold(const double plane_origin[3],
                             const double plane_normal[3])
{
  m_internals->m_mode = Threshold::Internals::Mode::PLANE;
  vtkm::Vec<vtkm::FloatDefault,3> vec_origin;
  vec_origin[0] = plane_origin[0];
  vec_origin[1] = plane_origin[1];
  vec_origin[2] = plane_origin[2];

  vtkm::Vec<vtkm::FloatDefault,3> vec_normal;
  vec_normal[0] = plane_normal[0];
  vec_normal[1] = plane_normal[1];
  vec_normal[2] = plane_normal[2];

  auto plane = vtkm::Plane(vec_origin, vec_normal);
  m_internals->m_thresh_func = plane;
}

//---------------------------------------------------------------------------//
// threshold by cylinder
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Threshold::SetCylinderThreshold(const double cylinder_center[3],
                                const double cylinder_axis[3],
                                const double cylinder_radius)
{
  m_internals->m_mode = Threshold::Internals::Mode::CYLINDER;
  vtkm::Vec<vtkm::FloatDefault,3> vec_center;
  vec_center[0] = cylinder_center[0];
  vec_center[1] = cylinder_center[1];
  vec_center[2] = cylinder_center[2];

  vtkm::Vec<vtkm::FloatDefault,3> vec_axis;
  vec_axis[0] = cylinder_axis[0];
  vec_axis[1] = cylinder_axis[1];
  vec_axis[2] = cylinder_axis[2];

  vtkm::FloatDefault r = cylinder_radius;

  auto cylinder = vtkm::Cylinder(vec_center, vec_axis, r);
  m_internals->m_thresh_func = cylinder;
}

//---------------------------------------------------------------------------//
// threshold by Sphere
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Threshold::SetSphereThreshold(const double sphere_center[3],
                              const double sphere_radius)

{
  m_internals->m_mode = Threshold::Internals::Mode::SPHERE;
  vtkm::Vec<vtkm::FloatDefault,3> vec_center;
  vec_center[0] = sphere_center[0];
  vec_center[1] = sphere_center[1];
  vec_center[2] = sphere_center[2];
  vtkm::FloatDefault r = sphere_radius;

  auto sphere = vtkm::Sphere(vec_center, r);
  m_internals->m_thresh_func = sphere;
}

//---------------------------------------------------------------------------//
void
Threshold::PreExecute()
{
  Filter::PreExecute();

  if(m_internals->m_mode == Threshold::Internals::Mode::UNKNOWN)
  { 
    // error!
  }

  if(m_internals->m_mode == Threshold::Internals::Mode::FIELD)
  {
    Filter::CheckForRequiredField(m_internals->m_field_name);
  }
}



//---------------------------------------------------------------------------//
void
Threshold::PostExecute()
{
  Filter::PostExecute();
}

//---------------------------------------------------------------------------//
void
Threshold::DoExecute()
{
  DataSet temp_data;
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    // field case
    if(m_internals->m_mode == Threshold::Internals::Mode::FIELD)
    {
      if(!dom.HasField(m_internals->m_field_name))
      {
        continue;
      }

      vtkm::filter::entity_extraction::Threshold thresholder;
      
      if(m_internals->m_invert)
      {
        thresholder.SetInvert(true);
      }
      else
      {
        thresholder.SetInvert(false);
      }

      thresholder.SetAllInRange(m_internals->m_return_all_in_range);
      thresholder.SetUpperThreshold(m_internals->m_field_range.Max);
      thresholder.SetLowerThreshold(m_internals->m_field_range.Min);
      thresholder.SetActiveField(m_internals->m_field_name);
      thresholder.SetFieldsToPass(this->GetFieldSelection());
      auto data_set = thresholder.Execute(dom);
      temp_data.AddDomain(data_set, domain_id);
    }
    else
    {
      // use implicit function w/ entity extractor
      vtkm::filter::entity_extraction::ExtractGeometry extractor;
      if(m_internals->m_invert)
      {
        extractor.SetExtractInside(false);
      }
      else
      {
        extractor.SetExtractInside(true);
      }

      if(m_internals->m_boundary)
      {
        extractor.SetExtractBoundaryCells(true);
      }
      else
      {
        extractor.SetExtractBoundaryCells(false);
      }

      extractor.SetImplicitFunction(m_internals->m_thresh_func);
      extractor.SetFieldsToPass(this->GetFieldSelection());
      auto data_set = extractor.Execute(dom);
      temp_data.AddDomain(data_set, domain_id);
    }
  }

  CleanGrid cleaner;
  cleaner.SetInput(&temp_data);
  cleaner.Update();
  this->m_output = cleaner.GetOutput();

}

//---------------------------------------------------------------------------//
std::string
Threshold::GetName() const
{
  return "vtkh::Threshold";
}

} //  namespace vtkh
