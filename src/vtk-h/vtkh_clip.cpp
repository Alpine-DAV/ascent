#include <vtkh_clip.hpp>

#include <vtkm/filter/ClipWithImplicitFunction.h>
#include <vtkm/filter/ClipWithImplicitFunction.h>

namespace vtkh 
{

struct Clip::InternalsType
{
  vtkm::filter::ClipWithImplicitFunction m_clipper; 
  InternalsType()
  {}
};

Clip::Clip()
  : m_internals(new InternalsType)
{

}

Clip::~Clip()
{

}

void 
Clip::SetBoxClip(const vtkm::Bounds &clipping_bounds)
{
  auto box = std::make_shared<vtkm::cont::Box>(clipping_bounds.X.Min,
                                               clipping_bounds.X.Max,
                                               clipping_bounds.Y.Min,
                                               clipping_bounds.Y.Max,
                                               clipping_bounds.Z.Min,
                                               clipping_bounds.Z.Max);

  m_internals->m_clipper.SetImplicitFunction(box);
}

void 
Clip::SetSphereClip(const double center[3], const double radius)
{
  vtkm::Vec<vtkm::FloatDefault,3> vec_center;
  vec_center[0] = center[0];
  vec_center[1] = center[1];
  vec_center[2] = center[2];
  vtkm::FloatDefault r = radius;

  std::cout<<"****center "<<vec_center<<" radius "<<r<<"\n";
  auto sphere = std::make_shared<vtkm::cont::Sphere>(vec_center, r);
  m_internals->m_clipper.SetImplicitFunction(sphere);
}

void 
Clip::SetPlaneClip(const double origin[3], const double normal[3]) 
{
  vtkm::Vec<vtkm::FloatDefault,3> vec_origin;
  vec_origin[0] = origin[0];
  vec_origin[1] = origin[1];
  vec_origin[2] = origin[2];

  vtkm::Vec<vtkm::FloatDefault,3> vec_normal;
  vec_normal[0] = normal[0];
  vec_normal[1] = normal[1];
  vec_normal[2] = normal[2];

  auto plane = std::make_shared<vtkm::cont::Plane>(vec_origin, vec_normal);
  m_internals->m_clipper.SetImplicitFunction(plane);
}

void 
Clip::AddMapField(const std::string &field_name)
{
  m_map_fields.push_back(field_name);
}

void 
Clip::ClearMapFields()
{
  m_map_fields.clear();
}

void Clip::PreExecute() 
{

}

void Clip::PostExecute()
{

}

void Clip::DoExecute()
{
  
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains(); 

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkm::filter::ResultDataSet res = m_internals->m_clipper.Execute(dom);

    for(size_t f = 0; f < m_map_fields.size(); ++f)
    {
      m_internals->m_clipper.MapFieldOntoOutput(res, dom.GetField(m_map_fields[f]));
    }
    this->m_output->AddDomain(res.GetDataSet(), domain_id);
    
  }
}

} //  namespace vtkh
