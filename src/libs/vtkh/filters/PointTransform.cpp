#include <vtkh/filters/PointTransform.hpp>
#include <vtkm/Transform3D.h>
#include <vtkh/vtkm_filters/vtkmPointTransform.hpp>

namespace vtkh
{

PointTransform::PointTransform()
{
  ResetTransform();
}

PointTransform::~PointTransform()
{

}

void PointTransform::ResetTransform()
{
  vtkm::MatrixIdentity(m_transform);
}

void
PointTransform::SetTranslation(const double& tx,
                               const double& ty,
                               const double& tz)
{
  vtkm::Matrix<double,4,4> matrix  = vtkm::Transform3DTranslate(tx, ty, tz);
  m_transform = vtkm::MatrixMultiply(m_transform, matrix);
}

void PointTransform::SetRotation(const double& angleDegrees,
                                 const vtkm::Vec<double, 3>& axis)
{
  vtkm::Matrix<double,4,4> matrix = vtkm::Transform3DRotate(angleDegrees, axis);
  m_transform = vtkm::MatrixMultiply(m_transform, matrix);
}

void PointTransform::SetTransform(const vtkm::Matrix<double, 4, 4>& mtx)
{
  m_transform = mtx;
}

void PointTransform::SetScale(const double& sx,
                              const double& sy,
                              const double& sz)
{
  vtkm::Matrix<double,4,4> matrix = vtkm::Transform3DScale(sx, sy, sz);
  m_transform = vtkm::MatrixMultiply(m_transform, matrix);
}

void PointTransform::PreExecute()
{
  Filter::PreExecute();
}

void PointTransform::PostExecute()
{
  Filter::PostExecute();
}

void PointTransform::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    vtkmPointTransform transformer;
    auto dataset = transformer.Run(dom,
                                   m_transform,
                                   this->GetFieldSelection());
    // insert interesting stuff
    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
PointTransform::GetName() const
{
  return "vtkh::PointTransform";
}

} //  namespace vtkh
