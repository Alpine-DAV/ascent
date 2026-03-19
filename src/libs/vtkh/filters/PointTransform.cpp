#include <vtkh/filters/PointTransform.hpp>
#include <viskores/Transform3D.h>
#include <vtkh/viskores_filters/viskoresPointTransform.hpp>

namespace vtkh
{

//---------------------------------------------------------------------------//
PointTransform::PointTransform()
{
  ResetTransform();
}

//---------------------------------------------------------------------------//
PointTransform::~PointTransform()
{

}

//---------------------------------------------------------------------------//
void
PointTransform::ResetTransform()
{
  viskores::MatrixIdentity(m_transform);
}

//---------------------------------------------------------------------------//
void
PointTransform::SetTranslation(const double& tx,
                               const double& ty,
                               const double& tz)
{
  viskores::Matrix<double,4,4> matrix = viskores::Transform3DTranslate(tx, ty, tz);
  m_transform = viskores::MatrixMultiply(m_transform, matrix);
}

//---------------------------------------------------------------------------//
void
PointTransform::SetRotation(const double& angleDegrees,
                            const double& axisX,
                            const double& axisY,
                            const double& axisZ)
{
  viskores::Matrix<double,4,4> matrix = viskores::Transform3DRotate(angleDegrees,
                                                            axisX,
                                                            axisY,
                                                            axisZ);
  m_transform = viskores::MatrixMultiply(m_transform, matrix);
}

//---------------------------------------------------------------------------//
void
PointTransform::SetTransform(const double *matrix_values)
{
  // Note: row vs col vs matrix vs array, what is the best
  //       order for users to provide flat values to matrix?
  //       This order is decent, was able to throw values in
  //       from example matrices in the wild easily.

  m_transform[0][0] = matrix_values[0];
  m_transform[0][1] = matrix_values[1];
  m_transform[0][2] = matrix_values[2];
  m_transform[0][3] = matrix_values[3];

  m_transform[1][0] = matrix_values[4];
  m_transform[1][1] = matrix_values[5];
  m_transform[1][2] = matrix_values[6];
  m_transform[1][3] = matrix_values[7];

  m_transform[2][0] = matrix_values[8];
  m_transform[2][1] = matrix_values[9];
  m_transform[2][2] = matrix_values[10];
  m_transform[2][3] = matrix_values[11];

  m_transform[3][0] = matrix_values[12];
  m_transform[3][1] = matrix_values[13];
  m_transform[3][2] = matrix_values[14];
  m_transform[3][3] = matrix_values[15];
}


//---------------------------------------------------------------------------//
void
PointTransform::SetTransform(const viskores::Matrix<double, 4, 4>& mtx)
{
  m_transform = mtx;
}

//---------------------------------------------------------------------------//
void
PointTransform::SetScale(const double& sx,
                         const double& sy,
                         const double& sz)
{
  viskores::Matrix<double,4,4> matrix = viskores::Transform3DScale(sx, sy, sz);
  m_transform = viskores::MatrixMultiply(m_transform, matrix);
}

//---------------------------------------------------------------------------//
void
PointTransform::SetReflect(const double& pointX,
                           const double& pointY,
                           const double& pointZ,
                           const double& normalX,
                           const double& normalY,
                           const double& normalZ)
{
  // reflect across plane defined by point(p) + normal(n):
  // M = (I - 2 n n^T) with translation 2 (n·p) n
  std::cerr <<" normal: " << normalX << " " << normalY << " " << normalZ << std::endl;
  std::cerr <<" point: " << pointX << " " << pointY << " " << pointZ << std::endl;

  viskores::Vec<viskores::Float64,3> normal;
  normal[0] = normalX;
  normal[1] = normalY;
  normal[2] = normalZ;
  viskores::Normalize(normal);

  viskores::Vec<viskores::Float64,3> point;
  point[0] = pointX;
  point[1] = pointY;
  point[2] = pointZ;

  // build outer product n n^T in homogeneous form
  viskores::Matrix<double,4,1> m_n;
  m_n[0] = normal[0];
  m_n[1] = normal[1];
  m_n[2] = normal[2];
  m_n[3] = 0.0;

  viskores::Matrix<double,1,4> m_nt   = viskores::MatrixTranspose(m_n);
  viskores::Matrix<double,4,4> matrix = viskores::MatrixMultiply(m_n, m_nt);

  // apply (I - 2 n n^T)
  matrix[0][0] = 1.0 - 2.0 * matrix[0][0];
  matrix[0][1] =     - 2.0 * matrix[0][1];
  matrix[0][2] =     - 2.0 * matrix[0][2];

  matrix[1][0] =     - 2.0 * matrix[1][0];
  matrix[1][1] = 1.0 - 2.0 * matrix[1][1];
  matrix[1][2] =     - 2.0 * matrix[1][2];

  matrix[2][0] =     - 2.0 * matrix[2][0];
  matrix[2][1] =     - 2.0 * matrix[2][1];
  matrix[2][2] = 1.0 - 2.0 * matrix[2][2];

  // compute translation: t = 2 (n · p) n
  double dot = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2];

  matrix[0][3] = 2.0 * dot * normal[0];
  matrix[1][3] = 2.0 * dot * normal[1];
  matrix[2][3] = 2.0 * dot * normal[2];

  // homogeneous row
  matrix[3][0] = 0.0;
  matrix[3][1] = 0.0;
  matrix[3][2] = 0.0;
  matrix[3][3] = 1.0;

  m_transform = matrix;
}

//---------------------------------------------------------------------------//
void
PointTransform::PreExecute()
{
  Filter::PreExecute();
}

//---------------------------------------------------------------------------//
void
PointTransform::PostExecute()
{
  Filter::PostExecute();
}

//---------------------------------------------------------------------------//
void
PointTransform::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();
  auto bounds = this->m_input->GetGlobalBounds();
  std::cerr << "input global bounds: " << bounds.X.Min << " " << bounds.X.Max << " , " << bounds.Y.Min << " " << bounds.Y.Max << " , " << bounds.Z.Min << " " << bounds.Z.Max << std::endl;

  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    viskoresPointTransform transformer;
    auto dataset = transformer.Run(dom,
                                   m_transform,
                                   this->GetFieldSelection());
    // insert interesting stuff
    m_output->AddDomain(dataset, domain_id);
  }
  auto out_bounds = m_output->GetGlobalBounds();
  std::cerr << "output global out_bounds: " << out_bounds.X.Min << " " << out_bounds.X.Max << " , " << out_bounds.Y.Min << " " << out_bounds.Y.Max << " , " << out_bounds.Z.Min << " " << out_bounds.Z.Max << std::endl;
}

//---------------------------------------------------------------------------//
std::string
PointTransform::GetName() const
{
  return "vtkh::PointTransform";
}

} //  namespace vtkh
