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
PointTransform::SetReflect(const double& axisX,
                           const double& axisY,
                           const double& axisZ)
{
    // reflect recipe:
    // identify - 2*(normal) * (normal)^T

    viskores::Vec<viskores::Float64,3> axis;
    axis[0] = axisX;
    axis[1] = axisY;
    axis[2] = axisZ;
    viskores::Normalize(axis);

    viskores::Matrix<double,4,1> m_n;
    m_n[0] = axis[0];
    m_n[1] = axis[1];
    m_n[2] = axis[2];
    m_n[3] = 0.0;

    viskores::Matrix<double,1,4> m_nt   = viskores::MatrixTranspose(m_n);
    viskores::Matrix<double,4,4> matrix = viskores::MatrixMultiply(m_n, m_nt);

    matrix[0][0] = 1.0 - 2.0 * matrix[0][0];
    matrix[0][1] =     - 2.0 * matrix[0][1];
    matrix[0][2] =     - 2.0 * matrix[0][2];
    matrix[0][3] =     - 2.0 * matrix[0][3];

    matrix[1][0] =     - 2.0 * matrix[1][0];
    matrix[1][1] = 1.0 - 2.0 * matrix[1][1];
    matrix[1][2] =     - 2.0 * matrix[1][2];
    matrix[1][3] =     - 2.0 * matrix[1][3];

    matrix[2][0] =     - 2.0 * matrix[2][0];
    matrix[2][1] =     - 2.0 * matrix[2][1];
    matrix[2][2] = 1.0 - 2.0 * matrix[2][2];
    matrix[2][3] =     - 2.0 * matrix[2][3];

    matrix[3][0] =     - 2.0 * matrix[3][0];
    matrix[3][1] =     - 2.0 * matrix[3][1];
    matrix[3][2] =     - 2.0 * matrix[3][2];
    matrix[3][3] =     - 2.0 * matrix[3][3];

    // to view/debug the matrix:
    // std::cout << matrix;
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
}

//---------------------------------------------------------------------------//
std::string
PointTransform::GetName() const
{
  return "vtkh::PointTransform";
}

} //  namespace vtkh
