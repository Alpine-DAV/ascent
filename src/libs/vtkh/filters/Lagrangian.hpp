#ifndef VTK_H_LAGRANGIAN_HPP
#define VTK_H_LAGRANGIAN_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API Lagrangian : public Filter
{
public:
  Lagrangian();
  virtual ~Lagrangian();
  std::string GetName() const override;
	void SetField(const std::string &field_name);
  void SetStepSize(const double &step_size);
  void SetWriteFrequency(const int &write_frequency);
	void SetCustomSeedResolution(const int &cust_res);
	void SetSeedResolutionInX(const int &x_res);
	void SetSeedResolutionInY(const int &y_res);
	void SetSeedResolutionInZ(const int &z_res);


protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
	double m_step_size;
	int m_write_frequency;
	int m_cust_res;
	int m_x_res, m_y_res, m_z_res;
};

} //namespace vtkh
#endif
