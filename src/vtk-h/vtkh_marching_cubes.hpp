#ifndef VTK_H_MARCHING_CUBES_HPP
#define VTK_H_MARCHING_CUBES_HPP

#include "vtkh.hpp"
#include "vtkh_filter.hpp"
#include "vtkh_data_set.hpp"

namespace vtkh
{

class MarchingCubes : public Filter
{
public:
  MarchingCubes(); 
  virtual ~MarchingCubes(); 
  void SetIsoValue(const double &iso_value);
  void SetIsoValues(const double *iso_values, const int &num_values);
  void AddMapField(const std::string &field_name);
  void ClearMapFields();
  void SetField(const std::string &field_name);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  bool ContainsIsoValues(vtkm::cont::DataSet &dom);
  std::vector<std::string> m_map_fields;
  std::vector<double> m_iso_values;
  std::string m_field_name;
};

} //namespace vtkh
#endif
