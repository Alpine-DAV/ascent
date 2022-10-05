#include "vtkmLagrangian.hpp"

#include <vtkm/filter/flow/Lagrangian.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmLagrangian::Run(vtkm::cont::DataSet &input,
                         std::string field_name,
                         double step_size,
                         int write_frequency,
                         int rank,
                         int cust_res,
                         int x_res,
                         int y_res,
                         int z_res)
{
#ifdef VTKH_BYPASS_VTKM_BIH
  return vtkm::cont::DataSet();
#else
  vtkm::filter::flow::Lagrangian lagrangianFilter;
  lagrangianFilter.SetStepSize(step_size);
  lagrangianFilter.SetWriteFrequency(write_frequency);
  lagrangianFilter.SetRank(rank);
  lagrangianFilter.SetActiveField(field_name);
  lagrangianFilter.SetCustomSeedResolution(cust_res);
  lagrangianFilter.SetSeedResolutionInX(x_res);
  lagrangianFilter.SetSeedResolutionInY(y_res);
  lagrangianFilter.SetSeedResolutionInZ(z_res);

  auto output = lagrangianFilter.Execute(input);
  return output;
#endif
}

} // namespace vtkh
