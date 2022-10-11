#ifndef VTK_H_VTKM_LAGRANGIAN_HPP
#define VTK_H_VTKM_LAGRANGIAN_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class vtkmLagrangian
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::string field_name,
                          double step_size,
                          int write_frequency,
                          int rank,
                          int cust_res,
                          int x_res,
                          int y_res,
                          int z_res);
};
}
#endif
