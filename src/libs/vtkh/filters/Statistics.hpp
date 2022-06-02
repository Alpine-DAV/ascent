#ifndef VTK_H_STATISTICS_HPP
#define VTK_H_STATISTICS_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API Statistics
{
public:

  struct Result
  {
    vtkm::Float32 mean;
    vtkm::Float32 variance;
    vtkm::Float32 skewness;
    vtkm::Float32 kurtosis;
    void Print(std::ostream &out)
    {
      out<<"Mean    : "<<mean<<"\n";
      out<<"Variance: "<<variance<<"\n";
      out<<"Skewness: "<<skewness<<"\n";
      out<<"Kurtosis: "<<kurtosis<<"\n";
    }
  };

  Statistics();
  ~Statistics();
  Statistics::Result Run(vtkh::DataSet &data_set, const std::string field_name);

};

} //namespace vtkh
#endif
