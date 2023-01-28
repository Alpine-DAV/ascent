#ifndef VTK_H_HISTOGRAM_HPP
#define VTK_H_HISTOGRAM_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

#include <vector>
#include <iostream>

namespace vtkh
{

class VTKH_API Histogram : public Filter
{
public:
  Histogram();
  virtual ~Histogram();

  std::string GetName() const override;
  void SetRange(const vtkm::Range &range);
  void SetNumBins(const int num_bins);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  
  int m_num_bins;
  vtkm::Range m_range;
};

} //namespace vtkh
#endif
