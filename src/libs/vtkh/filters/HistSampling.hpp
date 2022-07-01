#ifndef VTK_H_HIST_SAMPLING_HPP
#define VTK_H_HIST_SAMPLING_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API HistSampling : public Filter
{
public:
  HistSampling();
  virtual ~HistSampling();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetNumBins(const int num_bins);
  void SetGhostField(const std::string &field_name);
  std::string GetField() const;
  void SetSamplingPercent(const float percent);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  std::string m_ghost_field;
  float m_sample_percent;
  int m_num_bins;
};

} //namespace vtkh
#endif
