#ifndef VTK_H_AUTOCAMERA_HPP
#define VTK_H_AUTOCAMERA_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/rendering/Camera.h>


namespace vtkh
{

typedef vtkm::rendering::Camera vtkmCamera;

class VTKH_API AutoCamera : public Filter
{
public:
  AutoCamera();
  virtual ~AutoCamera();

  std::string GetName() const override;
  std::string GetField();
  std::string GetMetric();
  int GetNumSamples();
  int GetNumBins();
  int GetHeight();
  int GetWidth();

  vtkmCamera GetCamera();

  void SetMetric(std::string metric);
  void SetField(std::string field);
  void SetNumSamples(int num_samples);
  void SetNumBins(int bins);
  void SetHeight(int height);
  void SetWidth(int width);
  
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  int m_bins;
  int m_height;
  int m_width;
  int m_samples;
  std::string m_field;
  std::string m_metric;
  vtkmCamera m_camera;
};

} //namespace vtkh
#endif
