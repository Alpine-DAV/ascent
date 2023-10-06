#ifndef VTK_H_WARPXSTREAMLINE_HPP
#define VTK_H_WARPXSTREAMLINE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/filter/flow/WarpXStreamline.h>
#include <vtkm/Particle.h>

namespace vtkh
{

class VTKH_API WarpXStreamline : public Filter
{
public:
  WarpXStreamline();
  virtual ~WarpXStreamline();
  std::string GetName() const override;
  void SetBField(const std::string &Bfield_name);
  void SetEField(const std::string &Efield_name);
  void SetSteps(const double &steps);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  double m_steps;
  double m_length;
};

} //namespace vtkh
#endif
