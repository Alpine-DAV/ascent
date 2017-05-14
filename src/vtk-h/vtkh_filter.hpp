#ifndef VTK_H_FILTER_HPP
#define VTK_H_FILTER_HPP

#include "vtkh.hpp"
#include "vtkh_data_set.hpp"

namespace vtkh
{

class vtkhFilter
{
public:
  void SetInput(vtkhDataSet *input) { m_input = input; }
  vtkhDataSet* Update();
protected:
  virtual void DoExecute() = 0;
  virtual void PreExecute() {};
  virtual void PostExecute() {};

  vtkhDataSet *m_input;
};

} //namespace vtkh
#endif
