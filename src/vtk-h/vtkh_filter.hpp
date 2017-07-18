#ifndef VTK_H_FILTER_HPP
#define VTK_H_FILTER_HPP

#include "vtkh.hpp"
#include "vtkh_data_set.hpp"

namespace vtkh
{

class vtkhFilter
{
public:
  vtkhFilter() 
  { 
    m_input = nullptr; 
    m_output = nullptr; 
  };
  virtual ~vtkhFilter() { };
  void SetInput(DataSet *input) { m_input = input; }
  DataSet* GetOutput() { return m_output; }
  DataSet* Update()
  {
    PreExecute();
    DoExecute();
    PostExecute();
    return m_output;
  }
protected:
  virtual void DoExecute() = 0;
  virtual void PreExecute() {};
  virtual void PostExecute() {};

  DataSet *m_input;
  DataSet *m_output;
};

} //namespace vtkh
#endif
