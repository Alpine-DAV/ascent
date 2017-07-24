#ifndef VTK_H_FILTER_HPP
#define VTK_H_FILTER_HPP

#include "vtkh.hpp"
#include "vtkh_data_set.hpp"

namespace vtkh
{

class Filter
{
public:
  Filter() 
  { 
    m_input = nullptr; 
    m_output = nullptr; 
  }

  virtual ~Filter() { };
  void SetInput(DataSet *input) { m_input = input; }
  DataSet* GetOutput() { return m_output; }
  DataSet* Update()
  {
    PreExecute();
    DoExecute();
    PostExecute();
    return m_output;
  }

  void AddMapField(const std::string &field_name)
  {
    m_map_fields.push_back(field_name);
  }

  void ClearMapFields()
  {
    m_map_fields.clear();  
  }

protected:
  virtual void DoExecute() = 0;
  virtual void PreExecute() {};
  virtual void PostExecute() {};

  std::vector<std::string> m_map_fields;

  DataSet *m_input;
  DataSet *m_output;
};

} //namespace vtkh
#endif
