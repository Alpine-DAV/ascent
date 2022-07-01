#ifndef VTK_H_FILTER_HPP
#define VTK_H_FILTER_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class VTKH_API Filter
{
public:
  Filter();
  virtual ~Filter();
  virtual void SetInput(DataSet *input);
  virtual std::string GetName() const = 0;

  DataSet* GetOutput();
  DataSet* Update();

  void AddMapField(const std::string &field_name);

  void ClearMapFields();

protected:
  virtual void DoExecute() = 0;
  virtual void PreExecute();
  virtual void PostExecute();

  //@{
  /// These are all temporary methods added to gets things building again
  /// while we totally deprecate vtk-h compnents
  ///
  vtkm::filter::FieldSelection GetFieldSelection() const;
  //@}

  std::vector<std::string> m_map_fields;

  DataSet *m_input;
  DataSet *m_output;

  void MapAllFields();

  void PropagateMetadata();

  void CheckForRequiredField(const std::string &field_name);
};

} //namespace vtkh
#endif
