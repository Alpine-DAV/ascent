#ifndef VTK_H_DATA_SET_HPP
#define VTK_H_DATA_SET_HPP


#include <vector>
#include <string>

#include <vtkh.hpp>
#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class DataSet
{
protected:
  std::vector<vtkm::cont::DataSet> m_domains;
  std::vector<vtkm::Id>            m_domain_ids;
public:
  void AddDomain(vtkm::cont::DataSet data_set, vtkm::Id domain_id); 
  void GetDomain(const vtkm::Id index, 
                 vtkm::cont::DataSet &data_set, 
                 vtkm::Id &domain_id); 

  vtkm::cont::DataSet GetDomain(const vtkm::Id index); 
  
  vtkm::cont::Field GetField(const std::string &field_name, 
                             const vtkm::Id domain_index); 
  vtkm::Id GetNumberOfDomains() const;
  vtkm::Id GetGlobalNumberOfDomains() const;
  vtkm::Bounds GetBounds(vtkm::Id coordinate_system_index = 0) const;
  vtkm::Bounds GetGlobalBounds(vtkm::Id coordinate_system_index = 0) const;
  vtkm::Bounds GetDomainBounds(const int &domain_index,
                               vtkm::Id coordinate_system_index = 0) const;
  vtkm::cont::ArrayHandle<vtkm::Range> GetGlobalRange(const std::string &field_name) const;
  vtkm::cont::ArrayHandle<vtkm::Range> GetGlobalRange(const vtkm::Id index) const;
  
  void PrintSummary(std::ostream &stream) const;
};

} // namespace vtkh

#endif
