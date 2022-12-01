#ifndef VTK_H_DATA_SET_HPP
#define VTK_H_DATA_SET_HPP


#include <vector>
#include <string>

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/PartitionedDataSet.h>

namespace vtkh
{

class VTKH_API DataSet
{
protected:
  std::vector<vtkm::cont::DataSet> m_domains;
  std::vector<vtkm::Id>            m_domain_ids;
  vtkm::UInt64                     m_cycle;
  double                           m_time;
public:
  DataSet();
  ~DataSet();

  void AddDomain(vtkm::cont::DataSet data_set, vtkm::Id domain_id);

  void GetDomain(const vtkm::Id index,
                 vtkm::cont::DataSet &data_set,
                 vtkm::Id &domain_id);

  // set cycle meta data
  void SetCycle(const vtkm::UInt64 cycle);
  vtkm::UInt64 GetCycle() const;
  void SetTime(const double time);
  double GetTime() const;
  vtkm::cont::DataSet& GetDomain(const vtkm::Id index);
  vtkm::cont::DataSet& GetDomainById(const vtkm::Id domain_id);

  // check to see of field exists in at least one domain on this rank
  bool FieldExists(const std::string &field_name) const;
  // check to see if this field exists in at least one domain on any rank
  bool GlobalFieldExists(const std::string &field_name) const;

  // Use to indentify if the field is a scalar, vec2, vec3 ...
  // returns 0 if the field does not exist
  vtkm::Id NumberOfComponents(const std::string &field_name) const;

  vtkm::cont::Field GetField(const std::string &field_name,
                             const vtkm::Id domain_index);

  // checks to see if cells exist on this rank
  bool IsEmpty() const;
  // checks to see if cells exist on all ranks
  bool GlobalIsEmpty() const;


  // returns the number of domains on this rank
  vtkm::Id GetNumberOfDomains() const;
  // returns the number of domains on all ranks
  vtkm::Id GetGlobalNumberOfDomains() const;
  // returns the number of cells on this rank
  vtkm::Id GetNumberOfCells() const;
  // returns the number of cells on this rank
  vtkm::Id GetGlobalNumberOfCells() const;

  vtkm::cont::Field::Association GetFieldAssociation(const std::string field_name,
                                                     bool &valid_field) const;
  // returns the range of the scalar field across domains in this rank
  // If the field does not exist, the call returns an array of 0
  // throws an error if the number of components in different domains
  // do not match
  vtkm::cont::ArrayHandle<vtkm::Range> GetRange(const std::string &field_named) const;
  // returns the range of the scalar field across all ranks
  // If the field does not exist, the call returns an array of 0
  // throws an error if the number of components in different domains
  // do not match
  vtkm::cont::ArrayHandle<vtkm::Range> GetGlobalRange(const std::string &field_name) const;

  // returns the a list of domain ids on this rank
  std::vector<vtkm::Id> GetDomainIds() const;

  // add a scalar field to this data set with a constant value
  void AddConstantPointField(const vtkm::Float32 value, const std::string fieldname);

  bool HasDomainId(const vtkm::Id &domain_id) const;
  /*! \brief IsStructured returns true if all domains, globally,
   *         are stuctured data sets of the same topological dimension.
   *  \param topological_dims set to the dimensions of the cell set (1,2, or 3)
   *         If unstructred or structured with different dimensions, this value
   *         is set to -1
   *  \param cell_set_index the index of the cell set to perform the IsStructured
   *         test. Defaults to 0.
   */
  bool IsStructured(int &topological_dims) const;

  // returns true if every single domain is unstructrued
  bool IsUnstructured() const;

  bool IsPointMesh() const;

  void PrintSummary(std::ostream &stream) const;
};


  // return true if there is at most one domain on each rank
  bool OneDomainPerRank(vtkm::cont::PartitionedDataSet* data_set);

  // returns the union of all abounds on all ranks
  vtkm::Bounds GetGlobalBounds(vtkm::cont::PartitionedDataSet data_set,
		               vtkm::Id coordinate_system_index = 0);

  // returns the union of all domains bounds on this rank
  vtkm::Bounds GetBounds(vtkm::cont::PartitionedDataSet data_set,
		         vtkm::Id coordinate_system_index = 0);

  // returns a bounds of a single domain
  vtkm::Bounds GetDomainBounds(vtkm::cont::PartitionedDataSet data_set, 
		               const int &domain_index,
                               vtkm::Id coordinate_system_index = 0);

  bool IsStructured(vtkm::cont::PartitionedDataSet* data_set,
		    int &topological_dims);

  // returns true if every single domain is unstructrued
  bool IsUnstructured(vtkm::cont::PartitionedDataSet* data_set);

  bool IsPointMesh(vtkm::cont::PartitionedDataSet* data_set);

} // namespace vtkh

#endif
