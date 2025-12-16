#ifndef VTK_H_DATA_SET_HPP
#define VTK_H_DATA_SET_HPP


#include <vector>
#include <string>

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <viskores/cont/DataSet.h>

namespace vtkh
{

class VTKH_API DataSet
{
protected:
  std::vector<viskores::cont::DataSet> m_domains;
  std::vector<viskores::Id>            m_domain_ids;
  viskores::UInt64                     m_cycle;
  double                           m_time;
public:
  DataSet();
  ~DataSet();

  void AddDomain(viskores::cont::DataSet data_set, viskores::Id domain_id);

  void GetDomain(const viskores::Id index,
                 viskores::cont::DataSet &data_set,
                 viskores::Id &domain_id);

  // set cycle meta data
  void SetCycle(const viskores::UInt64 cycle);
  viskores::UInt64 GetCycle() const;
  void SetTime(const double time);
  double GetTime() const;
  viskores::cont::DataSet& GetDomain(const viskores::Id index);
  viskores::cont::DataSet& GetDomainById(const viskores::Id domain_id);

  // check to see of field exists in at least one domain on this rank
  bool FieldExists(const std::string &field_name) const;
  // check to see if this field exists in at least one domain on any rank
  bool GlobalFieldExists(const std::string &field_name) const;
  // remove a field if it exists, otherwise no-op
  void RemoveField(const std::string &field_name);

  // Use to identify if the field is a scalar, vec2, vec3 ...
  // returns 0 if the field does not exist
  viskores::Id NumberOfComponents(const std::string &field_name) const;

  viskores::cont::Field GetField(const std::string &field_name,
                             const viskores::Id domain_index);

  // checks to see if cells exist on this rank
  bool IsEmpty() const;
  // checks to see if cells exist on all ranks
  bool GlobalIsEmpty() const;

  // return true if there is at most one domain on each rank
  bool OneDomainPerRank() const;

  // returns the number of domains on this rank
  viskores::Id GetNumberOfDomains() const;
  // returns the number of domains on all ranks
  viskores::Id GetGlobalNumberOfDomains() const;
  // returns the number of cells on this rank
  viskores::Id GetNumberOfCells() const;
  // returns the number of cells on this rank
  viskores::Id GetGlobalNumberOfCells() const;
  // returns the union of all domains bounds on this rank
  viskores::Bounds GetBounds(viskores::Id coordinate_system_index = 0) const;
  // returns the union of all abounds on all ranks
  viskores::Bounds GetGlobalBounds(viskores::Id coordinate_system_index = 0) const;
  // returns a bounds of a single domain
  viskores::Bounds GetDomainBounds(const int &domain_index,
                               viskores::Id coordinate_system_index = 0) const;

  viskores::cont::Field::Association GetFieldAssociation(const std::string &field_name,
                                                     bool &valid_field) const;
  //Get an ID value representing the type of data field_name is
  //-1 if an invalid field i.e. globalFieldExists==false
  viskores::Id GetFieldType(const std::string &field_name,
                        bool &valid_field) const;
  // returns the range of the scalar field across domains in this rank
  // If the field does not exist, the call returns an array of 0
  // throws an error if the number of components in different domains
  // do not match
  viskores::cont::ArrayHandle<viskores::Range> GetRange(const std::string &field_named) const;
  // returns the range of the scalar field across all ranks
  // If the field does not exist, the call returns an array of 0
  // throws an error if the number of components in different domains
  // do not match
  viskores::cont::ArrayHandle<viskores::Range> GetGlobalRange(const std::string &field_name) const;

  // returns the a list of domain ids on this rank
  std::vector<viskores::Id> GetDomainIds() const;

  // add a scalar field to this data set with a constant value
  void AddConstantCellField(const viskores::Float32 value, const std::string &fieldname);
  void AddConstantPointField(const viskores::Float32 value, const std::string &fieldname);
  void AddLinearPointField(const viskores::Float32 value, const std::string &fieldname);
  void AddDomainIdField(const std::string &fieldname);

  bool HasDomainId(const viskores::Id &domain_id) const;
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

  bool IsLineMesh() const;

  void PrintSummary(std::ostream &stream) const;
};

} // namespace vtkh

#endif
