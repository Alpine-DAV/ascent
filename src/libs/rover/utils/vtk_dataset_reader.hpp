//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef rover_vtk_dataset_reader_h
#define rover_vtk_dataset_reader_h
#include <string>
#include <vector>
#include <vtkm_typedefs.hpp>
namespace rover {

class VTKReader
{
public:
  VTKReader();
  void read_file(const std::string &file_name);
  vtkmDataSet get_data_set();
protected:
  vtkmDataSet m_dataset;
};

class MultiDomainVTKReader
{
public:
  MultiDomainVTKReader();
  void read_file(const std::string &directory, const std::string &file_name);
  std::vector<vtkmDataSet> get_data_sets();
protected:
  std::vector<vtkmDataSet> m_datasets;
};
} // namespace rover

#endif
