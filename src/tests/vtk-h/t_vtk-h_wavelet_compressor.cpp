//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <vtkh.hpp>
#include <vtkh_data_set.hpp>
#include <vtkh_wavelet_compressor.hpp>
#include "t_test_utils.hpp"

#include <iostream>


TEST(vtkh_wavelet_compressor, vtkh_wavelet_compressor)
{
  vtkh::DataSet data_set;
 
  const int base_size = 32;
  const int num_blocks = 1; 
  
  for(int i = 0; i < num_blocks; ++i)
  {
    data_set.AddDomain(CreateTestData(i, num_blocks, base_size), i);
  }

  vtkh::WaveletCompressor compressor;
  
  compressor.SetInput(&data_set);
  compressor.Update();
}
