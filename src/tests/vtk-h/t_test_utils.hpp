#ifndef t_test_utils_hpp
#define t_test_utils_hpp

#include <assert.h>
#include <vtkm/cont/DataSet.h>

#define BASE_SIZE 32

struct SpatialDivision
{
  int m_mins[3];
  int m_maxs[3];

  SpatialDivision()
    : m_mins{0,0,0},
      m_maxs{1,1,1}
  {

  }

  bool CanSplit(int dim)
  {
    return m_maxs[dim] - m_mins[dim] + 1> 1;
  }

  SpatialDivision Split(int dim)
  {
    SpatialDivision r_split;
    r_split = *this;
    assert(CanSplit(dim));
    int size = m_maxs[dim] - m_mins[dim] + 1;
    int left_offset = size / 2;   
  
    //shrink the left side
    m_maxs[dim] = m_mins[dim] + left_offset - 1;
    //shrink the right side
    r_split.m_mins[dim] = m_maxs[dim] + 1;
    return r_split;    
  }
};

SpatialDivision GetBlock(int block, int num_blocks, SpatialDivision total_size)
{

  std::vector<SpatialDivision> divs; 
  divs.push_back(total_size);
  int assigned = 1;
  int avail = num_blocks - 1;
  int current_dim = 0;
  int missed_splits = 0;
  const int num_dims = 3;
  while(avail > 0)
  {
    const int current_size = divs.size();
    int temp_avail = avail;
    for(int i = 0; i < current_size; ++i)
    {
      if(avail == 0) break;
      if(!divs[i].CanSplit(current_dim))
      {
        continue;
      }
      divs.push_back(divs[i].Split(current_dim));
      --avail;
    }      
    if(temp_avail == avail)
    {
      // dims were too small to make any spit
      missed_splits++;
      if(missed_splits == 3)
      {
        // we tried all three dims and could
        // not make a split.
        for(int i = 0; i < avail; ++i)
        {
          SpatialDivision empty;
          empty.m_maxs[0] = 0;
          empty.m_maxs[1] = 0;
          empty.m_maxs[2] = 0;
          divs.push_back(empty);
        }
        if(block == 0)
        {
          std::cerr<<"** Warning **: data set size is too small to"
                   <<" divide between "<<num_blocks<<" blocks. "
                   <<" Adding "<<avail<<" empty data sets\n";
        }

        avail = 0; 
      }
    }
    else
    {
      missed_splits = 0;
    }

    current_dim = (current_dim + 1) % num_dims;
  }

  return divs.at(block);
}

vtkm::cont::Field CreateCellScalarField(int size)
{
  vtkm::cont::ArrayHandle<vtkm::Float32> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    vtkm::Float32 val = i / vtkm::Float32(size);
    data.GetPortalControl().Set(i, val); 
  }

  vtkm::cont::Field field("cell_data",
                          vtkm::cont::Field::ASSOC_CELL_SET,
                          "cells",
                          data);
  return field;
}

vtkm::cont::Field CreatePointScalarField(int size)
{
  vtkm::cont::ArrayHandle<vtkm::Float32> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    vtkm::Float32 val = i / vtkm::Float32(size);
    data.GetPortalControl().Set(i, val); 
  }

  vtkm::cont::Field field("point_data",
                          vtkm::cont::Field::ASSOC_POINTS,
                          data);
  return field;
}

vtkm::cont::Field CreatePointVecField(int size)
{
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    vtkm::Float32 val = i / vtkm::Float32(size);

    vtkm::Vec<vtkm::Float32, 3> vec(val, -val, val);

    data.GetPortalControl().Set(i, vec); 
  }

  vtkm::cont::Field field("vector_data",
                          vtkm::cont::Field::ASSOC_POINTS,
                          data);
  return field;
}

vtkm::cont::DataSet CreateTestData(int block, int num_blocks, int base_size)
{
  SpatialDivision mesh_size;

  mesh_size.m_mins[0] = 0;
  mesh_size.m_mins[1] = 0;
  mesh_size.m_mins[2] = 0;

  mesh_size.m_maxs[0] = num_blocks * base_size - 1; 
  mesh_size.m_maxs[1] = num_blocks * base_size - 1; 
  mesh_size.m_maxs[2] = num_blocks * base_size - 1; 

  SpatialDivision local_block = GetBlock(block, num_blocks, mesh_size);

  vtkm::Vec<vtkm::Float32,3> origin;
  origin[0] = local_block.m_mins[0]; 
  origin[1] = local_block.m_mins[1]; 
  origin[2] = local_block.m_mins[2]; 

  vtkm::Vec<vtkm::Float32,3> spacing(1.f, 1.f, 1.f);

  vtkm::Id3 point_dims;
  point_dims[0] = local_block.m_maxs[0] - local_block.m_mins[0] + 2;
  point_dims[1] = local_block.m_maxs[1] - local_block.m_mins[1] + 2;
  point_dims[2] = local_block.m_maxs[2] - local_block.m_mins[2] + 2;


  vtkm::Id3 cell_dims;
  cell_dims[0] = point_dims[0] - 1;
  cell_dims[1] = point_dims[1] - 1;
  cell_dims[2] = point_dims[2] - 1;

  vtkm::cont::DataSet data_set;

  data_set.AddCoordinateSystem( vtkm::cont::CoordinateSystem("coords",
                                                             point_dims,
                                                             origin,
                                                             spacing));
  vtkm::cont::CellSetStructured<3> cell_set("cells");   
  cell_set.SetPointDimensions(point_dims);
  data_set.AddCellSet(cell_set);
  int num_points = point_dims[0] * point_dims[1] * point_dims[2];
  int num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  data_set.AddField(CreatePointScalarField(num_points));
  data_set.AddField(CreatePointVecField(num_points));
  data_set.AddField(CreateCellScalarField(num_cells));

  return data_set;
}

#endif
