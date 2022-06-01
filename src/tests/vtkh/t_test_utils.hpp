#ifndef t_test_utils_hpp
#define t_test_utils_hpp

#include <assert.h>
#include <random>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderRectilinear.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>

#define BASE_SIZE 32
typedef vtkm::cont::ArrayHandleUniformPointCoordinates UniformCoords;

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

template <typename FieldType>
vtkm::cont::Field CreateCellScalarField(int size, const char* fieldName)
{
  vtkm::cont::ArrayHandle<FieldType> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    FieldType val = i / vtkm::Float32(size);
    data.WritePortal().Set(i, val);
  }


  vtkm::cont::Field field(fieldName,
                          vtkm::cont::Field::Association::CELL_SET,
                          data);
  return field;
}

vtkm::cont::Field CreateGhostScalarField(vtkm::Id3 dims)
{
  vtkm::Int32 size = dims[0] * dims[1] * dims[2];
  vtkm::cont::ArrayHandle<vtkm::Int32> data;
  data.Allocate(size);

  for(int z = 0; z < dims[2]; ++z)
    for(int y = 0; y < dims[1]; ++y)
      for(int x = 0; x < dims[0]; ++x)
  {
    vtkm::UInt8 flag = 0;
    if(x < 1 || x > dims[0] - 2) flag = 1;
    if(y < 1 || y > dims[1] - 2) flag = 1;
    if(z < 1 || z > dims[2] - 2) flag = 1;
    vtkm::Id index = z * dims[0] * dims[1] + y * dims[0] + x;
    data.WritePortal().Set(index, flag);
  }

  vtkm::cont::Field field("ghosts",
                          vtkm::cont::Field::Association::CELL_SET,
                          data);
  return field;
}

template <typename FieldType>
vtkm::cont::Field CreatePointScalarField(UniformCoords coords, const char* fieldName)

{
  const int size = coords.GetNumberOfValues();
  vtkm::cont::ArrayHandle<FieldType> data;
  data.Allocate(size);
  auto portal = coords.ReadPortal();
  for(int i = 0; i < size; ++i)
  {
    vtkm::Vec<FieldType,3> point = portal.Get(i);

    FieldType val = vtkm::Magnitude(point) + 1.f;
    data.WritePortal().Set(i, val);
  }

  vtkm::cont::Field field(fieldName,
                          vtkm::cont::Field::Association::POINTS,
                          data);
  return field;
}

template <typename FieldType>
vtkm::cont::Field CreatePointVecField(int size, const char* fieldName)
{
  vtkm::cont::ArrayHandle<vtkm::Vec<FieldType,3>> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    FieldType val = i / FieldType(size);

    vtkm::Vec<FieldType, 3> vec(val, -val, val);

    data.WritePortal().Set(i, vec);
  }

  vtkm::cont::Field field(fieldName,
                          vtkm::cont::Field::Association::POINTS,
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

  UniformCoords point_handle(point_dims,
                             origin,
                             spacing);

  vtkm::cont::CoordinateSystem coords("coords", point_handle);
  data_set.AddCoordinateSystem(coords);

  vtkm::cont::CellSetStructured<3> cell_set;
  cell_set.SetPointDimensions(point_dims);
  data_set.SetCellSet(cell_set);

  int num_points = point_dims[0] * point_dims[1] * point_dims[2];
  int num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  data_set.AddField(CreatePointScalarField<vtkm::Float32>(point_handle, "point_data_Float32"));
  data_set.AddField(CreatePointVecField<vtkm::Float32>(num_points, "vector_data_Float32"));
  data_set.AddField(CreateCellScalarField<vtkm::Float32>(num_cells, "cell_data_Float32"));
  data_set.AddField(CreatePointScalarField<vtkm::Float64>(point_handle, "point_data_Float64"));
  data_set.AddField(CreatePointVecField<vtkm::Float64>(num_points, "vector_data_Float64"));
  data_set.AddField(CreateCellScalarField<vtkm::Float64>(num_cells, "cell_data_Float64"));
  data_set.AddField(CreateGhostScalarField(cell_dims));
  return data_set;
}

vtkm::cont::DataSet CreateTestDataRectilinear(int block, int num_blocks, int base_size)
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

  std::vector<vtkm::Float64> xvals, yvals, zvals;
  xvals.resize((size_t)point_dims[0]);
  xvals[0] = static_cast<vtkm::Float64>(local_block.m_mins[0]);
  for (size_t i = 1; i < (size_t)point_dims[0]; i++)
    xvals[i] = xvals[i - 1] + spacing[0];

  yvals.resize((size_t)point_dims[1]);
  yvals[0] = static_cast<vtkm::Float64>(local_block.m_mins[1]);
  for (size_t i = 1; i < (size_t)point_dims[1]; i++)
    yvals[i] = yvals[i - 1] + spacing[1];

  zvals.resize((size_t)point_dims[2]);
  zvals[0] = static_cast<vtkm::Float64>(local_block.m_mins[2]);
  for (size_t i = 1; i < (size_t)point_dims[2]; i++)
    zvals[i] = zvals[i - 1] + spacing[2];

  vtkm::cont::DataSetBuilderRectilinear dataSetBuilder;
  vtkm::cont::DataSet data_set = dataSetBuilder.Create(xvals, yvals, zvals);

  int num_points = point_dims[0] * point_dims[1] * point_dims[2];

  data_set.AddField(CreatePointVecField<vtkm::Float32>(num_points, "vector_data_Float32"));
  data_set.AddField(CreatePointVecField<vtkm::Float64>(num_points, "vector_data_Float64"));

  return data_set;
}

vtkm::cont::DataSet CreateTestDataPoints(int num_points)
{
  std::vector<double> x_vals;
  std::vector<double> y_vals;
  std::vector<double> z_vals;
  std::vector<vtkm::UInt8> shapes;
  std::vector<vtkm::IdComponent> num_indices;
  std::vector<vtkm::Id> conn;
  std::vector<double> field;

  x_vals.resize(num_points);
  y_vals.resize(num_points);
  z_vals.resize(num_points);
  shapes.resize(num_points);
  conn.resize(num_points);
  num_indices.resize(num_points);
  field.resize(num_points);

  std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647> rgen{ 0 };
  std::uniform_real_distribution<double> dist{ -10., 10.};

  for(int i = 0; i < num_points; ++i)
  {
    x_vals[i] = dist(rgen);
    y_vals[i] = dist(rgen);
    z_vals[i] = dist(rgen);
    field[i] = dist(rgen);
    shapes[i] = vtkm::CELL_SHAPE_VERTEX;
    num_indices[i] = 1;
    conn[i] = i;
  }
  vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
  vtkm::cont::DataSet data_set = dataSetBuilder.Create(x_vals,
                                                       y_vals,
                                                       z_vals,
                                                       shapes,
                                                       num_indices,
                                                       conn);
  vtkm::cont::Field vfield = vtkm::cont::make_Field("point_data_Float64",
                                              vtkm::cont::Field::Association::POINTS,
                                              field,
                                              vtkm::CopyFlag::On);
  data_set.AddField(vfield);
  return data_set;
}

#endif
