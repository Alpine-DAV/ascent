#ifndef t_test_utils_hpp
#define t_test_utils_hpp

#include <assert.h>
#include <random>
#include <viskores/Matrix.h>
#include <viskores/VectorAnalysis.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/DataSetBuilderRectilinear.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/cont/DataSetBuilderUniform.h>
//#include <viskores/cont/testing/Testing.h>

#define BASE_SIZE 32
typedef viskores::cont::ArrayHandleUniformPointCoordinates UniformCoords;

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
viskores::cont::Field CreateCellScalarField(int size, const char* fieldName)
{
  viskores::cont::ArrayHandle<FieldType> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    FieldType val = i / viskores::Float32(size);
    data.WritePortal().Set(i, val);
  }


  viskores::cont::Field field(fieldName,
                          viskores::cont::Field::Association::Cells,
                          data);
  return field;
}

viskores::cont::Field CreateGhostScalarField(viskores::Id3 dims)
{
  viskores::Int32 size = dims[0] * dims[1] * dims[2];
  viskores::cont::ArrayHandle<viskores::Int32> data;
  data.Allocate(size);

  for(int z = 0; z < dims[2]; ++z)
    for(int y = 0; y < dims[1]; ++y)
      for(int x = 0; x < dims[0]; ++x)
  {
    viskores::UInt8 flag = 0;
    if(x < 1 || x > dims[0] - 2) flag = 1;
    if(y < 1 || y > dims[1] - 2) flag = 1;
    if(z < 1 || z > dims[2] - 2) flag = 1;
    viskores::Id index = z * dims[0] * dims[1] + y * dims[0] + x;
    data.WritePortal().Set(index, flag);
  }

  viskores::cont::Field field("ghosts",
                          viskores::cont::Field::Association::Cells,
                          data);
  return field;
}

template <typename FieldType>
viskores::cont::Field CreatePointScalarField(UniformCoords coords, const char* fieldName)

{
  const int size = coords.GetNumberOfValues();
  viskores::cont::ArrayHandle<FieldType> data;
  data.Allocate(size);
  auto portal = coords.ReadPortal();
  for(int i = 0; i < size; ++i)
  {
    viskores::Vec<FieldType,3> point = portal.Get(i);

    FieldType val = viskores::Magnitude(point) + 1.f;
    data.WritePortal().Set(i, val);
  }

  viskores::cont::Field field(fieldName,
                          viskores::cont::Field::Association::Points,
                          data);
  return field;
}

template <typename FieldType>
viskores::cont::Field CreatePointVecField(int size, const char* fieldName)
{
  viskores::cont::ArrayHandle<viskores::Vec<FieldType,3>> data;
  data.Allocate(size);

  for(int i = 0; i < size; ++i)
  {
    FieldType val = i / FieldType(size);

    viskores::Vec<FieldType, 3> vec(val, -val, val);

    data.WritePortal().Set(i, vec);
  }

  viskores::cont::Field field(fieldName,
                          viskores::cont::Field::Association::Points,
                          data);
  return field;
}

viskores::cont::DataSet CreateTestData(int block, int num_blocks, int base_size)
{
  SpatialDivision mesh_size;

  mesh_size.m_mins[0] = 0;
  mesh_size.m_mins[1] = 0;
  mesh_size.m_mins[2] = 0;

  mesh_size.m_maxs[0] = num_blocks * base_size - 1;
  mesh_size.m_maxs[1] = num_blocks * base_size - 1;
  mesh_size.m_maxs[2] = num_blocks * base_size - 1;

  SpatialDivision local_block = GetBlock(block, num_blocks, mesh_size);

  viskores::Vec<viskores::Float32,3> origin;
  origin[0] = local_block.m_mins[0];
  origin[1] = local_block.m_mins[1];
  origin[2] = local_block.m_mins[2];

  viskores::Vec<viskores::Float32,3> spacing(1.f, 1.f, 1.f);

  viskores::Id3 point_dims;
  point_dims[0] = local_block.m_maxs[0] - local_block.m_mins[0] + 2;
  point_dims[1] = local_block.m_maxs[1] - local_block.m_mins[1] + 2;
  point_dims[2] = local_block.m_maxs[2] - local_block.m_mins[2] + 2;


  viskores::Id3 cell_dims;
  cell_dims[0] = point_dims[0] - 1;
  cell_dims[1] = point_dims[1] - 1;
  cell_dims[2] = point_dims[2] - 1;

  viskores::cont::DataSet data_set;

  UniformCoords point_handle(point_dims,
                             origin,
                             spacing);

  viskores::cont::CoordinateSystem coords("coords", point_handle);
  data_set.AddCoordinateSystem(coords);

  viskores::cont::CellSetStructured<3> cell_set;
  cell_set.SetPointDimensions(point_dims);
  data_set.SetCellSet(cell_set);

  int num_points = point_dims[0] * point_dims[1] * point_dims[2];
  int num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  data_set.AddField(CreatePointScalarField<viskores::Float32>(point_handle, "point_data_Float32"));
  data_set.AddField(CreatePointVecField<viskores::Float32>(num_points, "vector_data_Float32"));
  data_set.AddField(CreateCellScalarField<viskores::Float32>(num_cells, "cell_data_Float32"));
  data_set.AddField(CreatePointScalarField<viskores::Float64>(point_handle, "point_data_Float64"));
  data_set.AddField(CreatePointVecField<viskores::Float64>(num_points, "vector_data_Float64"));
  data_set.AddField(CreateCellScalarField<viskores::Float64>(num_cells, "cell_data_Float64"));
  data_set.AddField(CreateGhostScalarField(cell_dims));
  return data_set;
}

viskores::cont::DataSet CreateTestDataRectilinear(int block, int num_blocks, int base_size)
{
  SpatialDivision mesh_size;

  mesh_size.m_mins[0] = 0;
  mesh_size.m_mins[1] = 0;
  mesh_size.m_mins[2] = 0;

  mesh_size.m_maxs[0] = num_blocks * base_size - 1;
  mesh_size.m_maxs[1] = num_blocks * base_size - 1;
  mesh_size.m_maxs[2] = num_blocks * base_size - 1;

  SpatialDivision local_block = GetBlock(block, num_blocks, mesh_size);

  viskores::Vec<viskores::Float32,3> origin;
  origin[0] = local_block.m_mins[0];
  origin[1] = local_block.m_mins[1];
  origin[2] = local_block.m_mins[2];

  viskores::Vec<viskores::Float32,3> spacing(1.f, 1.f, 1.f);

  viskores::Id3 point_dims;
  point_dims[0] = local_block.m_maxs[0] - local_block.m_mins[0] + 2;
  point_dims[1] = local_block.m_maxs[1] - local_block.m_mins[1] + 2;
  point_dims[2] = local_block.m_maxs[2] - local_block.m_mins[2] + 2;


  viskores::Id3 cell_dims;
  cell_dims[0] = point_dims[0] - 1;
  cell_dims[1] = point_dims[1] - 1;
  cell_dims[2] = point_dims[2] - 1;

  std::vector<viskores::Float64> xvals, yvals, zvals;
  xvals.resize((size_t)point_dims[0]);
  xvals[0] = static_cast<viskores::Float64>(local_block.m_mins[0]);
  for (size_t i = 1; i < (size_t)point_dims[0]; i++)
    xvals[i] = xvals[i - 1] + spacing[0];

  yvals.resize((size_t)point_dims[1]);
  yvals[0] = static_cast<viskores::Float64>(local_block.m_mins[1]);
  for (size_t i = 1; i < (size_t)point_dims[1]; i++)
    yvals[i] = yvals[i - 1] + spacing[1];

  zvals.resize((size_t)point_dims[2]);
  zvals[0] = static_cast<viskores::Float64>(local_block.m_mins[2]);
  for (size_t i = 1; i < (size_t)point_dims[2]; i++)
    zvals[i] = zvals[i - 1] + spacing[2];

  viskores::cont::DataSetBuilderRectilinear dataSetBuilder;
  viskores::cont::DataSet data_set = dataSetBuilder.Create(xvals, yvals, zvals);

  int num_points = point_dims[0] * point_dims[1] * point_dims[2];

  data_set.AddField(CreatePointVecField<viskores::Float32>(num_points, "vector_data_Float32"));
  data_set.AddField(CreatePointVecField<viskores::Float64>(num_points, "vector_data_Float64"));

  return data_set;
}

viskores::cont::DataSet CreateTestDataPoints(int num_points)
{
  std::vector<double> x_vals;
  std::vector<double> y_vals;
  std::vector<double> z_vals;
  std::vector<viskores::UInt8> shapes;
  std::vector<viskores::IdComponent> num_indices;
  std::vector<viskores::Id> conn;
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
    shapes[i] = viskores::CELL_SHAPE_VERTEX;
    num_indices[i] = 1;
    conn[i] = i;
  }
  viskores::cont::DataSetBuilderExplicit dataSetBuilder;
  viskores::cont::DataSet data_set = dataSetBuilder.Create(x_vals,
                                                       y_vals,
                                                       z_vals,
                                                       shapes,
                                                       num_indices,
                                                       conn);
  viskores::cont::Field vfield = viskores::cont::make_Field("point_data_Float64",
                                              viskores::cont::Field::Association::Points,
                                              field,
                                              viskores::CopyFlag::On);
  data_set.AddField(vfield);
  return data_set;
}

//-----------------------------------------------------------------------------
//Create Viskores Data Sets
//-----------------------------------------------------------------------------

//Make a 2Duniform dataset.
inline viskores::cont::DataSet Make2DUniformDataSet0()
{
  viskores::cont::DataSetBuilderUniform dsb;
  constexpr viskores::Id2 dimensions(3, 2);
  viskores::cont::DataSet dataSet = dsb.Create(dimensions);

  constexpr viskores::Id nVerts = 6;
  constexpr viskores::Float32 var[nVerts] = { 10.1f, 20.1f, 30.1f, 40.1f, 50.1f, 60.1f };

  dataSet.AddPointField("pointvar", var, nVerts);

  constexpr viskores::Float32 cellvar[2] = { 100.1f, 200.1f };
  dataSet.AddCellField("cellvar", cellvar, 2);

  return dataSet;
}

//Make a 2D rectilinear dataset.
inline viskores::cont::DataSet Make2DRectilinearDataSet0()
{
  viskores::cont::DataSetBuilderRectilinear dsb;
  std::vector<viskores::Float32> X(3), Y(2);

  X[0] = 0.0f;
  X[1] = 1.0f;
  X[2] = 2.0f;
  Y[0] = 0.0f;
  Y[1] = 1.0f;

  viskores::cont::DataSet dataSet = dsb.Create(X, Y);

  const viskores::Id nVerts = 6;
  viskores::Float32 var[nVerts];
  for (int i = 0; i < nVerts; i++)
    var[i] = (viskores::Float32)i;
  dataSet.AddPointField("pointvar", var, nVerts);

  const viskores::Id nCells = 2;
  viskores::Float32 cellvar[nCells];
  for (int i = 0; i < nCells; i++)
    cellvar[i] = (viskores::Float32)i;
  dataSet.AddCellField("cellvar", cellvar, nCells);

  return dataSet;
}

inline viskores::cont::DataSet Make3DExplicitDataSet5()
{
  viskores::cont::DataSet dataSet;

  const int nVerts = 11;
  using CoordType = viskores::Vec3f_32;
  CoordType coordinates[nVerts] = {
    CoordType(0, 0, 0),     //0
    CoordType(1, 0, 0),     //1
    CoordType(1, 0, 1),     //2
    CoordType(0, 0, 1),     //3
    CoordType(0, 1, 0),     //4
    CoordType(1, 1, 0),     //5
    CoordType(1, 1, 1),     //6
    CoordType(0, 1, 1),     //7
    CoordType(2, 0.5, 0.5), //8
    CoordType(0, 2, 0),     //9
    CoordType(1, 2, 0)      //10
  };
  viskores::Float32 vars[nVerts] = { 10.1f, 20.1f, 30.2f, 40.2f, 50.3f, 60.2f,
			                                   70.2f, 80.3f, 90.f,  10.f,  11.f };

  dataSet.AddCoordinateSystem(
    viskores::cont::make_CoordinateSystem("coordinates", coordinates, nVerts, viskores::CopyFlag::On));

  //Set point scalar
  dataSet.AddField(make_Field(
    "pointvar", viskores::cont::Field::Association::Points, vars, nVerts, viskores::CopyFlag::On));

  //Set cell scalar
  const int nCells = 4;
  viskores::Float32 cellvar[nCells] = { 100.1f, 110.f, 120.2f, 130.5f };
  dataSet.AddField(make_Field(
    "cellvar", viskores::cont::Field::Association::Cells, cellvar, nCells, viskores::CopyFlag::On));

  viskores::cont::CellSetExplicit<> cellSet;
  viskores::Vec<viskores::Id, 8> ids;

  cellSet.PrepareToAddCells(nCells, 23);

  ids[0] = 0;
  ids[1] = 1;
  ids[2] = 5;
  ids[3] = 4;
  ids[4] = 3;
  ids[5] = 2;
  ids[6] = 6;
  ids[7] = 7;
  cellSet.AddCell(viskores::CELL_SHAPE_HEXAHEDRON, 8, ids);

  ids[0] = 1;
  ids[1] = 5;
  ids[2] = 6;
  ids[3] = 2;
  ids[4] = 8;
  cellSet.AddCell(viskores::CELL_SHAPE_PYRAMID, 5, ids);

  ids[0] = 5;
  ids[1] = 8;
  ids[2] = 10;
  ids[3] = 6;
  cellSet.AddCell(viskores::CELL_SHAPE_TETRA, 4, ids);

  ids[0] = 4;
  ids[1] = 7;
  ids[2] = 9;
  ids[3] = 5;
  ids[4] = 6;
  ids[5] = 10;
  cellSet.AddCell(viskores::CELL_SHAPE_WEDGE, 6, ids);

  cellSet.CompleteAddingCells(nVerts);

  //todo this need to be a reference/shared_ptr style class
  dataSet.SetCellSet(cellSet);

  return dataSet;
}

inline viskores::cont::DataSet Make3DUniformDataSet0()
{
  viskores::cont::DataSetBuilderUniform dsb;
  constexpr viskores::Id3 dimensions(3, 2, 3);
  viskores::cont::DataSet dataSet = dsb.Create(dimensions);

  constexpr int nVerts = 18;
  constexpr viskores::Float32 vars[nVerts] = { 10.1f,  20.1f,  30.1f,  40.1f,  50.2f,  60.2f,
                                           70.2f,  80.2f,  90.3f,  100.3f, 110.3f, 120.3f,
                                           130.4f, 140.4f, 150.4f, 160.4f, 170.5f, 180.5f };

  //Set point and cell scalar
  dataSet.AddPointField("pointvar", vars, nVerts);

  constexpr viskores::Float32 cellvar[4] = { 100.1f, 100.2f, 100.3f, 100.4f };
  dataSet.AddCellField("cellvar", cellvar, 4);

  return dataSet;
}

inline viskores::cont::DataSet Make3DExplicitDataSet2()
{
  viskores::cont::DataSet dataSet;

  const int nVerts = 8;
  using CoordType = viskores::Vec3f_32;
  CoordType coordinates[nVerts] = {
    CoordType(0, 0, 0), // 0
    CoordType(1, 0, 0), // 1
    CoordType(1, 0, 1), // 2
    CoordType(0, 0, 1), // 3
    CoordType(0, 1, 0), // 4
    CoordType(1, 1, 0), // 5
    CoordType(1, 1, 1), // 6
    CoordType(0, 1, 1)  // 7
  };
  viskores::Float32 vars[nVerts] = { 10.1f, 20.1f, 30.2f, 40.2f, 50.3f, 60.2f, 70.2f, 80.3f };

  dataSet.AddCoordinateSystem(
    viskores::cont::make_CoordinateSystem("coordinates", coordinates, nVerts, viskores::CopyFlag::On));

  //Set point scalar
  dataSet.AddField(make_Field(
    "pointvar", viskores::cont::Field::Association::Points, vars, nVerts, viskores::CopyFlag::On));

  //Set cell scalar
  viskores::Float32 cellvar[2] = { 100.1f };
  dataSet.AddField(make_Field(
    "cellvar", viskores::cont::Field::Association::Cells, cellvar, 1, viskores::CopyFlag::On));

  viskores::cont::CellSetExplicit<> cellSet;
  viskores::Vec<viskores::Id, 8> ids;
  ids[0] = 0;
  ids[1] = 1;
  ids[2] = 2;
  ids[3] = 3;
  ids[4] = 4;
  ids[5] = 5;
  ids[6] = 6;
  ids[7] = 7;

  cellSet.PrepareToAddCells(1, 8);
  cellSet.AddCell(viskores::CELL_SHAPE_HEXAHEDRON, 8, ids);
  cellSet.CompleteAddingCells(nVerts);

  //todo this need to be a reference/shared_ptr style class
  dataSet.SetCellSet(cellSet);

  return dataSet;
}

namespace detail
{

template <typename T>
struct TestValueImpl;
} //namespace detail

// Many tests involve getting and setting values in some index-based structure
// (like an array). These tests also often involve trying many types. The
// overloaded TestValue function returns some unique value for an index for a
// given type. Different types might give different values.
//
template <typename T>
static inline T TestValue(viskores::Id index, T)
{
  return detail::TestValueImpl<T>()(index);
}

namespace detail
{

template <typename T>
struct TestValueImpl
{
  T DoIt(viskores::Id index, viskores::TypeTraitsIntegerTag) const
  {
    constexpr bool larger_than_2bytes = sizeof(T) > 2;
    if (larger_than_2bytes)
    {
       return T(index * 100);
    }
    else
    {
      return T(index + 100);
    }
  }

  T DoIt(viskores::Id index, viskores::TypeTraitsRealTag) const
  {
    return T(0.01f * static_cast<float>(index) + 1.001f);
  }

  T operator()(viskores::Id index) const
  {
    return this->DoIt(index, typename viskores::TypeTraits<T>::NumericTag());
  }
};

template <typename T, viskores::IdComponent N>
struct TestValueImpl<viskores::Vec<T, N>>
{
  viskores::Vec<T, N> operator()(viskores::Id index) const
  {
    viskores::Vec<T, N> value;
    for (viskores::IdComponent i = 0; i < N; i++)
    {
      value[i] = TestValue(index * N + i, T());
    }
    return value;
  }
};

template <typename U, typename V>
struct TestValueImpl<viskores::Pair<U, V>>
{
  viskores::Pair<U, V> operator()(viskores::Id index) const
  {
    return viskores::Pair<U, V>(TestValue(2 * index, U()), TestValue(2 * index + 1, V()));
  }
};

template <typename T, viskores::IdComponent NumRow, viskores::IdComponent NumCol>
struct TestValueImpl<viskores::Matrix<T, NumRow, NumCol>>
{
  viskores::Matrix<T, NumRow, NumCol> operator()(viskores::Id index) const
  {
    viskores::Matrix<T, NumRow, NumCol> value;
    viskores::Id runningIndex = index * NumRow * NumCol;
    for (viskores::IdComponent row = 0; row < NumRow; ++row)
    {
      for (viskores::IdComponent col = 0; col < NumCol; ++col)
      {
        value(row, col) = TestValue(runningIndex, T());
        ++runningIndex;
      }
    }
    return value;
  }
};

template <>
struct TestValueImpl<std::string>
{
  std::string operator()(viskores::Id index) const
  {
    std::stringstream stream;
    stream << index;
    return stream.str();
  }
};

} //namespace detail

// Verifies that the contents of the given array portal match the values
// returned by viskores::testing::TestValue.
template <typename PortalType>
static inline void CheckPortal(const PortalType& portal)
{
  using ValueType = typename PortalType::ValueType;
  for (viskores::Id index = 0; index < portal.GetNumberOfValues(); index++)
  {
    ValueType expectedValue = TestValue(index, ValueType());
    ValueType foundValue = portal.Get(index);
    if (!test_equal(expectedValue, foundValue))
    {
      ASCENT_ERROR("Got unexpected value in array. Expected: " << expectedValue
              << ", Found: " << foundValue << "\n");
    }
  }
}

/// Sets all the values in a given array portal to be the values returned
/// by viskores::testing::TestValue. The ArrayPortal must be allocated first.
template <typename PortalType>
static inline void SetPortal(const PortalType& portal)
{
  using ValueType = typename PortalType::ValueType;

  for (viskores::Id index = 0; index < portal.GetNumberOfValues(); index++)
  {
    portal.Set(index, TestValue(index, ValueType()));
  }
}

inline viskores::cont::DataSet Make3DExplicitDataSetCowNose()
{
  // prepare data array
  const int nVerts = 17;
  using CoordType = viskores::Vec3f_64;
  CoordType coordinates[nVerts] = {
    CoordType(0.0480879, 0.151874, 0.107334),     CoordType(0.0293568, 0.245532, 0.125337),
    CoordType(0.0224398, 0.246495, 0.1351),       CoordType(0.0180085, 0.20436, 0.145316),
    CoordType(0.0307091, 0.152142, 0.0539249),    CoordType(0.0270341, 0.242992, 0.107567),
    CoordType(0.000684071, 0.00272505, 0.175648), CoordType(0.00946217, 0.077227, 0.187097),
    CoordType(-0.000168991, 0.0692243, 0.200755), CoordType(-0.000129414, 0.00247137, 0.176561),
    CoordType(0.0174172, 0.137124, 0.124553),     CoordType(0.00325994, 0.0797155, 0.184912),
    CoordType(0.00191765, 0.00589327, 0.16608),   CoordType(0.0174716, 0.0501928, 0.0930275),
    CoordType(0.0242103, 0.250062, 0.126256),     CoordType(0.0108188, 0.152774, 0.167914),
    CoordType(5.41687e-05, 0.00137834, 0.175119)
  };
  const int connectivitySize = 57;
  viskores::Id pointId[connectivitySize] = { 0, 1, 3,  2, 3,  1, 4,  5,  0,  1, 0,  5,  7,  8,  6,
                                         9, 6, 8,  0, 10, 7, 11, 7,  10, 0, 6,  13, 12, 13, 6,
                                         1, 5, 14, 1, 14, 2, 0,  3,  15, 0, 13, 4,  6,  16, 12,
                                         6, 9, 16, 7, 11, 8, 0,  15, 10, 7, 6,  0 };

  // create DataSet
  viskores::cont::DataSet dataSet;
  dataSet.AddCoordinateSystem(
  viskores::cont::make_CoordinateSystem("coordinates", coordinates, nVerts, viskores::CopyFlag::On));

  viskores::cont::ArrayHandle<viskores::Id> connectivity;
  connectivity.Allocate(connectivitySize);

  for (viskores::Id i = 0; i < connectivitySize; ++i)
  {
    connectivity.WritePortal().Set(i, pointId[i]);
  }
  viskores::cont::CellSetSingleType<> cellSet;
  cellSet.Fill(nVerts, viskores::CELL_SHAPE_TRIANGLE, 3, connectivity);
  dataSet.SetCellSet(cellSet);

  std::vector<viskores::Float32> pointvar(nVerts);
  std::iota(pointvar.begin(), pointvar.end(), 15.f);
  std::vector<viskores::Float32> cellvar(connectivitySize / 3);
  std::iota(cellvar.begin(), cellvar.end(), 132.f);

  viskores::cont::ArrayHandle<viskores::Vec3f> pointvec;
  pointvec.Allocate(nVerts);
  SetPortal(pointvec.WritePortal());

  viskores::cont::ArrayHandle<viskores::Vec3f> cellvec;
  cellvec.Allocate(connectivitySize / 3);
  SetPortal(cellvec.WritePortal());

  dataSet.AddPointField("pointvar", pointvar);
  dataSet.AddCellField("cellvar", cellvar);
  dataSet.AddPointField("point_vectors", pointvec);
  dataSet.AddCellField("cell_vectors", cellvec);

  return dataSet;
}

inline viskores::cont::DataSet Make3DRectilinearDataSet0()
{
  viskores::cont::DataSetBuilderRectilinear dsb;
  std::vector<viskores::Float32> X(3), Y(2), Z(3);

  X[0] = 0.0f;
  X[1] = 1.0f;
  X[2] = 2.0f;
  Y[0] = 0.0f;
  Y[1] = 1.0f;
  Z[0] = 0.0f;
  Z[1] = 1.0f;
  Z[2] = 2.0f;

  viskores::cont::DataSet dataSet = dsb.Create(X, Y, Z);

  const viskores::Id nVerts = 18;
  viskores::Float32 var[nVerts];
  for (int i = 0; i < nVerts; i++)
    var[i] = (viskores::Float32)i;
  dataSet.AddPointField("pointvar", var, nVerts);

  const viskores::Id nCells = 4;
  viskores::Float32 cellvar[nCells];
  for (int i = 0; i < nCells; i++)
    cellvar[i] = (viskores::Float32)i;
  dataSet.AddCellField("cellvar", cellvar, nCells);

  return dataSet;
}
             

#endif
