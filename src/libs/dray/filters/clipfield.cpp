// Copyright 2022 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/filters/clipfield.hpp>
#include <dray/filters/point_average.hpp>

#include <dray/dispatcher.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#include <conduit/conduit.hpp>
#include <conduit/conduit_relay.hpp>

// Some flags for conditionally compiled code. Uncomment as needed when debugging.
//#define PRINT_CASES
//#define WRITE_YAML_FILE
//#define WRITE_POINT3D_FILE

namespace dray
{
namespace detail
{

// Borrowed from VisIt.

// Points of original cell (up to 8, for the hex)
// Note: we assume P0 is zero in several places.
// Note: we assume these values are contiguous and monotonic.
#define P0     0
#define P1     1
#define P2     2
#define P3     3
#define P4     4
#define P5     5
#define P6     6
#define P7     7

// Edges of original cell (up to 12, for the hex)
// These are also points as an edge like EA is a point in between
// 2 original points.
// Note: we assume these values are contiguous and monotonic.
#define EA     20
#define EB     21
#define EC     22
#define ED     23
#define EE     24
#define EF     25
#define EG     26
#define EH     27
#define EI     28
#define EJ     29
#define EK     30
#define EL     31

// New interpolated points (ST_PNT outputs)
// Note: we assume these values are contiguous and monotonic.
#define N0     40
#define N1     41
#define N2     42
#define N3     43
#define N4     44
#define N5     45
#define N6     46

// Shapes
#define ST_TET 100
#define ST_PYR 101
#define ST_WDG 102
#define ST_HEX 103
#define ST_TRI 104
#define ST_QUA 105
#define ST_VTX 106
#define ST_LIN 107
#define ST_PNT 108

// Colors
#define COLOR0  120
#define COLOR1  121
#define NOCOLOR 122

// Include clip cases that make tets. These were developed using VisIt's
// clip editor but all of the cases were reworked to produce tets since
// dray can't make a mixture of cell types and lacks wedges and pyramids.
// The hex case actually was developed using some pyramids and these were
// split in a post-processing step to get the orientations right.
//
// Limitations:
//   1. The hex cases where the cells contained wedges were adapted to instead
//      use 3 tets. However, the automatic re-orientation of the clip editor
//      might produce some orientations that line up to neighboring cells.
//      This affects the watertightness of the cell, though all space is
//      covered.
//
//   2. Tet cases 0, 15 pass through the original tet rather than decomposing
//      into a set of tets that imprint 4 triangles per original tet face as
//      is done in the other cases. This will probably create the same
//      watertightness issue but it is also kind of better than blowing up
//      unclipped cells into many smaller tets. If size is not an issue,
//      cases 0, 15 could further decompose the tets to make it watertight.
//              
#include <dray/filters/internal/clip_cases_hex.cpp>
#include <dray/filters/internal/clip_cases_tet.cpp>

//---------------------------------------------------------------------------
// Just dispatch over P1 mesh types
template<typename Functor>
void dispatch_p1(Mesh *mesh, Field *field, Functor &func)
{
  if (!dispatch_mesh_field((HexMesh_P1*)0, mesh, field, func) &&
      !dispatch_mesh_field((TetMesh_P1*)0, mesh, field, func) &&

      !dispatch_mesh_field((QuadMesh_P1*)0, mesh, field, func) &&
      !dispatch_mesh_field((TriMesh_P1*)0,  mesh, field, func))

    detail::cast_mesh_failed(mesh, __FILE__, __LINE__);
}

//---------------------------------------------------------------------------
template<typename Functor>
void dispatch_p0p1(Field *field, Functor &func)
{
  if (//!dispatch_field_only((UnstructuredField<HexScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<HexScalar_P2>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<TetScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<TetScalar_P2>*)0, field, func) &&

      //!dispatch_field_only((UnstructuredField<QuadScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadScalar_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadScalar_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<QuadScalar_P2>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<TriScalar>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar_P0>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar_P1>*)0,  field, func) &&
      //!dispatch_field_only((UnstructuredField<TriScalar_P2>*)0,  field, func) &&

      //!dispatch_field_only((UnstructuredField<HexVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<HexVector_P2>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<TetVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<TetVector_P2>*)0, field, func) &&

      //!dispatch_field_only((UnstructuredField<QuadVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<QuadVector_P2>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<QuadVector_2D>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_2D_P0>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_2D_P1>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<QuadVector_2D_P2>*)0, field, func) &&
      //!dispatch_field_only((UnstructuredField<TriVector>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_P0>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_P1>*)0,  field, func) &&
      //!dispatch_field_only((UnstructuredField<TriVector_P2>*)0,  field, func) &&
      //!dispatch_field_only((UnstructuredField<TriVector_2D>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_2D_P0>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_2D_P1>*)0,  field, func)// &&
      //!dispatch_field_only((UnstructuredField<TriVector_2D_P2>*)0,  field, func)
     )
  {
    // NOTE: Building with CUDA does not like these lines so comment them out.
    //cout << "dispatch_p0p1 is not yet handling type " << field->type_name()
    //     << " for field " << field->name() << ". The field will be missing in "
    //        "the output." << endl;

    //detail::cast_field_failed(field, __FILE__, __LINE__);
  }
}

//---------------------------------------------------------------------------
/**
 @brief Blend a field using blend group information.
 */
class BlendFieldFunctor
{ 
public:
  BlendFieldFunctor(const Array<uint32> * _uNames,
                    const Array<uint32> * _uIndices,
                    const Array<int32>  * _blendGroupSizes,
                    const Array<int32>  * _blendGroupStart,
                    const Array<int32>  * _blendIds,
                    const Array<Float>  * _blendCoeff,
                    const Array<int32>  * _fragments,
                    const Array<int32>  * _fragmentOffsets,
                    const Array<int32>  * _conn,
                    int32 _total_elements)
  {
    // Stash blend parameters
    m_uNames = _uNames;
    m_uIndices = _uIndices;
    m_blendGroupSizes = _blendGroupSizes;
    m_blendGroupStart = _blendGroupStart;
    m_blendIds = _blendIds;
    m_blendCoeff = _blendCoeff;
    m_fragments = _fragments;
    m_fragmentOffsets = _fragmentOffsets;
    m_conn = _conn;
    m_total_elements = _total_elements;
    m_output = nullptr;
  }

  void reset()
  {
    m_output = nullptr;
  }

  // Called by dispatch_3d when we want to blend fields.
  template <typename FEType>
  void operator()(const UnstructuredField<FEType> &field)
  {
    if(field.order() == 0)
    {
      // Create a new GridFunction that contains the replicated data.
      auto bgf = replicate(field.get_dof_data());
      // Make a new UnstructuredField
      m_output = std::make_shared<UnstructuredField<FEType>>(bgf, 0, field.name());
    }
    else
    {
      // Create a new GridFunction that contains the blended data.
      auto bgf = blend(field.get_dof_data());
      // Make a new UnstructuredField
      m_output = std::make_shared<UnstructuredField<FEType>>(bgf, 1, field.name());
    }
  }

  std::shared_ptr<Field> get_output() const
  {
    return m_output;
  }

  // node-centered
  template <typename GridFuncType>
  GridFuncType blend(const GridFuncType &in_gf) const
  {
    auto nbg = m_uIndices->size();
    const auto uNames_ptr = m_uNames->get_device_ptr_const();
    const auto uIndices_ptr = m_uIndices->get_device_ptr_const();
    const auto blendGroupSizes_ptr = m_blendGroupSizes->get_device_ptr_const();
    const auto blendGroupStart_ptr = m_blendGroupStart->get_device_ptr_const();
    const auto blendIds_ptr = m_blendIds->get_device_ptr_const();
    const auto blendCoeff_ptr = m_blendCoeff->get_device_ptr_const();

    GridFuncType gf;
    gf.m_values.resize(nbg);
    DeviceGridFunction<GridFuncType::get_ncomp()> dgf(gf);
    DeviceGridFunctionConst<GridFuncType::get_ncomp()> in_dgf(in_gf);

    // Each loop iteration handles one unique blend group.
    //cout << "Start blending " << nbg << " groups" << endl;
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nbg), [=] DRAY_LAMBDA (int32 bgid)
    {
      // Original blendGroup index.
      auto origBGIdx = uIndices_ptr[bgid];
      auto npts = blendGroupSizes_ptr[origBGIdx];
      auto bgIdx = blendGroupStart_ptr[origBGIdx];

      //cout << "bg=" << origBGIdx << ", name=" << uNames_ptr[bgid] << endl;
      auto blended = Vec<Float, GridFuncType::get_ncomp()>::zero();
      for(int32 ii = 0; ii < npts; ii++)
      {
         //cout << "\t id=" << blendIds_ptr[bgIdx] << ", coeff=" << blendCoeff_ptr[bgIdx] << ", pt=" << in_dgf.m_values_ptr[blendIds_ptr[bgIdx]] << endl;
         blended += in_dgf.m_values_ptr[blendIds_ptr[bgIdx]] * blendCoeff_ptr[bgIdx];
         bgIdx++;
      }
      //cout << "\t saving " << blended << endl;
      dgf.m_values_ptr[bgid] = blended;
    });
    DRAY_ERROR_CHECK();

    // Finish filling in gf.
    gf.m_el_dofs = 4; // tet mesh
    gf.m_size_el = m_total_elements;
    gf.m_ctrl_idx = *m_conn;
    gf.m_size_ctrl = m_conn->size();
    return gf;
  }

  // element-centered
  template <typename GridFuncType>
  GridFuncType replicate(const GridFuncType &in_gf) const
  {
    const auto fragments_ptr = m_fragments->get_device_ptr_const();
    const auto fragmentOffsets_ptr = m_fragmentOffsets->get_device_ptr_const();
    GridFuncType gf;
    gf.m_ctrl_idx.resize(m_total_elements);
    gf.m_values.resize(m_total_elements);
    DeviceGridFunction<GridFuncType::get_ncomp()> dgf(gf);
    DeviceGridFunctionConst<GridFuncType::get_ncomp()> in_dgf(in_gf);

    // Replicate the data for elements that produced fragments into the 
    // output grid function.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, m_fragments->size()), [=] DRAY_LAMBDA (int32 elid)
    {
      // Repeat the element value in the output.
      auto start = fragmentOffsets_ptr[elid];
      auto n = fragments_ptr[elid];
      for(int32 i = 0; i < n; i++)
          dgf.m_values_ptr[start + i] = in_dgf.m_values_ptr[elid];
    });
    DRAY_ERROR_CHECK();

    // Not all elements in the original mesh produce fragments. So, we populate
    // the ctrl_idx in a second loop.
    auto ctrl_idx_ptr = gf.m_ctrl_idx.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, m_total_elements), [=] DRAY_LAMBDA (int32 i)
    {
      ctrl_idx_ptr[i] = i;
    });
    DRAY_ERROR_CHECK();

    // Finish filling in gf.
    gf.m_el_dofs = 1;
    gf.m_size_el = m_total_elements;
    gf.m_size_ctrl = m_total_elements;
    return gf;
  }

private:
  const Array<uint32> *m_uNames;
  const Array<uint32> *m_uIndices;
  const Array<int32>  *m_blendGroupSizes;
  const Array<int32>  *m_blendGroupStart;
  const Array<int32>  *m_blendIds;
  const Array<Float>  *m_blendCoeff;
  const Array<int32>  *m_fragments;
  const Array<int32>  *m_fragmentOffsets;
  const Array<int32>  *m_conn;
  int32                m_total_elements;
  std::shared_ptr<Field> m_output;
};

//---------------------------------------------------------------------------
// Applies a clip field operation on a DataSet. This code assumes that the
// mesh is P1 (linear) in connectivity, coordinates, and distance function.
// 
struct ClipFieldLinear
{
  // Keep a handle to the original dataset because we need it to be able to
  // access the other fields.
  DataSet m_input;

  // Output dataset produced by the functor.
  DataSet m_output;

  // ClipField attributes.
  Float m_clip_value;
  std::string   m_field_name;
  bool m_invert;
  bool m_exclude_clip_field;

  //-------------------------------------------------------------------------
  ClipFieldLinear(DataSet &input, Float value, const std::string &field_name,
     bool invert, bool ecf) : m_input(input), m_output(), m_clip_value(value),
      m_field_name(field_name), m_invert(invert), m_exclude_clip_field(ecf)
  {
  }

  //-------------------------------------------------------------------------
  // Execute the filter for the input mesh across all possible field types.
  void execute()
  {
    // This iterates over the product of possible mesh and scalar field types
    // to call the operator() function that can operate on concrete types.
    Field *field = m_input.field(m_field_name);
    if(field != nullptr && field->components() == 1)
    {
      // If the field is cell-centered then we have to move it to the points
      DataSet temp;
      PointAverage pavg;
      if(field->order() == Order::Constant)
      {
        pavg.set_field(m_field_name);
        temp = pavg.execute(m_input);
        field = temp.field(m_field_name);
      }
      dispatch_p1(m_input.mesh(), field, *this);
    }

    // Make sure the output domain id is the same as the input id.
    m_output.domain_id(m_input.domain_id());
  }

  //-------------------------------------------------------------------------
  // Load lookup tables for mesh element type into the lut arrays.
  template <typename MeshType>
  void load_lookups(const MeshType &/*m*/,
      Array<int32> &/*lut_nshapes*/,
      Array<int32> &/*lut_offset*/,
      Array<unsigned char> &/*lut_shapes*/) const
  {
    // Template specialization used later to load the real values.
  }

  //-------------------------------------------------------------------------
  template <typename FieldType>
  DRAY_EXEC static int32 clip_case(int32 elid,
                                   int32 el_dofs,
                                   const int32 *conn_ptr,
                                   const FieldType *dist_ptr,
                                   int32 *el_ids)
  {
    // Determine the clip case.
    int32 start = elid * el_dofs;
    int32 clipcase = 0;
    for(int32 j = 0; j < el_dofs; j++)
    {
      static const int32 reorder[] = {0,1,3,2,4,5,7,6};
      // dray hex and VisIt hex (tables are geared to VisIt hex) have
      // different node order.
      int32 nidx = (el_dofs == 8) ? reorder[j] : j;

      // Access data in dray order
      int32 dofid = conn_ptr[start + nidx];

      // Save the dofid in VisIt order.
      el_ids[j] = dofid;

      if(dist_ptr[dofid][0] > 0.)
      {
        clipcase |= (1 << j);
      }
    }
#ifdef PRINT_CASES
    cout << "elid " << elid << ": clipcase=" << clipcase << ", el_ids={";
    for(int32 i = 0; i < el_dofs; i++)
      cout << el_ids[i] << ", ";  
    cout << "}" << endl;
#endif
    return clipcase;
  }

  //-------------------------------------------------------------------------
  // NOTE: For larger meshes, a hash that makes uint64 might be desirable.
  DRAY_EXEC static uint32 jenkins_hash(const uint8 *data, uint32 length)
  {
    uint32 i = 0;
    uint32 hash = 0;
#if 1
    // Build the length into the hash so {1} and {0,1} hash to different values.
    const auto ldata = reinterpret_cast<const uint8 *>(&length);
    for(int e = 0; e < 4; e++)
    {
      hash += ldata[e];
      hash += hash << 10;
      hash ^= hash >> 6;
    }
#endif
    while(i != length)
    {
      hash += data[i++];
      hash += hash << 10;
      hash ^= hash >> 6;
    }
    hash += hash << 3;
    hash ^= hash >> 11;
    hash += hash << 15;
    return hash;
  };

  //-------------------------------------------------------------------------
  DRAY_EXEC static uint32 make_name_1(int32 id)
  {
    return jenkins_hash(reinterpret_cast<uint8*>(&id), sizeof(int32));
  };

  //-------------------------------------------------------------------------
  DRAY_EXEC static uint32 make_name_2(int32 id0, int32 id1)
  {
    int32 data[2] = {id0, id1};
    if(id1 < id0)
    {
      data[0] = id1;
      data[1] = id0;
    }
    return jenkins_hash(reinterpret_cast<uint8*>(data), 2*sizeof(int32));
  };

  //-------------------------------------------------------------------------
  DRAY_EXEC static uint32 make_name_n(const int32 *start, int32 n)
  {
    if(n == 2)
      return make_name_2(start[0], start[1]);

    //cout << n <<":(";
    int32 v[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // pick largest number of blends.
    // Copy unsorted values into data[].
    for(int32 i = 0; i < n; i++)
    {
      //cout << start[i] << ",";
      v[i] = start[i];
    }
    //cout << "),";
    // Sort (crude).
    for(int32 i = 0; i < n-1; i++)
    {
      for(int32 j = 0; j < n-i-1; j++)
      {
         if(v[j] > v[j+1])
         {
           int32 tmp = v[j]; // swap
           v[j] = v[j+1];
           v[j+1] = tmp;
         }
      }
    }
    return jenkins_hash(reinterpret_cast<uint8*>(v), n*sizeof(int32));
  };

  //-------------------------------------------------------------------------
  DRAY_EXEC static int32 bsearch(uint32 name, const uint32 *names, int32 n)
  {
    int32 index = -1;
    int32 left = 0;
    int32 right = n - 1;
    while(left <= right)
    {
      int32 m = (left + right) / 2;
      if(names[m] < name)
        left = m + 1;
      else if(names[m] > name)
        right = m - 1;
      else
      {
        index = m;
        break;
      }
    }
    //cout << "bsearch(" << name << ") -> " << index << endl;
    return index;
  }

  //-------------------------------------------------------------------------
  // This method gets invoked by dispatch, which will have converted the field
  // into a concrete derived type so this method is able to call methods on
  // the derived type with no virtual calls.
  template<typename MeshType, typename ScalarField>
  void operator()(const MeshType &mesh, const ScalarField &field)
  {
    /*
    The algorithm determines which clip case is being used for the cell. Then it
    iterates over the relevant points and tets for that case and keeps track of
    which corner points, edges, faces, or center points are being created. These
    points are used to build a set of blend groups that make the new points for
    the output element. A clean corner point for example that appears in a tet
    will be make a blend group that contains a single point, itself. An edge point
    will make a blend group that contains the 2 edge endpoints and their respective
    blend coefficients. A face or center point is created using the ST_PNT "command"
    in the lookup case and it makes a blend group that consists of various other
    points.


    elid------------
                   |
                   v
    blendGroups = [4,0,2,...]    (number of points created from element)
  blendGroupLen = [7,0,3,...]    (number of ints per element in blendIds)
    blendOffset = [0,7,7,10,...] (exclusive scan of blendGroupLen: where elem data begins)
                   |   | |
                   |   | ------------------------------------------
                   |   ------------------------------             |
                   v                                v             v
    blendCoeff= [{1. }{0.5 0.5}{0.3 0.7}{0.4 0.6},,{0.5 0.5}{1. },...]
    blendIds  = [{100}{100 101}{102 103}{104 105},,{104,105}{106},...]
                   ^         (elem 0)      ^        (elem 2)
                   |                       |
    blendNames= [{100,0x54321, 0x23456,0x34567}   {0x34567,  106},...]

    blendGroupSizes = [{1,2,2,2},,{2,1}...]
    blendGroupOffsets=[0,1,3,5,    7,9,...} (exclusive scan of blendGroupSizes)

    blendNames is made from combining the values in a blend group into a single id.
    For singles, we can use the point id itself. For pairs, we can do <p0:p1>. For
    longer lists, we sort the point ids and hash them.

    Each group in the blendIds is encoded to a blendName. We can then put
    the blendNames through a uniqueness filter so we can eliminate duplicate
    blend groups later on. This should make adjacent cells share points, even
    if they had to be created.

    origIndex   = [0,   1,       2,      3,          4,        5]
    blendNames  = [{100,0x54321, 0x23456,0x34567}   {0x34567,  106},...]
    uNames = [100, 106, 0x23456, 0x34567, 0x54321}
    uIndex = [0, 5, 2, 3, 1]    (3,4 were dups)

    For a point in the cell, get its blend group index in the cell. For example,
    if we're in cell 0 and we get point 0x54321. We want to find its blend group data
    in blendCoeff, blendIds.

    uidx = bsearch(uNames, 0x54321)                   (4)
    obidx = uIndex[uidx]                              (1 blend group)
    start = blendGroupOffsets[obidx]                  (1)
    int32 nids = blendGroupSizes[obidx]               (2)
    int32 *ids = &blendIds[start]                     ({0.5 0.5})
    Float *coeff = &blendCoeff[start]                 ({100 101})

    uidx = bsearch(uNames, 106)                       (1)
    obidx = uIndex[uidx]                              (5 blend group)
    start = blendGroupOffsets[obidx]                  (9)
    int32 nids = blendGroupSizes[obidx]               (1)
    int32 *ids = &blendIds[start]                     ({1.})
    Float *coeff = &blendCoeff[start]                 ({106})

    The bsearch result for the name in the uniqueNames is the global point id to
    use in the new dofs.
    */
    DRAY_LOG_OPEN("clipfield");

    // Make a distance field.
    ScalarField distance = create_distance_field(field);
    auto distance_gf = distance.get_dof_data();
    int32 nelem = mesh.cells(); // number of elements in mesh.
    auto el_dofs = mesh.get_dof_data().m_el_dofs;

    // Load the mesh/element-appropriate lut into arrays.
    Array<int32> lut_nshapes, lut_offset;
    Array<unsigned char> lut_shapes;
    load_lookups(mesh, lut_nshapes, lut_offset, lut_shapes);

    // We'll compute some per-element values for the outputs.
    Array<int32> fragments, blendGroups, blendGroupLen;
    fragments.resize(nelem);
    blendGroups.resize(nelem);
    blendGroupLen.resize(nelem);

    // Get pointers to pass to the lambdas.
    const auto dist_ptr = distance_gf.m_values.get_device_ptr();
    const auto conn_ptr = distance_gf.m_ctrl_idx.get_device_ptr();
    const auto lut_nshapes_ptr = lut_nshapes.get_device_ptr();
    const auto lut_offset_ptr = lut_offset.get_device_ptr();
    const auto lut_shapes_ptr = lut_shapes.get_device_ptr();
    auto fragments_ptr = fragments.get_device_ptr();
    auto blendGroups_ptr = blendGroups.get_device_ptr();
    auto blendGroupLen_ptr = blendGroupLen.get_device_ptr();

    // ----------------------------------------------------------------------
    //
    // Stage 1: Iterate over elements and their respective clip cases to
    //          determine sizes of outputs.
    //
    // ----------------------------------------------------------------------
    RAJA::ReduceSum<reduce_policy, int> fragment_sum(0);
    RAJA::ReduceSum<reduce_policy, int> blendGroups_sum(0);
    RAJA::ReduceSum<reduce_policy, int> blendGroupLen_sum(0);
    const bool do_invert = m_invert;
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
    {
      // Determine the clip case.
      int32 el_ids[8]; // max p1 ids.
      int32 clipcase = clip_case(elid, el_dofs, conn_ptr, dist_ptr, el_ids);

      // Get the shapes for this clip case.
      unsigned char *shapes = &lut_shapes_ptr[lut_offset_ptr[clipcase]];

      // The number of tets (cell fragments) produced for the case. We
      // need this to know overall how many cells there will be in the
      // output.
      int32 thisFragments = 0;

      // The number of blend groups (corners, centers, edges, faces)
      // for this element.
      int32 thisBlendGroups = 0;

      // The number of ints we need to store the ids for all blend records
      // in this element.
      int32 thisblendGroupLen = 0;

      // Points used in cell (range [P0,N6])
      unsigned char ptused[50] = {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
      };
      
      for(int32 si = 0; si < lut_nshapes_ptr[clipcase]; si++)
      {
        if(shapes[0] == ST_PNT)
        {
          // ST_PNT, 0, COLOR0, 8, P0, P1, P2, P3, P4, P5, P6, P7, 
          if(shapes[2] == NOCOLOR ||
             (!do_invert && shapes[2] == COLOR0) ||
             (do_invert && shapes[2] == COLOR1))
          {
            // The point is a keeper.

            for(unsigned char ni = 0; ni < shapes[3]; ni++)
            {
              auto ptid = shapes[4 + ni];

              // Increase the blend size to include this center point.
              if(/*ptid >= P0 &&*/ ptid <= P7)
              {
                 // corner point.
                 thisblendGroupLen++;
              }
              else if(ptid >= EA && ptid <= EL)
              {
                // edge points are derived from 2 corner points. If
                // those appear here then we're probably creating a
                // face point. We can store the 2 corner points in place
                // of the edge point (along with some blending coeff).
                thisblendGroupLen += 2;
              }
            }

            // This center or face point counts as a blend group.
            thisBlendGroups++;
          }

          shapes += (4 + shapes[3]);
        }
        else if(shapes[0] == ST_TET)
        {
          // ST_TET COLOR0 p0 p1 p2 p3
          if((!do_invert && shapes[1] == COLOR0) ||
             (do_invert && shapes[1] == COLOR1))
          {
            thisFragments++;

            // Count the points used in this cell.
            ptused[shapes[2]]++;
            ptused[shapes[3]]++;
            ptused[shapes[4]]++;
            ptused[shapes[5]]++;
          }
          shapes += 6;
        }
        // FUTURE: clip triangles
      }

      // Count which points in the original cell are used.
      for(unsigned char pid = P0; pid <= P7; pid++)
      {
        if(ptused[pid] > 0)
        {
          thisblendGroupLen++; // {p0}
          thisBlendGroups++;
        }
      }

      // Count edges that are used.
      for(unsigned char pid = EA; pid <= EL; pid++)
      {
        if(ptused[pid] > 0)
        {
          thisblendGroupLen += 2; // {p0 p1}
          thisBlendGroups++;
        }
      }

      // Save the results.
      blendGroups_ptr[elid] = thisBlendGroups;
      blendGroupLen_ptr[elid] = thisblendGroupLen;
      fragments_ptr[elid] = thisFragments;

      // Sum up the sizes overall.
      fragment_sum += thisFragments;
      blendGroups_sum += thisBlendGroups;
      blendGroupLen_sum += thisblendGroupLen;
    });
    DRAY_ERROR_CHECK();

    // ----------------------------------------------------------------------
    //
    // Stage 2: Do some scans to fill out blendOffset and blendGroupOffsets,
    //          which is where we fill in the real data.
    //
    // blendOffset : Starting offset for blending data like blendIds, blendCoeff.
    // blendGroupOffset : Starting offset for blendNames, blendGroupSizes.
    // fragmentOffsets : Where an element's fragments begin in the output.
    // ----------------------------------------------------------------------
    Array<int32> blendOffset, blendGroupOffsets, fragmentOffsets;
    blendOffset.resize(nelem);
    blendGroupOffsets.resize(nelem);
    fragmentOffsets.resize(nelem);
    auto blendOffset_ptr = blendOffset.get_device_ptr();
    auto blendGroupOffsets_ptr = blendGroupOffsets.get_device_ptr();
    auto fragmentOffsets_ptr = fragmentOffsets.get_device_ptr();

    // Make offsets via scan.
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(blendGroupLen_ptr, nelem),
                                     RAJA::make_span(blendOffset_ptr, nelem),
                                     RAJA::operators::plus<int>{});
    DRAY_ERROR_CHECK();
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(blendGroups_ptr, nelem),
                                     RAJA::make_span(blendGroupOffsets_ptr, nelem),
                                     RAJA::operators::plus<int>{});
    DRAY_ERROR_CHECK();
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(fragments_ptr, nelem),
                                     RAJA::make_span(fragmentOffsets_ptr, nelem),
                                     RAJA::operators::plus<int>{});
    DRAY_ERROR_CHECK();

    // ----------------------------------------------------------------------
    //
    // Stage 3: Iterate over the elements/cases again and fill in the blend
    //          groups that get produced: blendNames, blendGroupSizes,
    //          blendCoeff, blendIds. These are used to produce the new points.
    //
    //          NOTE: blendGroupStart is a scan of blendGroupSizes.
    //
    // ----------------------------------------------------------------------
    Array<uint32> blendNames;
    Array<int32> blendIds, blendGroupSizes, blendGroupStart;
    Array<Float> blendCoeff;
    blendNames.resize(blendGroups_sum.get());
    blendGroupSizes.resize(blendGroups_sum.get());
    blendGroupStart.resize(blendGroups_sum.get());
    blendIds.resize(blendGroupLen_sum.get());
    blendCoeff.resize(blendGroupLen_sum.get());

    auto blendNames_ptr = blendNames.get_device_ptr();
    auto blendGroupSizes_ptr = blendGroupSizes.get_device_ptr();
    auto blendGroupStart_ptr = blendGroupStart.get_device_ptr();
    auto blendIds_ptr = blendIds.get_device_ptr();
    auto blendCoeff_ptr = blendCoeff.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
    {
      // Determine the clip case.
      int32 el_ids[8]; // max p1 ids.
      int32 clipcase = clip_case(elid, el_dofs, conn_ptr, dist_ptr, el_ids);

      // Get the shapes for this lookup case.
      unsigned char *shapes = &lut_shapes_ptr[lut_offset_ptr[clipcase]];

      // Points used in cell (range [P0,N6])
      unsigned char ptused[50] = {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
      };

      // NOTE: I was going to make load_lookups load these into a device-accessible
      //       array so the template specialization would load the right one. I
      //       must not have done it right. Do this for now.
      static const unsigned char hex_edge_to_corners[12][2] = {
        { 0, 1 },   /* EA */
        { 1, 2 },   /* EB */
        { 3, 2 },   /* EC */
        { 3, 0 },   /* ED */
        { 4, 5 },   /* EE */
        { 5, 6 },   /* EF */
        { 6, 7 },   /* EG */
        { 7, 4 },   /* EH */
        { 0, 4 },   /* EI */
        { 1, 5 },   /* EJ */
        { 3, 7 },   /* EK */
        { 2, 6 }    /* EL */
      };
      static const unsigned char tet_edge_to_corners[6][2] = {
        { 0, 1 },   /* EA */
        { 1, 2 },   /* EB */
        { 2, 0 },   /* EC */
        { 0, 3 },   /* ED */
        { 1, 3 },   /* EE */
        { 2, 3 }    /* EF */
      };
      const unsigned char (*edge_to_corners)[2] =
        (el_dofs == 8) ? hex_edge_to_corners : tet_edge_to_corners;
      
      // Starting offset of where we store this element's blend groups.
      int32 bgStart = blendOffset_ptr[elid];
      int32 bgOffset = blendGroupOffsets_ptr[elid];
#ifdef PRINT_CASES
      if(clipcase != 255)
          cout << "elem " << elid << ": clipcase: " << clipcase << endl;
#endif
      for(int32 si = 0; si < lut_nshapes_ptr[clipcase]; si++)
      {
        if(shapes[0] == ST_PNT)
        {
          // ST_PNT, 0, COLOR0, 8, P0, P1, P2, P3, P4, P5, P6, P7, 
          if(shapes[2] == NOCOLOR ||
             (!do_invert && shapes[2] == COLOR0) ||
             (do_invert && shapes[2] == COLOR1))
          {
            // The point is a keeper. Add a blend group for this point.
            auto one_over_n = 1.f / static_cast<float>(shapes[3]);
            int32 start = bgStart;

            auto npts = shapes[3];
#ifdef PRINT_CASES
            cout << "\tST_PNT " << (int)shapes[1] << ", " << (int)shapes[2];
            for(unsigned char ni = 0; ni < npts; ni++)
            {
              auto ptid = shapes[4 + ni];
              cout << ", " << (int)ptid;
            }
            cout << endl;
#endif
            // Store where this blendGroup starts in the blendIds,blendCoeff.
            blendGroupStart_ptr[bgOffset] = bgStart;

            for(unsigned char ni = 0; ni < npts; ni++)
            {
              auto ptid = shapes[4 + ni];

              // Add the point to the blend group.
              if(ptid <= P7)
              {
                // corner point.
                blendIds_ptr[bgStart] = el_ids[ptid];
                blendCoeff_ptr[bgStart] = one_over_n;

                bgStart++;
              }
              else if(ptid >= EA && ptid <= EL)
              {
                // edge points are derived from 2 corner points. If
                // those appear here then we're probably creating a
                // face point. We can store the 2 corner points in place
                // of the edge point (along with some blending coeff).
                const unsigned char *c = edge_to_corners[ptid - EA];
                int32 id0 = el_ids[c[0]];
                int32 id1 = el_ids[c[1]];
                // Figure out the blend for edge.
                Float d0 = dist_ptr[id0][0];
                Float d1 = dist_ptr[id1][0];
                Float delta = d1 - d0;
                Float abs_delta = (delta < 0) ? -delta : delta;
                Float t = (abs_delta != 0.) ? (-d0 / delta) : 0.;
                
                blendIds_ptr[bgStart]   = id0;
                blendIds_ptr[bgStart+1] = id1;
                blendCoeff_ptr[bgStart] = one_over_n * (1. - t);
                blendCoeff_ptr[bgStart+1] = one_over_n * t;

                bgStart += 2;
              }
            }

            // Store how many points make up this blend group. Note that the
            // size will not necessarily be equal to npts if edges were involved.
            int32 nblended = bgStart - blendGroupStart_ptr[bgOffset];
            blendGroupSizes_ptr[bgOffset] = nblended;

            // Store "name" of blend group.
            auto blendName = make_name_n(blendIds_ptr + start, nblended);
            blendNames_ptr[bgOffset++] = blendName;
          }

          shapes += (4 + shapes[3]);
        }
        else if(shapes[0] == ST_TET)
        {
          // ST_TET COLOR0 p0 p1 p2 p3
          if((!do_invert && shapes[1] == COLOR0) ||
             (do_invert && shapes[1] == COLOR1))
          {
#ifdef PRINT_CASES
            cout << "\tST_TET ";
            int c = (int)shapes[1];
            if(c == 120) cout << "COLOR0";
            if(c == 121) cout << "COLOR1";
            if(c == 122) cout << "NOCOLOR";
            cout << ", " << (int)shapes[2]
                 << ", " << (int)shapes[3]
                 << ", " << (int)shapes[4]
                 << ", " << (int)shapes[5] << endl;
#endif

            // Count the points used in this cell.
            ptused[shapes[2]]++;
            ptused[shapes[3]]++;
            ptused[shapes[4]]++;
            ptused[shapes[5]]++;
          }
          shapes += 6;
        }
        // FUTURE: Handle clipping triangles
      }

      // Add blend group for each original point that was used.
      for(unsigned char pid = P0; pid <= P7; pid++)
      {
        if(ptused[pid] > 0)
        {
          // Store blend group info                
          blendIds_ptr[bgStart] = el_ids[pid];
          blendCoeff_ptr[bgStart] = 1.;

          // Store how many points make up this blend group.
          blendGroupSizes_ptr[bgOffset] = 1;

          // Store where this blendGroup starts in the blendIds,blendCoeff.
          blendGroupStart_ptr[bgOffset] = bgStart;

          // Store "name" of blend group.
          blendNames_ptr[bgOffset++] = make_name_1(el_ids[pid]);

          bgStart++;
        }
      }

      // Add blend group for each edge point that was used.
      for(unsigned char pid = EA; pid <= EL; pid++)
      {
        if(ptused[pid] > 0)
        {
          const unsigned char *c = edge_to_corners[pid - EA];
          int32 id0 = el_ids[c[0]];
          int32 id1 = el_ids[c[1]];
          // Figure out the blend for edge.
          Float d0 = dist_ptr[id0][0];
          Float d1 = dist_ptr[id1][0];
          Float delta = d1 - d0;
          Float abs_delta = (delta < 0) ? -delta : delta;
          Float t = (abs_delta != 0.) ? (-d0 / delta) : 0.;

          // Store blend group info                
          blendIds_ptr[bgStart]   = id0;
          blendIds_ptr[bgStart+1] = id1;
          blendCoeff_ptr[bgStart] = (1. - t);
          blendCoeff_ptr[bgStart+1] = t;

          // Store how many points make up this blend group.
          blendGroupSizes_ptr[bgOffset] = 2;

          // Store where this blendGroup starts in the blendIds,blendCoeff.
          blendGroupStart_ptr[bgOffset] = bgStart;

          // Store "name" of blend group.
          blendNames_ptr[bgOffset++] = make_name_2(id0, id1);

          bgStart += 2;
        }
      }
    });
    DRAY_ERROR_CHECK();

    // ----------------------------------------------------------------------
    //
    // Stage 4 - Make the blend groups unique based on their blendName.
    //
    // ----------------------------------------------------------------------
    // At this point, we have created the blend group data. We can now use the
    // blendNames to make unique blend groups. uNames contains a sorted list of
    // the unique blend group names while uIndices is their original index in
    // blendNames/blendGroupOffsets/blendGroupSizes.
    Array<uint32> uNames, uIndices;
    unique(blendNames, uNames, uIndices);
    uint32 *uNames_ptr = uNames.get_device_ptr();
    uint32 *uIndices_ptr = uIndices.get_device_ptr();

    // ----------------------------------------------------------------------
    //
    // Stage 5 - Iterate over the cases again and make new connectivity.
    //
    // ----------------------------------------------------------------------
    Array<int32> conn_out;
    conn_out.resize(fragment_sum.get() * 4);
    auto conn_out_ptr = conn_out.get_device_ptr();
    int32 uNames_len = uNames.size();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
    {
      // If there are no fragments, return from lambda.
      if(fragments_ptr[elid] == 0)
        return;

      // Determine the clip case.
      int32 el_ids[8]; // max p1 ids.
      int32 clipcase = clip_case(elid, el_dofs, conn_ptr, dist_ptr, el_ids);

      // Get the shapes for this lookup case.
      unsigned char *shapes = &lut_shapes_ptr[lut_offset_ptr[clipcase]];

      // Points used in cell (range [P0,N6])
      unsigned char ptused[50] = {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
      };

      // Iterate over the tet fragments to see which points they use.
      // The points correspond to blend groups that we have made in
      // the final output.
      for(int32 si = 0; si < lut_nshapes_ptr[clipcase]; si++)
      {
        if(shapes[0] == ST_PNT)
        {
          // Count the point as used since a blend group would have
          // been emitted for it under these conditions. This makes
          // sure that we get the ordering for point_2_newdof right
          // when a set of shapes happens to not use the blended point.
          // That, of course, means that the lut needs to be fixed a bit.
          if(shapes[2] == NOCOLOR ||
             (!do_invert && shapes[2] == COLOR0) ||
             (do_invert && shapes[2] == COLOR1))
          {
             ptused[N0 + shapes[1]]++;
          }

          // ST_PNT, 0, COLOR0, 8, P0, P1, P2, P3, P4, P5, P6, P7, 
          shapes += (4 + shapes[3]);
        }
        else if(shapes[0] == ST_TET)
        {
          // ST_TET COLOR0 p0 p1 p2 p3
          if((!do_invert && shapes[1] == COLOR0) ||
             (do_invert && shapes[1] == COLOR1))
          {
            // Count the points used in this cell.
            ptused[shapes[2]]++;
            ptused[shapes[3]]++;
            ptused[shapes[4]]++;
            ptused[shapes[5]]++;
          }
          shapes += 6;
        }
        // FUTURE: Handle clipping triangles
      }

      // This element's blendNames start at blendGroupOffsets[elid] in the
      // original, non-unique list.

      // Seek to the start of the blend groups for this element.
      int32 bgStart = blendGroupOffsets_ptr[elid];

      // Go through the points in the order they would have been added as blend
      // groups, get their blendName, and then overall index of that blendName
      // in uNames, the unique list of new dof names. That will be their index
      // in the final dofs.
      uint32 point_2_newdof[50];
      for(unsigned char pid = N0; pid <= N6; pid++)
      {
        if(ptused[pid] > 0)
        {
          auto name = blendNames_ptr[bgStart++];
          point_2_newdof[pid] = bsearch(name, uNames_ptr, uNames_len);
        }
      }
      for(unsigned char pid = P0; pid <= P7; pid++)
      {
        if(ptused[pid] > 0)
        {
          auto name = blendNames_ptr[bgStart++];
          point_2_newdof[pid] = bsearch(name, uNames_ptr, uNames_len);
        }
      }
      for(unsigned char pid = EA; pid <= EL; pid++)
      {
        if(ptused[pid] > 0)
        {
          auto name = blendNames_ptr[bgStart++];
          point_2_newdof[pid] = bsearch(name, uNames_ptr, uNames_len);
        }
      }

      // Iterate over the tet fragments again and output their
      // dofs.
      shapes = &lut_shapes_ptr[lut_offset_ptr[clipcase]];
      // This is where the output fragments start for this element
      int32 tetOutput = fragmentOffsets_ptr[elid] * 4;
      for(int32 si = 0; si < lut_nshapes_ptr[clipcase]; si++)
      {
        if(shapes[0] == ST_PNT)
        {
          // Skip past the point definition.
          // ST_PNT, 0, COLOR0, 8, P0, P1, P2, P3, P4, P5, P6, P7, 
          shapes += (4 + shapes[3]);
        }
        else if(shapes[0] == ST_TET)
        {
          // ST_TET COLOR0 p0 p1 p2 p3
          if((!do_invert && shapes[1] == COLOR0) ||
             (do_invert && shapes[1] == COLOR1))
          {
            // Output the dofs used for this tet.
            conn_out_ptr[tetOutput++] = point_2_newdof[shapes[2]];
            conn_out_ptr[tetOutput++] = point_2_newdof[shapes[3]];
            conn_out_ptr[tetOutput++] = point_2_newdof[shapes[4]];
            conn_out_ptr[tetOutput++] = point_2_newdof[shapes[5]];
          }
          shapes += 6;
        }
        // FUTURE: Handle clipping triangles
      }
    });
    DRAY_ERROR_CHECK();

    // ----------------------------------------------------------------------
    //
    // Stage 6 - Finish making the output mesh.
    //
    // ----------------------------------------------------------------------
    BlendFieldFunctor bff(&uNames, &uIndices, &blendGroupSizes, &blendGroupStart,
                          &blendIds, &blendCoeff, &fragments, &fragmentOffsets,
                          &conn_out, fragment_sum.get());
    // Blend coordinate dofs.
    GridFunction<3> gf = bff.blend(mesh.get_dof_data());
#ifdef WRITE_POINT3D_FILE
    // Write Point3D file containing xyz locations and point name.
    auto uNames_hptr = uNames.get_host_ptr();
    auto v_hptr = gf.m_values.get_host_ptr();
    FILE *f = fopen("uNames.3D", "wt");
    fprintf(f, "X Y Z Name\n");
    for(int i = 0; i < uNames.size(); i++)
    {
        fprintf(f, "%f %f %f %u\n", v_hptr[i][0], v_hptr[i][1], v_hptr[i][2], uNames_hptr[i]);
    }
    fclose(f);
#endif
    // Make a new P1 tet mesh.
    auto newmesh = std::make_shared<UnstructuredMesh<Tet_P1>>(gf, 1);
    newmesh->name(mesh.name());
    m_output.add_mesh(newmesh);

    // Blend fields and put them on the dataset.
    const int nfields = m_input.number_of_fields();
    for(int i = 0; i < nfields; i++)
    {
      // Get the field and check whether we want it on the output.
      Field *field = m_input.field(i);
      if(m_exclude_clip_field && field->name() == m_field_name)
        continue;

      // Dispatch to BlendFieldFunctor to blend the field.
      bff.reset();
      dispatch_p0p1(field, bff);
      auto f = bff.get_output();
      if(f != nullptr)
        m_output.add_field(f);
    }

#ifdef WRITE_YAML_FILE
    // Save the data to a YAML file to look at it.
    cout << "Writing clip debugging information." << endl;
    conduit::Node n;
    conduit::Node &inp = n["input"];
    mesh.to_node(inp);

    conduit::Node &s0 = n["stage0/distance"];
    distance.to_node(s0);

    conduit::Node &s1 = n["stage1"];
    s1["nelem"] = nelem;
    s1["fragment_sum"] = fragment_sum.get();
    s1["blendGroups_sum"] = blendGroups_sum.get();
    s1["blendGroupLen_sum"] = blendGroupLen_sum.get();
    s1["fragments"].set_external(fragments.get_host_ptr(), fragments.size());
    s1["blendGroups"].set_external(blendGroups.get_host_ptr(), blendGroups.size());
    s1["blendGroupLen"].set_external(blendGroupLen.get_host_ptr(), blendGroupLen.size());

    conduit::Node &s2 = n["stage2"];
    s2["blendOffset"].set_external(blendOffset.get_host_ptr(), blendOffset.size());
    s2["blendGroupOffsets"].set_external(blendGroupOffsets.get_host_ptr(), blendGroupOffsets.size());
    s2["fragmentOffsets"].set_external(fragmentOffsets.get_host_ptr(), fragmentOffsets.size());

    conduit::Node &s3 = n["stage3"];
    s3["blendGroupSizes"].set_external(blendGroupSizes.get_host_ptr(), blendGroupSizes.size());
    s3["blendGroupStart"].set_external(blendGroupStart.get_host_ptr(), blendGroupStart.size());
    s3["blendNames"].set_external(blendNames.get_host_ptr(), blendNames.size());
    s3["blendIds"].set_external(blendIds.get_host_ptr(), blendIds.size());
    s3["blendCoeff"].set_external(blendCoeff.get_host_ptr(), blendCoeff.size());

    conduit::Node &s4 = n["stage4"];
    s4["uNames"].set_external(uNames.get_host_ptr(), uNames.size());
    s4["uIndices"].set_external(uIndices.get_host_ptr(), uIndices.size());

    conduit::Node &s5 = n["stage5"];
    s5["conn_out"].set_external(conn_out.get_host_ptr(), conn_out.size());

    conduit::Node &s6 = n["stage6"];
    s6["gf.m_values"].set_external(reinterpret_cast<float*>(gf.m_values.get_host_ptr()), 3*gf.m_values.size());
    s6["gf.m_ctrl_idx"].set_external(gf.m_ctrl_idx.get_host_ptr(), gf.m_ctrl_idx.size());

    conduit::Node &sout = n["output"];
    m_output.to_node(sout);

    conduit::relay::io::save(n, "clipfield.yaml", "yaml");
#endif

    DRAY_LOG_CLOSE();
  }

  //-------------------------------------------------------------------------
  void
  unique(const Array<uint32> &keys_orig, Array<uint32> &skeys, Array<uint32> &sindices)
  {
    // Make a copy of the values and make original indices.
    Array<uint32> keys, indices;
    int32 n = keys_orig.size();
    keys.resize(n);
    indices.resize(n);
    auto keys_orig_ptr = keys_orig.get_device_ptr_const();
    auto keys_ptr = keys.get_device_ptr();
    auto indices_ptr = indices.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] RAJA_HOST_DEVICE (uint32 i) {
      keys_ptr[i] = keys_orig_ptr[i];
      indices_ptr[i] = i;
    });

    // Sort the keys, indices in place.
    RAJA::sort_pairs<for_policy>(RAJA::make_span(keys_ptr, n),
                                  RAJA::make_span(indices_ptr, n));

    // Make a mask array for where differences occur.
    Array<uint32> mask;
    mask.resize(n);
    auto mask_ptr = mask.get_device_ptr();
    RAJA::ReduceSum<reduce_policy, uint32> mask_sum(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] RAJA_HOST_DEVICE (uint32 i) {
      uint32 m = 1;
      if(i >= 1)
         m = (keys_ptr[i] != keys_ptr[i-1]) ? 1 : 0;

      mask_ptr[i] = m;
      mask_sum += m;
    });

    // Do a scan on the mask array to build an offset array.
    Array<uint32> offsets;
    offsets.resize(n);
    auto offsets_ptr = offsets.get_device_ptr();
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(mask_ptr, n),
                                      RAJA::make_span(offsets_ptr, n),
                                      RAJA::operators::plus<int>{});

    // Iterate over the mask/offsets to store values at the right
    // offset in the new array.
    uint32 newsize = mask_sum.get();
    skeys.resize(newsize);
    sindices.resize(newsize);
    auto skeys_ptr = skeys.get_device_ptr();
    auto sindices_ptr = sindices.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] RAJA_HOST_DEVICE (uint32 i) {
      if(mask_ptr[i])
      {
        skeys_ptr[offsets_ptr[i]] = keys_ptr[i];
        sindices_ptr[offsets_ptr[i]] = indices_ptr[i];
      }
    });
  }

  //-------------------------------------------------------------------------
  template<typename FEType>
  UnstructuredField<FEType>
  create_distance_field(const UnstructuredField<FEType> &field) const
  {
    // The size of the field's values
    int32 sz = field.get_dof_data().m_values.size();

    // Make a new distance array that is the same size as the input field.
    Array<Vec<Float,1>> dist;
    dist.resize(sz);

    Vec<Float,1> *dist_ptr = dist.get_device_ptr();

    // NOTE: Using CUDA, the following line throws an exception. Make it const.
    const auto *field_ptr = field.get_dof_data().m_values.get_device_ptr_const();

    // Make the distance field for all the dofs in the input field.
    const Float clip_value = m_clip_value;
    const RAJA::RangeSegment range(0, sz);
    RAJA::forall<for_policy>(range,
      [=] DRAY_LAMBDA (int32 i) {
        dist_ptr[i][0] = field_ptr[i][0] - clip_value;
    });
    DRAY_ERROR_CHECK();

    // Make a new grid function.
    GridFunction<1> gf;
    gf.m_ctrl_idx = field.get_dof_data().m_ctrl_idx;
    gf.m_values = dist;
    gf.m_el_dofs = field.get_dof_data().m_el_dofs;
    gf.m_size_el = field.get_dof_data().m_size_el;
    gf.m_size_ctrl = field.get_dof_data().m_size_ctrl;

    // Wrap as a new field.
    UnstructuredField<FEType> newfield(gf, field.get_poly_order());
    return newfield;
  }
};

//---------------------------------------------------------------------------
// Load Hex lookup data.
template <>
void
ClipFieldLinear::load_lookups(const HexMesh_P1 &m,
      Array<int32> &lut_nshapes,
      Array<int32> &lut_offset,
      Array<unsigned char> &lut_shapes) const
{
  lut_nshapes.set(reinterpret_cast<const int32*>(numClipShapesHex),
                  sizeof(numClipShapesHex)/sizeof(int));
  lut_offset.set(reinterpret_cast<const int32*>(startClipShapesHex),
                 sizeof(startClipShapesHex)/sizeof(int));
  lut_shapes.set(clipShapesHex,
                 sizeof(clipShapesHex)/sizeof(unsigned char));
}

//---------------------------------------------------------------------------
// Load Tet lookup data.
template <>
void
ClipFieldLinear::load_lookups(const TetMesh_P1 &m,
      Array<int32> &lut_nshapes,
      Array<int32> &lut_offset,
      Array<unsigned char> &lut_shapes) const
{
  lut_nshapes.set(reinterpret_cast<const int32*>(numClipShapesTet),
                  sizeof(numClipShapesTet)/sizeof(int));
  lut_offset.set(reinterpret_cast<const int32*>(startClipShapesTet),
                 sizeof(startClipShapesTet)/sizeof(int));
  lut_shapes.set(clipShapesTet,
                 sizeof(clipShapesTet)/sizeof(unsigned char));
}


}//namespace detail

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
ClipField::ClipField() : m_clip_value(0.), m_field_name(), m_invert(false),
  m_exclude_clip_field(false)
{
}

ClipField::~ClipField()
{
}

void
ClipField::set_clip_value(Float value)
{
  m_clip_value = value;
}

void
ClipField::set_field(const std::string &field)
{
  m_field_name = field;
}

void
ClipField::set_invert_clip(bool value)
{
  m_invert = value;
}

Float
ClipField::clip_value() const
{
  return m_clip_value;
}

const std::string &
ClipField::field() const
{
  return m_field_name;
}

bool
ClipField::invert() const
{
  return m_invert;
}

void
ClipField::exclude_clip_field(bool value)
{
  m_exclude_clip_field = value;
}

DataSet
ClipField::execute(DataSet dom)
{
  DataSet output;
  if(dom.mesh() != nullptr && dom.mesh()->order() == 1)
  {
    detail::ClipFieldLinear func(dom, m_clip_value, m_field_name, m_invert, m_exclude_clip_field);
    func.execute();
    output = func.m_output;
  }
  else
  {
    DRAY_ERROR("TODO: support high order meshes.");
  }

  return output;
}

Collection
ClipField::execute(Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dom = collection.domain(i);
    if(dom.mesh() != nullptr && dom.mesh()->order() == 1)
    {
      detail::ClipFieldLinear func(dom, m_clip_value, m_field_name, m_invert, m_exclude_clip_field);
      func.execute();
      res.add_domain(func.m_output);
    }
    else
    {
      DRAY_ERROR("TODO: support high order meshes.");
    }
  }
  return res;
}

}//namespace dray
