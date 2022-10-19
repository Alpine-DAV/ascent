#include <dray/filters/clipfield.hpp>

#include <dray/dispatcher.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{


/*

a hex will make 6 tets. The number of edges is:

12 from the original hex
6 to divide each face
1 internal diagonal

Each cell can make 19 edges.


int cell_edge_count[ncells] = 0;
int cell_edge_start[ncells][19];
int cell_edge_end[ncells][19];

[parallel] for each hex cell:
    // look up the distances for the hex corners.
    dist[8] = ...

    for each tet in hex
        tetdist[] = the dist values that matter for this tet.
        for each edge in tet
            if edge appears in solid case
                edgecount = cell_edge_count[cellid]++;
                cell_edge_start[cellid][edgecount] = edge.start;
                cell_edge_end[cellid][edgecount] = edge.end;
                cell_edge_blend[cellid][edgecount] = t;
            
// We will have determined all of the edges for all of the cells
// at this point. The cell_edge arrays will contain some duplicates.
int lowest_mask[possible_edges] = 0
int lowest_index[possible_edges] = [0,1,2,3,4,...]

[parallel] foreach possible_edge:
    edge = determine_from_possible_edge_index()
    index = -1;
    foreach ce in cell_edge:
        if edge == ce:
            if index == -1:
                index = e
                lowest_mask[edgeindex] = 1
                
                break

// Can we do the exclusive scan trick to get the indices that we'll pull
// from the cell_edge...




unique

  We have a list of keys: [0,1,3,1,3,22,5]
  We want to make a unique list.

  keys    = [0,1,3,1,3,22,5]
  indices = [0,1,2,3,4,5,6]

  Use keys to sort indices via sort_pairs.

  keys    = [0,1,1,3,3,5,22]
  indices = [0,1,3,2,4,6,5]

  RAJA::sort_pairs<for_policy>(keys, indices,...)

  The keys are now in sorted order. The indices for each key refer to the
  original index that we would want to preserve.

  Now we run exclusive or scan over the key elements. Non-zero values in
  the sequence ought to be unique since it means that there is a change in
  value.
  RAJA::inclusive_scan_inplace< exec_policy >(keys, xord, RAJA::operators::bit_xor<int>)

  The first value should always count as "on". How to address this?

  xord = [1,0,2,0,6,19]

  for_all<>((mask, xord, i){
     if(i == 0)
         mask[i] = 1;
     else
         mask[i] = xord[i-1] != 0;
  });

  mask    = [1,1,0,1,0,1,1]
  keys    = [0,1,1,3,3,5,22]
  indices = [0,1,3,2,4,6,5]   <-- the ones that got sorted by keys

             * * *     *  *   <--- selected indices using mask
  origkeys= [0,1,3,1,3,22,5]  <--- just for illustration

  Use inclusive scan sum to count the number of 1's in mask. That's how 
  many elements the final array will have.

  int n = scan.get();
  
  Do exclusive scan of mask to produce offset array.

  RAJA::exclusive_scan<for_policy>(mask, offset)

  offset = [0,1,1,2,2,3,4]

  unique = new Array[n]
  for_all<>(offset,unique,, i){
     if(mask[i] == 1)
         unique[offset[i]] = keys[i];
  });
  
  I'm not sure we need indices[] at all here.

  At this point, unique should contain:

  unique = [0,1,3,5,22]


 What we might want to do in general is:
 1. Iterate over cells and determine clip case and how many new edge points 
    each cell will make.
 2. reduce+sum the number of edge points
 3. Allocate edge array
 4. Iterate over the cells again, use the clip case we stored to figure out
    which edges will be made. Figure out an identifier for each point. HOW TO DO THIS?
    For points along an edge, we could do pt1:pt2 into a long. THESE NEED TO BE UNIQUE
    ACROSS THE DATASET. Also, we have face points that blend more than 2 points.
    HOW TO TAG THOSE?

 5. Run edge array through unique

THE GOAL IS TO MAKE A POINT BLENDING ARRAY...

<npts0 p0, p1,...><npts1 p0, p1,...>...
offset = [0, npts0+1, npts1+1, ...]

THEN WITH THE POINT BLENDING ARRAY, WE CAN BLEND COORDS AND FIELDS

*/

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

// Include Hex clip cases that make tets.
#include <dray/filters/post.C>

// Just dispatch over
template<typename Functor>
void dispatch_p1(Mesh *mesh, Field *field, Functor &func)
{
  if (!dispatch_mesh_field((HexMesh_P1*)0, mesh, field, func) &&
      !dispatch_mesh_field((TetMesh_P1*)0, mesh, field, func) &&

      !dispatch_mesh_field((QuadMesh_P1*)0, mesh, field, func) &&
      !dispatch_mesh_field((TriMesh_P1*)0,  mesh, field, func))

    detail::cast_mesh_failed(mesh, __FILE__, __LINE__);
}

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

  ClipFieldLinear(DataSet &input, Float value, const std::string &field_name,
     bool invert) : m_input(input), m_output(), m_clip_value(value),
      m_field_name(field_name), m_invert(invert)
  {
  }

  // Execute the filter for the input mesh across all possible field types.
  void execute()
  {
    // This iterates over the product of possible mesh and scalar field types
    // to call the operator() function that can operate on concrete types.
    Field *field = m_input.field(m_field_name);
    if(field != nullptr && field->components() == 1)
    {

// TODO: make sure that the field is on the nodes!!!

      dispatch_p1(m_input.mesh(), field, *this);
    }

    // Make sure the output domain id is the same as the input id.
    m_output.domain_id(m_input.domain_id());
  }

  // Load lookup tables for mesh element type into the lut arrays.
  template <typename MeshType>
  void load_lookups(MeshType &/*m*/,
      Array<int32> &/*lut_nshapes*/,
      Array<int32> &/*lut_offset*/,
      Array<unsigned char> &/*lut_shapes*/) const
  {
  }

  // This method gets invoked by dispatch, which will have converted the field
  // into a concrete derived type so this method is able to call methods on
  // the derived type with no virtual calls.
  template<typename MeshType, typename ScalarField>
  void operator()(MeshType &mesh, ScalarField &field)
  {
    DRAY_LOG_OPEN("clipfield");

    // Figure out which elements to keep based on the input field.
    ScalarField distance = create_distance_field(field);

    // Load the mesh-appropriate lut into arrays.
    Array<int32> lut_nshapes, lut_offset;
    Array<unsigned char> lut_shapes;
    load_lookups(mesh, lut_nshapes, lut_offset, lut_shapes);
#if 1
    auto distance_gf = distance.get_dof_data();
    int32 nelem = distance_gf.get_num_elem(); // number of elements in gf.
    auto el_dofs = distance_gf.m_el_dofs;

    // We'll compute some per-element values for the outputs.
    Array<int32> fragments, blendGroups, blendGroupLen;
    fragments.resize(nelem);
    blendGroups.resize(nelem);
    blendGroupLen.resize(nelem);

    // Get pointers to pass to the lambda.
    const auto dist_ptr = distance_gf.m_values.get_device_ptr();
    const auto conn_ptr = distance_gf.m_ctrl_idx.get_device_ptr();
    const auto lut_nshapes_ptr = lut_nshapes.get_device_ptr();
    const auto lut_offset_ptr = lut_offset.get_device_ptr();
    const auto lut_shapes_ptr = lut_shapes.get_device_ptr();
    auto fragments_ptr = fragments.get_device_ptr();
    auto blendGroups_ptr = blendGroups.get_device_ptr();
    auto blendGroupLen_ptr = blendGroupLen.get_device_ptr();

    //
    // Step 1: iterate over cells to determine sizes of outputs.
    //
    RAJA::ReduceSum<reduce_policy, int> fragment_sum(0);
    RAJA::ReduceSum<reduce_policy, int> blendGroups_sum(0);
    RAJA::ReduceSum<reduce_policy, int> blendGroupLen_sum(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
    {
      int32 start = elid * el_dofs;
      const int32 reorder[] = {0,1,3,2,4,5,7,6};

      // Determine the clip case.
      int32 clipcase = 0;
      for(int32 j = 0; j < el_dofs; j++)
      {
        // dray hex and VisIt hex (tables are geared to VisIt hex) have
        // different node order.
        int32 nidx = (el_dofs == 8) ? reorder[j] : j;

        int32 dofid = conn_ptr[start + nidx];
        if(dist_ptr[dofid][0] > 0.)
            clipcase |= (1 << nidx);
      }

      // Iterate over the shapes and determine how many points and
      // cell fragments we'll make.
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
             (!m_invert && shapes[2] == COLOR0) ||
             (m_invert && shapes[2] == COLOR1))
          {
            // The point is a keeper.

            for(unsigned char ni = 0; ni < shapes[3]; ni++)
            {
              auto ptid = shapes[4 + ni];

              // count the point as used.
              ptused[ptid]++;

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
          if((!m_invert && shapes[1] == COLOR0) ||
             (m_invert && shapes[1] == COLOR1))
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
        // TODO: Oh, we need to handle triangles too for the 2D case...
#if 1
        else
        {
          // Error! This should not happen because
          break;
        }
#endif
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
#endif

    /*
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
    blendGroupOffsets=[0,1,3,5,    7,9,...} (exlusive scan of blendGroupSizes)

    blendNames is made from combining the values in a blend group into a single id.
    For singles, we can use the point id itself. For pairs, we can do <p0:p1>. For
    longer lists, we can sort the point ids and checksum them in some way.

    Each group in the blendData is encoded to a blendName. We can then put
    the blendNames through a uniqueness filter so we can eliminate duplicate
    blend groups.

    origIndex   = [0,   1,       2,      3,          4,        5]
    blendNames  = [{100,0x54321, 0x23456,0x34567}   {0x34567,  106},...]
    uniqueNames = [100, 106, 0x23456, 0x34567, 0x54321}
    u2origIndex = [0, 5, 2, 3, 1]    (3,4 were dups)

    For a point in the cell, get its blend group index in the cell. For example,
    if we're in cell 0 and we get point 0x54321. We want to find its blend group data
    in blendCoeff, blendIds.

    uidx = bsearch(uniqueNames, 0x54321)                   (4)
    obidx = u2origIndex[uidx]                              (1 blend group)
    start = blendGroupOffsets[obidx]                       (1)
    int32 nids = blendGroupSizes[obidx]                    (2)
    int32 *ids = &blendIds[start]                          ({0.5 0.5})
    Float *coeff = &blendCoeff[start]                      ({100 101})

    uidx = bsearch(uniqueNames, 106)                       (1)
    obidx = u2origIndex[uidx]                              (5 blend group)
    start = blendGroupOffsets[obidx]                       (9)
    int32 nids = blendGroupSizes[obidx]                    (1)
    int32 *ids = &blendIds[start]                          ({1.})
    Float *coeff = &blendCoeff[start]                      ({106})

    The bsearch result for the name in the uniqueNames is the global point id to
    use in the connectivity.

    If we wanted to condense the blendCoeff,blendIds based on the uniqueNames, we
    can iterate over all elements in the uniqueNames and use the above algorithm
    figure out a list of blendGroupSizes for each element in uniqueNames. We could
    then scan to make offsets for a new array and do another loop to gather into
    the condensed arrays. Then we'd use the new arrays to blend point data.

    Array<Float>
    blend_array(const Array<int32> &uniqueBlendGroupSizes,
                const Array<int32> &uniqueBlendGroupOffsets,
                const Array<int32> &uniqueBlendCoeff,
                const Array<int32> &uniqueBlendIds,
                const Array<Float> &origField)
    {
      Array<Float> blended;
      size_t n = uniqueBlendGroupSizes.size();
      blended.resize(n);

      auto dest_ptr = blended.get_device_ptr();
      const auto bgSizes_ptr = uniqueBlendGroupSizes.get_device_ptr();
      const auto bgOffsets_ptr = uniqueBlendGroupOffsets.get_device_ptr();
      const auto bgCoeff_ptr = uniqueBlendCoeff.get_device_ptr();
      const auto bgIds_ptr = uniqueBlendIds.get_device_ptr();
      const auto origField_ptr = origField.get_device_ptr();

      RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] DRAY_LAMBDA (int32 i)
      {
        Float sum = 0.;
        int32 start = bgOffsets_ptr[i];
        int32 nvalues = bgSizes_ptr[i] + start;
        for(int32 idx = start; idx < nvalues; idx++)
        {
          sum += bgCoeff_ptr[idx] * origField_ptr[bgIds_ptr[idx]];
        }
        dest_ptr[i] = sum;
      });
    }

    */
#if 1
    //
    // Step 2: Make blendOffset and blendCoeff, blendIds, blendNames.
    //
    Array<int32> blendOffset, blendGroupOffsets, blendIds;
    Array<Float> blendCoeff;
    Array<int32> blendNames;
    blendOffset.resize(nelem);
    blendGroupOffsets.resize(nelem);
    blendIds.resize(blendGroupLen_sum.get());
    blendCoeff.resize(blendGroupLen_sum.get());

    auto blendOffset_ptr = blendOffset.get_device_ptr();
    auto blendGroupOffsets_ptr = blendGroupOffsets.get_device_ptr();
    auto blendIds_ptr = blendIds.get_device_ptr();
    auto blendCoeff_ptr = blendCoeff.get_device_ptr();
    auto blendNames_ptr = blendNames.get_device_ptr();

    // Make offsets via scan.
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(blendGroupLen_ptr, nelem),
                                     RAJA::make_span(blendOffset_ptr, nelem),
                                     RAJA::operators::plus<int>{});
    DRAY_ERROR_CHECK();
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(blendGroups_ptr, nelem),
                                     RAJA::make_span(blendGroupOffsets_ptr, nelem),
                                     RAJA::operators::plus<int>{});
    DRAY_ERROR_CHECK();

    // Populate blendCoeff, blendIds, blendNames.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 elid)
    {
      int32 start = elid * el_dofs;
      const int32 reorder[] = {0,1,3,2,4,5,7,6};
      int32 el_ids[8]; // max p1 ids.

      // Determine the clip case.
      int32 clipcase = 0;
      for(int32 j = 0; j < el_dofs; j++)
      {
        // dray hex and VisIt hex (tables are geared to VisIt hex) have
        // different node order.
        int32 nidx = (el_dofs == 8) ? reorder[j] : j;

        int32 dofid = conn_ptr[start + nidx];
        el_ids[j] = dofid;

        if(dist_ptr[dofid][0] > 0.)
            clipcase |= (1 << nidx);
      }

      // Iterate over the shapes and determine how many points and
      // cell fragments we'll make.
      unsigned char *shapes = &lut_shapes_ptr[lut_offset_ptr[clipcase]];

      // The number of tets (cell fragments) produced for the case. We
      // need this to know overall how many cells there will be in the
      // output.
      int32 thisFragments = 0;

      // The number of blend groups (corners, centers, edges, faces)
      // for this element.
//      int32 thisBlendGroups = 0;

      // The number of ints we need to store the ids for all blend records
      // in this element.
//      int32 thisblendGroupLen = 0;

      // Points used in cell (range [P0,N6])
      unsigned char ptused[50] = {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
      };
      const unsigned char hex_edge_to_corners[12][2] = {
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

      auto make_name_1 = [](int32 id)
      {
          int32 name = 0;
          return name;
      };

      auto make_name_2 = [](int32 id0, int32 id1)
      {
          int32 name = 0;
          return name;
      };

      auto make_name_n = [](const int32 *start, const int32 *end)
      {
          int32 name = 0;
#if 0
def simple_sort(a):
    # Copy input a to b
    v = [0]*len(a)
    for i in range(len(a)):
        v[i] = a[i]

    n = len(v)
    for i in range(n):
        j = i
        while j > 0 and v[j] < v[j-1]:
           tmp = v[j]
           v[j] = v[j-1]
           v[j-1] = tmp
           j = j - 1
    return v

uint32_t jenkins_one_at_a_time_hash(const uint8_t* key, size_t length) {
  size_t i = 0;
  uint32_t hash = 0;
  while (i != length) {
    hash += key[i++];
    hash += hash << 10;
    hash ^= hash >> 6;
  }
  hash += hash << 3;
  hash ^= hash >> 11;
  hash += hash << 15;
  return hash;
}
#endif
          return name;
      };

      // Starting offset of where we store this element's blend groups.
      int32 bgStart = blendGroupOffsets_ptr[elid];
      int32 bgOffset = blendOffset_ptr[elid];

      for(int32 si = 0; si < lut_nshapes_ptr[clipcase]; si++)
      {
        if(shapes[0] == ST_PNT)
        {
          // ST_PNT, 0, COLOR0, 8, P0, P1, P2, P3, P4, P5, P6, P7, 
          if(shapes[2] == NOCOLOR ||
             (!m_invert && shapes[2] == COLOR0) ||
             (m_invert && shapes[2] == COLOR1))
          {
            // The point is a keeper. Add a blend group for this point.
            auto one_over_n = 1.f / static_cast<float>(shapes[3]);
            int32 start = bgStart;

            auto npts = shapes[3];
            for(unsigned char ni = 0; ni < npts; ni++)
            {
              auto ptid = shapes[4 + ni];

              // count the point as used.
              ptused[ptid]++;

              // Increase the blend size to include this center point.
              if(/*ptid >= P0 &&*/ ptid <= P7)
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
                const unsigned char *c = hex_edge_to_corners[ptid - EA];
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

            // Store "name" of blend group.
            auto blendName = make_name_n(blendIds_ptr + start,
                                         blendIds_ptr + start + shapes[3]);
            blendNames_ptr[bgOffset++] = blendName;
          }

          shapes += (4 + shapes[3]);
        }
        else if(shapes[0] == ST_TET)
        {
          // ST_TET COLOR0 p0 p1 p2 p3
          if((!m_invert && shapes[1] == COLOR0) ||
             (m_invert && shapes[1] == COLOR1))
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
        // TODO: Oh, we need to handle triangles too for the 2D case...
#if 1
        else
        {
          // Error! This should not happen because
          break;
        }
#endif
      }

      // Add blend group for each original point that was used.
      for(unsigned char pid = P0; pid <= P7; pid++)
      {
        if(ptused[pid] > 0)
        {
          // Store blend group info                
          blendIds_ptr[bgStart] = el_ids[pid];
          blendCoeff_ptr[bgStart] = 1.;
          bgStart++;

          // Store "name" of blend group.
          blendNames_ptr[bgOffset++] = make_name_1(el_ids[pid]);
        }
      }

      // Add blend group for each edge point that was used.
      for(unsigned char pid = EA; pid <= EL; pid++)
      {
        if(ptused[pid] > 0)
        {
          const unsigned char *c = hex_edge_to_corners[pid - EA];
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
          bgStart += 2;

          // Store "name" of blend group.
          blendNames_ptr[bgOffset++] = make_name_2(id0, id1);
        }
      }
    });
#endif

    // At this point, we have created the blend group data. We can now use the
    // blendNames to make unique blend groups. uNames contains a sorted list of
    // the unique blend group names while uIndices is their original index in
    // blendNames/blendGroupOffsets/blendGroupSizes.
    Array<int32> uNames, uIndices;
    unique(blendNames, uNames, uIndices);


    DRAY_LOG_CLOSE();
  }

  //-------------------------------------------------------------------------
  void
  unique(const Array<int32> &keys_orig, Array<int32> &skeys, Array<int32> &sindices)
  {
    // Make a copy of the values and make original indices.
    Array<int32> keys, indices;
    int32 n = keys_orig.size();
    keys.resize(n);
    indices.resize(n);
    auto keys_orig_ptr = indices.get_device_ptr_const();
    auto keys_ptr = keys.get_device_ptr();
    auto indices_ptr = indices.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] RAJA_HOST_DEVICE (int32 i) {
      keys_ptr[i] = keys_orig_ptr[i];
      indices_ptr[i] = i;
    });

    // Sort the keys, indices in place.
    RAJA::sort_pairs<for_policy>(RAJA::make_span(keys_ptr, n),
                                  RAJA::make_span(indices_ptr, n));

    // Make a mask array for where differences occur.
    Array<int32> mask;
    mask.resize(n);
    auto mask_ptr = mask.get_device_ptr();
    RAJA::ReduceSum<reduce_policy, int32> mask_sum(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] RAJA_HOST_DEVICE (int32 i) {
      int32 m = 1;
      if(i > 1)
         m = (keys_ptr[i] != keys_ptr[i-1]) ? 1 : 0;

      mask_ptr[i] = m;
      mask_sum += m;
    });

    // Do a scan on the mask array to build an offset array.
    Array<int32> offsets;
    offsets.resize(n);
    auto offsets_ptr = offsets.get_device_ptr();
    RAJA::exclusive_scan<for_policy>(RAJA::make_span(mask_ptr, n),
                                      RAJA::make_span(offsets_ptr, n),
                                      RAJA::operators::plus<int>{});

    // Iterate over the mask/offsets to store values at the right
    // offset in the new array.
    int32 newsize = mask_sum.get();
    skeys.resize(newsize);
    sindices.resize(newsize);
    auto skeys_ptr = skeys.get_device_ptr();
    auto sindices_ptr = sindices.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n), [=] RAJA_HOST_DEVICE (int32 i) {
      if(mask_ptr[i])
      {
        skeys_ptr[offsets_ptr[i]] = keys_ptr[i];
        sindices_ptr[offsets_ptr[i]] = indices_ptr[i];
      }
    });
  }

  //-------------------------------------------------------------------------

  template<typename ScalarField>
  ScalarField
  create_distance_field(ScalarField &field) const
  {
    // The size of the field's control idx (connectivity)
    int32 sz = field.get_dof_data().m_size_ctrl;

    // Make a new distance array that is the same size as the input field's m_ctrl_idx.
    Array<Vec<Float,1>> dist;
    dist.resize(sz);
    Vec<Float,1> *dist_ptr = dist.get_device_ptr();
    auto field_ptr = field.get_dof_data().m_values.get_device_ptr();

    // Make the distance field for all the dofs in the input field.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, sz), [=] DRAY_LAMBDA (int32 i)
    {
        dist_ptr[i].m_data[0] = field_ptr[i][0] - m_clip_value;
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
    ScalarField newfield(gf, field.get_poly_order());
    return newfield;
  }
};

// Load Hex lookup data.
template <>
void
ClipFieldLinear::load_lookups(HexMesh_P1 &m,
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


}//namespace detail

ClipField::ClipField() : m_clip_value(0.), m_field_name(), m_invert(false)
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

DataSet
ClipField::execute(DataSet dom)
{
  DataSet output;
  if(dom.mesh() != nullptr && dom.mesh()->order() == 1)
  {
    detail::ClipFieldLinear func(dom, m_clip_value, m_field_name, m_invert);
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
        detail::ClipFieldLinear func(dom, m_clip_value, m_field_name, m_invert);
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
