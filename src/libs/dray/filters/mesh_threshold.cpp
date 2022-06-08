#include <dray/filters/mesh_threshold.hpp>

#include <dray/dispatcher.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/elem_utils.hpp>
#include <dray/data_model/elem_ops.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{

// OLD CODE - DELETE AS WE GO.
#if 0
//---------------------------------------------------------------------------
// extract_face_dofs<Tensor>
template <int32 ndof>
GridFunction<ndof> extract_face_dofs(const ShapeHex,
                                     const GridFunction<ndof> &orig_data_3d,
                                     const int32 poly_order,
                                     const Array<Vec<int32, 2>> &elid_faceid)
{
  // copy_face_dof_subset
  // - If true, make a new array filled with just the surface geometry
  //   The dof ids for the surface mesh will no longer correspond to
  //   their ids in the volume mesh.
  // - If false, the old geometry array will be re-used, so dof ids
  //   will match the volume mesh.
  const bool copy_face_dof_subset = false;

  GridFunction<ndof> new_data_2d;

  if (copy_face_dof_subset)    // New geometry array with subset of dofs.
  {
    DRAY_ERROR("extract_face_dofs() with copy_face_dof_subset==true is NotImplemented.");
    //TODO
  }
  else                         // Re-use the old geometry, make new topology.
  {
    // Make sure to initialize all 5 members. TODO a new constructor?
    new_data_2d.m_el_dofs = (poly_order+1)*(poly_order+1);
    new_data_2d.m_size_el = elid_faceid.size();
    new_data_2d.m_size_ctrl = orig_data_3d.m_size_ctrl;
    new_data_2d.m_values = orig_data_3d.m_values;

    new_data_2d.m_ctrl_idx.resize((poly_order+1)*(poly_order+1) * elid_faceid.size());

    const Vec<int32,2> * elid_faceid_ptr = elid_faceid.get_device_ptr_const();
    const int32 * orig_dof_idx_ptr       = orig_data_3d.m_ctrl_idx.get_device_ptr_const();
    int32 * new_dof_idx_ptr              = new_data_2d.m_ctrl_idx.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, new_data_2d.m_size_el), [=] DRAY_LAMBDA (int32 face_idx)
    {
      //
      // Convention for face ids and lexicographic ordering of dofs:
      //
      // =========   =========   =========    =========   =========   =========
      // faceid 0:   faceid 1:   faceid 2:    faceid 3:   faceid 4:   faceid 5:
      // z^          z^          y^           z^          z^          y^
      //  |(4) (6)    |(4) (5)    |(2) (3)     |(5) (7)    |(6) (7)    |(6) (7)
      //  |           |           |            |           |           |
      //  |(0) (2)    |(0) (1)    |(0) (1)     |(1) (3)    |(2) (3)    |(4) (5)
      //  ------->    ------->    ------->     ------->    ------->    ------->
      // (x=0)   y   (y=0)   x   (z=0)   x    (X=1)   y   (Y=1)   x   (Z=1)   x
      //

      const int32 eldofs0 = 1;
      const int32 eldofs1 = eldofs0 * (poly_order+1);
      const int32 eldofs2 = eldofs1 * (poly_order+1);
      const int32 eldofs3 = eldofs2 * (poly_order+1);
      const int32 axis_strides[3] = {eldofs0, eldofs1, eldofs2};

      const int32 faceid      = elid_faceid_ptr[face_idx][1];
      const int32 orig_offset = elid_faceid_ptr[face_idx][0] * eldofs3;
      const int32 new_offset  = face_idx * eldofs2;

      const int32 face_axis = (faceid == 0 || faceid == 3 ? 0    // Conditional
                             : faceid == 1 || faceid == 4 ? 1    // instead of
                                                          : 2);  //    % /

      const int32 face_start = (faceid < 3 ? 0 : (eldofs1 - 1) * axis_strides[face_axis]);
      const int32 major_stride = axis_strides[(face_axis == 2 ? 1 : 2)];
      const int32 minor_stride = axis_strides[(face_axis == 0 ? 1 : 0)];

      for (int32 ii = 0; ii < eldofs1; ii++)
        for (int32 jj = 0; jj < eldofs1; jj++)
          new_dof_idx_ptr[new_offset + eldofs1*ii + jj] =
              orig_dof_idx_ptr[orig_offset + face_start + major_stride*ii + minor_stride*jj];
    });
    DRAY_ERROR_CHECK();
  }

  return new_data_2d;
}


// extract_face_dofs<Simplex>
template <int32 ndof>
GridFunction<ndof> extract_face_dofs(const ShapeTet,
                                     const GridFunction<ndof> &orig_data_3d,
                                     const int32 poly_order,
                                     const Array<Vec<int32, 2>> &elid_faceid)
{
  const int32 eldofs2 = (poly_order + 1) / 1 * (poly_order + 2) / 2;
  const int32 eldofs3 = eldofs2 * (poly_order + 3) / 3;

  // copy_face_dof_subset
  // - If true, make a new array filled with just the surface geometry
  //   The dof ids for the surface mesh will no longer correspond to
  //   their ids in the volume mesh.
  // - If false, the old geometry array will be re-used, so dof ids
  //   will match the volume mesh.
  const bool copy_face_dof_subset = false;

  GridFunction<ndof> new_data_2d;

  if (copy_face_dof_subset)    // New geometry array with subset of dofs.
  {
    DRAY_ERROR("extract_face_dofs() with copy_face_dof_subset==true is NotImplemented.");
    //TODO
  }
  else                         // Re-use the old geometry, make new topology.
  {
    // Make sure to initialize all 5 members. TODO a new constructor?
    new_data_2d.m_el_dofs = eldofs2;
    new_data_2d.m_size_el = elid_faceid.size();
    new_data_2d.m_size_ctrl = orig_data_3d.m_size_ctrl;
    new_data_2d.m_values = orig_data_3d.m_values;

    new_data_2d.m_ctrl_idx.resize(eldofs2 * elid_faceid.size());

    const Vec<int32,2> * elid_faceid_ptr = elid_faceid.get_device_ptr_const();
    const int32 * orig_dof_idx_ptr       = orig_data_3d.m_ctrl_idx.get_device_ptr_const();
    int32 * new_dof_idx_ptr              = new_data_2d.m_ctrl_idx.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, new_data_2d.m_size_el), [=] DRAY_LAMBDA (int32 face_idx)
    {
      // The reference tetrahedron. Vertex v3 is at the origin.
      //
      //  Front:
      //          (z)
      //          v2
      //         /. \_
      //        / .  \_
      //       /.v3.  \_
      //     v0______`_v1
      //   (x)          (y)
      //
      //
      //   =========     =========      =========      =========
      //   face id 0     face id 1      face id 2      face id 3
      //
      //   (2)           (2)            (1)            (1)
      //    z             z              y              y
      //    |\            |\             |\             |\
      //    | \           | \            | \            | \
      //    o__y          o__x           o__x           z__x
      //  (3)  (1)      (3)  (0)       (3)  (0)       (2)  (0)
      //

      const int32 faceid      = elid_faceid_ptr[face_idx][1];
      const int32 orig_offset = elid_faceid_ptr[face_idx][0] * eldofs3;
      const int32 new_offset  = face_idx * eldofs2;

      const uint8 p = (uint8) poly_order;
      uint8 b[4];      // barycentric indexing
      b[faceid] = 0;

      uint8 pi[3];  // permutation depending on faceid
      switch (faceid)
      {
        case 0: pi[0] = 1;  pi[1] = 2;  pi[2] = 3;  break;
        case 1: pi[0] = 0;  pi[1] = 2;  pi[2] = 3;  break;
        case 2: pi[0] = 0;  pi[1] = 1;  pi[2] = 3;  break;
        case 3: pi[0] = 0;  pi[1] = 1;  pi[2] = 2;  break;
        // TODO throw an error otherwise
      }
      // Note that pi[] != faceid, so b[faceid] is always 0.

      for (uint8 jj = 0; jj <= p; jj++)
      {
        b[pi[1]] = jj;
        for (int8 ii = 0; ii <= p - jj; ii++)
        {
          b[pi[0]] = ii;
          b[pi[2]] = p - ii - jj;

          new_dof_idx_ptr[new_offset + cartesian_to_tri_idx(ii, jj, p+1)] =
              orig_dof_idx_ptr[orig_offset + cartesian_to_tet_idx(b[0], b[1], b[2], p+1)];
        }
      }
    });
    DRAY_ERROR_CHECK();
  }

  return new_data_2d;
}


template<class MElemT>
DataSet
boundary_execute(UnstructuredMesh<MElemT> &mesh, Array<Vec<int32, 2>> &elid_faceid_state)
{
  constexpr ElemType etype = MElemT::get_etype();

  using OutMeshElement = Element<MElemT::get_dim()-1, 3, MElemT::get_etype (), MElemT::get_P ()>;

  UnstructuredMesh<MElemT> orig_mesh = mesh;
  const int32 mesh_poly_order = orig_mesh.order();

  //
  // Step 1: Extract the boundary mesh: Matt's external_faces() algorithm.
  //

  // Identify unique/external faces.
  Array<Vec<int32,4>> face_corner_ids = detail::extract_faces(orig_mesh);
  Array<int32> orig_face_idx = detail::sort_faces(face_corner_ids);
  detail::unique_faces(face_corner_ids, orig_face_idx);
  elid_faceid_state = detail::reconstruct<etype>(orig_face_idx);

  // Copy the dofs for each face.
  // The template argument '3u' means 3 components (embedded in 3D).
  GridFunction<3u> mesh_data_2d
      = detail::extract_face_dofs(Shape<3, etype>{},
                                  orig_mesh.get_dof_data(),
                                  mesh_poly_order,
                                  elid_faceid_state);

  // Wrap the mesh data inside a mesh and dataset.
  UnstructuredMesh<OutMeshElement> boundary_mesh(mesh_data_2d, mesh_poly_order);

  DataSet out_data_set(std::make_shared<UnstructuredMesh<OutMeshElement>>(boundary_mesh));

  return out_data_set;
}

template <typename FElemT>
std::shared_ptr<Field>
boundary_field_execute(UnstructuredField<FElemT> &in_field,
                       const Array<int32> &elid,
                       const Array<int32> &dofid)
{
  //
  // Step 2: For each field, add boundary field to the boundary_dataset.
  //
  // We already know what kind of elements we have
  constexpr int32 in_dim = FElemT::get_dim();
  constexpr int32 ncomp = FElemT::get_ncomp();
  constexpr ElemType etype = FElemT::get_etype();
  constexpr int32 P = FElemT::get_P();

  const std::string fname = in_field.name();
  const int32 field_poly_order = in_field.order();

  GridFunction<FElemT::get_ncomp()> out_data
      = detail::extract_face_dofs(Shape<3, etype>{},
                                  in_field.get_dof_data(),
                                  field_poly_order,
                                  elid_faceid_state);

  // Reduce dimension, keep everything else the same as input.
  using OutFElemT = Element<in_dim-1, ncomp, etype, P>;

  std::shared_ptr<UnstructuredField<OutFElemT>> out_field
    = std::make_shared<UnstructuredField<OutFElemT>>(out_data, field_poly_order);
  out_field->name(fname);

  return out_field;
}


struct ThresholdFieldFunctor
{
  const Array<int32> m_elid;
  const Array<int32> m_dofid; // nodeids.

  std::shared_ptr<Field> m_output;

  BoundaryFieldFunctor(const Array<int32> m_elid, const Array<int32> m_dofid)
    : m_elid{elid}, m_dofid{dofid}
  { }

  template <typename FieldType>
  void operator()(FieldType &in_field)
  {
    m_output = boundary_field_execute(in_field, m_elid, m_dofid);
  }
};
//---------------------------------------------------------------------------
#endif

// Iterate over the thresh_field to determine a set of element and dof ids that
// we need to preserve in the output.

/*
template <typename MElemT>
void
determine_ids(UnstructuredMesh<MElemT> &mesh, UnstructuredField<MElemT> &thresh_field,
    Array<int32> &m_elid, Array<int32> &m_dofid)
*/
template <typename MeshType, typename FieldType>
void
determine_ids(MeshType &mesh, FieldType &thresh_field,
    Array<int32> &m_elid, Array<int32> &m_dofid)
{
// TODO: write this.
#if 0
    if(thresh_field->order() == 0)
    {
        // cell-centered.

        // dofs is a GridFunction.
        auto dofs = thresh_field->get_dof_data();

        m_elid.resize(dofs.get_num_elem());
        cout << "dofs.get_num_elem()=" << dofs.get_num_elem() << endl;


        int32 n = dofs.get_num_elem();
        for(int32 elem = 0; elem < n; elem++)
        {
            Range node_range;
            get_elt_node_range(coeff, elem, &node_range);
        }

    }
    else
    {
        // node/dof-centered.
    }

    // create the base grid func
    GridFunction<1> gf;
    gf.m_values    = values;
    gf.m_ctrl_idx  = ctrl_idx;
    gf.m_el_dofs   = num_dofs_per_elem;
    gf.m_size_el   = num_elems;
    gf.m_size_ctrl = ctrl_idx.size();

    std::shared_ptr<Field> field;
    field = std::make_shared<UnstructuredField<HexScalar_P0>>(gf, order, field_name);
#endif
}

// Applies a threshold operation on a mesh.
struct ThresholdFunctor
{
  // Keep a handle to the original dataset because we need it to be able to
  // access the other fields.
  DataSet m_input;

  // Output dataset produced by the functor.
  DataSet m_output;

  // Internal storage to pass set of element and dof ids to different stages.
  Array<int32> m_elid;
  Array<int32> m_dofid;

  // Threshold attributes.
  std::string m_field_name;
  Range m_range;
  bool m_return_all_in_range;

  ThresholdFunctor(DataSet &input, const std::string &field_name,
     const Range range, bool return_all_in_range)
    : m_input(input), m_output(), m_field_name(field_name),
      m_range(range), m_return_all_in_range(return_all_in_range)
  {
  }

  // Execute the filter for the input mesh across all possible mesh,field types.
  void execute()
  {
    // This iterates over the product of possible mesh and scalar field types
    // to call the operator() function that can operate on concrete types.
    Field *field = m_input.field(m_field_name);
    if(field != nullptr && field->components() == 1)
      dispatch(m_input.mesh(), field, *this);
  }

  // This method gets invoked by dispatch, which will have converted the Mesh
  // into a concrete derived type like UnstructuredMesh<Hex_P1> so this method
  // is able to call methods on the derived type with no virtual calls.
  template<typename MeshType, typename ScalarField>
  void operator()(MeshType &mesh, ScalarField &field)
  {
    DRAY_LOG_OPEN("mesh_threshold");

    // Figure out which dofs and cells to keep from the input field.
    determine_ids(mesh, field, m_elid, m_dofid);
#if 0
        // Make the mesh dataset, keeping only the cells in m_elid
        m_output = threshold_execute(mesh, m_elid);

    // Then we extract the fields and add them onto the dataset.

    // Iterate over the rest of the fields and 

    const int32 num_fields = m_input.number_of_fields();
    for (int32 field_idx = 0; field_idx < num_fields; field_idx++)
    {
      Field * field = m_input.field(field_idx);
      ThresholdFieldFunctor tff(m_elid, m_dofid);
      try
      {
        dispatch(m_input.mesh(), field, tff);
        m_output.add_field(tff.m_output);
      }
      catch (const DRayError &dispatch_excpt)
      {
        std::cerr << "Boundary: Field '" << field->name() << "' not supported. Skipping. "
                  << "Reason: " << dispatch_excpt.GetMessage() << "\n";
      }
    }
#endif
    DRAY_LOG_CLOSE();
  }
};


}//namespace detail


MeshThreshold::MeshThreshold() : m_range(), m_field_name(), 
  m_return_all_in_range(false)
{
}

MeshThreshold::~MeshThreshold()
{
}

void
MeshThreshold::set_upper_threshold(Float value)
{
  m_range.set_min(value);
}

void
MeshThreshold::set_lower_threshold(Float value)
{
  m_range.set_max(value);
}

void
MeshThreshold::set_field(const std::string &field_name)
{
  m_field_name = field_name;
}

void
MeshThreshold::set_all_in_range(bool value)
{
  m_return_all_in_range = value;
}

Float
MeshThreshold::get_upper_threshold() const
{
  return m_range.max();
}

Float
MeshThreshold::get_lower_threshold() const
{
  return m_range.min();
}

bool
MeshThreshold::get_all_in_range() const
{
  return m_return_all_in_range;
}

const std::string &
MeshThreshold::get_field() const
{
  return m_field_name;
}

Collection
MeshThreshold::execute(Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dom = collection.domain(i);
    detail::ThresholdFunctor func(dom, m_field_name, m_range, m_return_all_in_range);
    func.execute();
    res.add_domain(func.m_output);
  }
  return res;
}


}//namespace dray
