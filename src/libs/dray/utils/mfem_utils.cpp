#include <dray/utils/mfem_utils.hpp>
#include <dray/policies.hpp>
#include <mfem.hpp>

namespace dray
{
namespace detail
{

bool is_positive_basis(const mfem::FiniteElementCollection* fec)
{
  // HACK: Check against several common expected FE types

  if(fec == nullptr)
  {
    return false;
  }

  if(const mfem::H1_FECollection* h1Fec =
       dynamic_cast<const mfem::H1_FECollection*>(fec))
  {
    return h1Fec->GetBasisType() == mfem::BasisType::Positive;
  }

  if(const mfem::L2_FECollection* l2Fec =
       dynamic_cast<const mfem::L2_FECollection*>(fec))
  {
    return l2Fec->GetBasisType() == mfem::BasisType::Positive;
  }

  if( dynamic_cast<const mfem::NURBSFECollection*>(fec)       ||
      dynamic_cast<const mfem::LinearFECollection*>(fec)      ||
      dynamic_cast<const mfem::QuadraticPosFECollection*>(fec) )
  {
    return true;
  }

  return false;
}

/*!
 * \brief Utility function to get a positive (i.e. Bernstein)
 * collection of bases corresponding to the given FiniteElementCollection.
 *
 * \return A pointer to a newly allocated FiniteElementCollection
 * corresponding to \a fec
 * \note   It is the user's responsibility to deallocate this pointer.
 * \pre    \a fec is not already positive
 */
mfem::FiniteElementCollection* get_pos_fec(
  const mfem::FiniteElementCollection* fec,
  int order,
  int dim,
  int map_type)
{
  //SLIC_CHECK_MSG( !isPositiveBasis( fec),
  //                "This function is only meant to be called "
  //                "on non-positive finite element collection" );

  // Attempt to find the corresponding positive H1 fec
  if(dynamic_cast<const mfem::H1_FECollection*>(fec))
  {
    return new mfem::H1_FECollection(order, dim, mfem::BasisType::Positive);
  }

  // Attempt to find the corresponding positive L2 fec
  if(dynamic_cast<const mfem::L2_FECollection*>(fec))
  {
    // should we throw a not supported error here?
    return new mfem::L2_FECollection(order, dim, mfem::BasisType::Positive,
                                     map_type);
  }

  // Attempt to find the corresponding quadratic or cubic fec
  // Note: Linear FECollections are positive
  if(dynamic_cast<const mfem::QuadraticFECollection*>(fec) ||
     dynamic_cast<const mfem::CubicFECollection*>(fec) )
  {
    //SLIC_ASSERT( order == 2 || order == 3);
    return new mfem::H1_FECollection(order, dim, mfem::BasisType::Positive);
  }

  // Give up -- return NULL
  return nullptr;
}



/// template <typename T>
/// void grid_function_bounds(const mfem::GridFunction *gf, const int32 refinement, T &lower, T &upper, const int32 _comp)
/// {
///   const int32 comp = _comp - 1;
///   const mfem::FiniteElementSpace *fe_space = gf->FESpace();
///   const mfem::Mesh *mesh = fe_space->GetMesh();
///
///   const int32 num_elts = fe_space->GetNE();
///
///   RAJA::ReduceMin<reduce_cpu_policy, T> field_min(infinity32());
///   RAJA::ReduceMax<reduce_cpu_policy, T> field_max(neg_infinity32());
///
///   // Iterate over all elements.
///   /// RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_elts), [=] (int32 elt_idx)
///   /// {
///
///   // Note: The below usage of mfem objects is not thread safe.
///   //  Using RAJA, even with for_cpu_policy, causes seg faults.
///   //  Whatever gets stored into elt_vals is suspect. Seg fault on elt_vals.Width().
///
///   for (int32 elt_idx = 0; elt_idx < num_elts; elt_idx++)
///   {
///     mfem::DenseMatrix elt_vals;
///     mfem::ElementTransformation *tr = fe_space->GetElementTransformation(elt_idx);
///     mfem::RefinedGeometry *ref_g = mfem::GlobGeometryRefiner.Refine(mesh->GetElementBaseGeometry(elt_idx), refinement);
///     gf->GetVectorValues(*tr, ref_g->RefPts, elt_vals);
///     // Size of elt_vals becomes  sdims x #ref_pts.
///
///     T elt_min = infinity32();
///     T elt_max = neg_infinity32();
///
///     // For each refinement point, compare with current min and max.
///     const int32 num_refpts = elt_vals.Width();
///     //const int32 f_dims = elt_vals.Height();    // Needed in the vector min/max method, not this one.
///     for (int32 j = 0; j < num_refpts; j++)
///     {
///       elt_min = (elt_vals(comp, j) < elt_min) ? elt_vals(comp, j) : elt_min;
///       elt_max = (elt_vals(comp, j) > elt_max) ? elt_vals(comp, j) : elt_max;
///     }
///     // Update overall min/max.
///     field_min.min(elt_min);
///     field_max.max(elt_max);
///   }
///
///   /// });
///
///   lower = field_min.get();
///   upper = field_max.get();
/// }
/// template void grid_function_bounds(const mfem::GridFunction *gf, const int32 refinement, float32 &lower, float32 &upper, const int32 comp);
/// template void grid_function_bounds(const mfem::GridFunction *gf, const int32 refinement, float64 &lower, float64 &upper, const int32 comp);


} // namespace detail
} // namespace dray
