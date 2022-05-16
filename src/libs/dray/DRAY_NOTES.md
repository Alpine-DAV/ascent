# Devil Ray README : Some useful information

2018-08-10 Masado Ishii

## Spaces / Math

    Each element parameterizes a region of space by transforming reference coordinates to world coordinates. I'll use parametric coordinates (u,v,w) for element reference space, and world coordinates (x,y,z) for world space. Each element owns a transformation from its reference space to world space, \PHI: [0,1]^3 -> R^3, \PHI(u,v,w) = (x,y,z).

    For now we assume that \PHI is a trivariate polynomial (trivariate == 3 function of variables).

    A polynomial function can be represented in different bases. A basis is a special set of functions (we are talking about a linear basis in the vector space of polynomial functions). A given polynomial is a linear combination of the basis polynomials, which means each basis function has an associated coefficient, and then everything is added into a single function. From high school algebra, one usually uses the power basis, which looks like (1+10x+25x2). The special functions of the power basis are {1,x,x2,x3,...,xn}, and the coefficients are (1,10,25). There are other bases. Another is the Bernstein basis. Right now Devil Ray uses the trivariate Bernstein basis.

    I do not fully explain the Bernstein basis here, but here are some properties we are interested in:

    * Any polynomial evaluated in the unit cube will have a value within the convex hull of the set of coefficients relative to the Bernstein basis. This is a consequence of the "partition of unity" property of the Bernstein basis.
    * In the univariate Berntein basis, the 0th and nth coefficients are the values of the polynomial at the endpoints of the interval [0,1], respectively.
    * The trivariate Bernstein basis we use is a tensor product of three 1D Bernstein bases over different variables. A (p,p,p)-order polynomial has (p+1)^3 coefficients, which can be arrayed in a cube lattice.
    * If you take a trivariate polynomial and restrict one of the variables to an endpoint (say you fix y=1), you get a bivariate polynomial whose coefficients in the Bernstein basis are a subset of the coefficients from the original polynomial. They are exactly the coefficients at the face of the cube lattice corresponding to the selected face of the reference space cube.

    Representing a high-order mesh: Assume a particular basis, such as the trivariate Bernstein basis of order (p,p,p), for some integer power p. A polynomial function of order p (or less) has a unique representation in this basis. To represent an element transformation in the Bernstein basis, one needs the coefficients corresponding to the basis functions. Because an element transformation has vector values, the coefficients are vectors. Equivalently, each spatial component can be considered a separate scalar trivariate polynomial.

    Aside from the element transformation, any scalar field defined on an element can be represented using another set of coefficients. The coefficients for a scalar field will be scalars. (The polynomial basis for a scalar field need not be the same order, or even the same basis, as that used for the element transformation. But it should be if the element is so-called "isoparametric.")

    In the finite element method, the coefficients of basis polynomials are also called "degrees of freedom." Sometimes (e.g., in H1 continuous spaces), adjacent elements share degrees of freedom along their edges.

    Thus, to represent a mesh, we have to store two arrays:

    * a set of degrees of freedom (array of Vec3f); and
    * a set of local lookup tables (array of int32).

    Each local lookup table is a map from element coefficient positions, to global degree-of-freedom positions.

    MFEM uses a representation like this. Devil Ray essentially uses the same representation, but note that I have enforced different assumptions about the order of things in DRay.


## Some algorithms and explanation

Algorithm: _Point location_

Input: Geometry (dray::MeshField), World points (dray::Array<dray::Vec3f>)

Output: Element id and reference point for each input world point. (dray::Array<dray::int32>, dray::Array<dray::Vec3f>)

1. RAJA::forall<>({each point P})

    1. BVH traversal. Before we can worry about reference coordinates, we need an element. BVH traversal gives -> list of candidates.

    2. foreach(candidate)

        1. Put together an ElTrans object that specifies the polynomial order and element coefficients

            1. The polynomial order is specified somewhere in the mesh (I have assumed that all elements in the mesh have the same order).

            2. The polynomial evaluator component of ElTrans is BernsteinBasis. BernsteinBasis requires a pointer to a read/writable sub-array of main memory in order to evaluate the 1D Bernstein basis polynomials.

            3. The coefficient-iterator part of ElTrans requires pointers to the coefficient value array, coefficient lookup array, and the element index of the current candidate.

            // Result: The ElTrans object now has enough information to transform any reference point to a world space point.

        2. Solve the equation \PHI(u,v,w) = P, using the Newton-Raphson method.

            1. Set an initial guess. (For now, it is the middle, (.5,.5,.5).)

            2. For each iteration of Newton's method

                // *Note: Here I have used Q and U, but in the code they are y and x.*

                1. Evaluate Q = \PHI(u,v,w) and its partial derivatives at (u,v,w).
                2. Compute \delta Q = (P - Q). Compare Q with target P. If close enough, exit.
                3. Otherwise, perform the Newton step.
                    1. Construct a matrix J (the jacobian of \PHI) from the partial derivatives of \PHI that we evaluated earlier.
                    2. Compute the reference increment by solving J \delta U = \delta Q for \delta U. This uses either matrix inversion or LU_solve().
                    3. In case of a singularity, exit.  //TODO might be good to add a return type of error.
                    4. Otherwise, add the increment \delta U to (u,v,w).
                    5. If the increment \delta U barely changed, exit because we either have arrived or never will.

             // Result: A new reference point (u',v',w').

        3. If the Newton-Raphson was successful then use the current candidate (a point shall be contained in at most one element).
        //TODO this test is "!=NotConverged," but maybe it should be "==ConvergePhys"

        4. Otherwise, continue with the next candidate.

    // Result: Either (el_id, ref_pt), or (-1, __).


## The Data Structures

(I have done summaries for these)

+ ElTransData
+ ElTransIter, ElTransBdryIter
+ BinomRow
+ BernsteinBasis
+ ElTransOp, ElTransPairOp, ElTransRayOp

(I haven't done summaries for the list below)

+ NewtonSolve::solve()
+ Intersector_PointVol, Intersector_RayIsosurf, (Intersector_RayBoundSurf)
+ MeshField


### ElTransData

Arrays where the data is stored. Sizes of the arrays. Number of degrees of freedom per element, in order to segment the array "m_ctrl_idx."


### ElTransIter, ElTransBdryIter

Iterator over the degrees of freedom for a particular element or face. Other methods need the values of the element dofs, e.g. dof-multiplication (linear_combo). By overloading the operator[], ElTrans(Bdry)Iter hides the double-indirection local_dof->global_dof->value.

In some cases the double-indirection may not be necessary. For example, for fields which are discontinuous at element boundaries, every element requires its own instance of all its degrees of freedom. In this case, you could define ElTransData without m_ctrl_idx, and you could define an ElTransIter that is simply the identity function. That is, you could optimize this case, with minimal code changes outside of these two classes.

ElTransBdryIter is a special case to jumpstart the task of boundary intersection. Right now ElTransBdryIter assumes 3D Hex elements with dofs sequenced in a certain order. (It's untested, so, fingers crossed basically.)

ElTransIter (generally, CoefficientIterType) is packed into ElTransOp via composition.


### BinomRow::fill_single_row()

Computes the binomial coefficients (N choose k), for fixed N and variable 0<=k<=N. Applies a formula to sequentially fill (N+1) memory locations in O(N) steps.

The binomial coefficients (N choose k) are used in the Bernstein basis functions.

For other ways to compute binomial coefficients, see the rest of binomial.hpp. These methods are not currently in use, but they could be useful later.

+ BinomRowT offers a way to access binomial coefficients with no runtime computations. They are provided as constexpr values via template metaprogramming. But you have to know N at compile time. Could be useful for optimization when polynomial order, or a bound on polynomial order, is known at compile time.

+ GlobBinomTable offers a way to read binomial coefficients after a one-time computation. Whenever GlobBinomTable is built or expanded, it must be done from host code, and the maximum N desired must be known at that time. If a larger N is needed later, it is possible to expand GlobBinomTable, but only from host code.


### BernsteinBasis

Evaluates a multivariate polynomial by streaming in the coefficients. (The coefficient are relative to the Bernstein basis.) The coefficients can be vector-valued. BernsteinBasis requires an external CoefficientIterator (ElTransIter).

BernsteinBasis also requires a small memory buffer (aux_mem) in order to evaluate the 1D basis function and derivative for each component of a reference point. (We take advantage of the tensor structure of the 3D basis functions to save memory as O(p) rather than O(p^3)). The static method get_aux_req() tells you how many sizeof(T)-size elements to allocate per query.

To initialize BernsteinBasis, it needs a pointer to the allocated aux_mem, and the 1d-polynomial order. The aux_mem pointer is for an individual query, so it must be offset by the sizes of preceding aux_mem segments.

BernsteinBasis (generally, ShapeOpType) is packed into ElTransOp via inheritance.

__Assumptions__

+ A multivariate polynomial has the same degree for each component of reference point. That is, 3D polynomial order is always (p,p,p), not (p,q,r).

+ I define a particular sequence for the degrees of freedom.
    - For a single reference component u, the degrees of freedom are sequenced in order of increasing u.
    - In a tensor product over multiple reference components (u,v,w), the degrees of freedom are sequenced like (U0,V0,W0), (U0,V0,W1), ..., (U0,V0,Wp), (U0,V1,W0), ..., ..., (U0,Vp,Wp), (U1,V0,W0), ..., ..., ..., (Up,Vp,Wp). That is, the first reference coordinate is treated as the most significant digit.
    - The adapter in mfem2dray knows about this sequence.

__Implementation__

The 1d basis polynomials evaluation routines are copied from MFEM almost verbatim. The (p+1) basis functions and their derivatives are evaluated together. There are two phases, up and down. "Up" accumulates powers of x, whilst dropping prefix-products into increasing memory addresses. "Down" accumulates powers of (1-x), whilst dropping prefix products into decreasing memory addresses.

It makes sense to evaluate in this way in a serial program for which memory abounds. It avoids having to repeatedly compute successive powers of x. However, if we chose to repeatedly compute successive powers of x, we could do away with the aux_mem read/write buffer.


### ElTransOp, ElTransPairOp, ElTransRayOp

ElTransOp is an interface that represents a transformation: {reference space}->{world space} or a field {reference space}->{field value}. Essentially ElTransOp is a wrapper that unifies ElTransIter (coefficient streaming, generally CoefficientIterType), and BernsteinBasis (polynomial evaluation & combining, generally ShapeOpType). The unified ElTransOp::eval() interface allows us to treat ElTransOp as a stateful functor. For example it is possible to template NewtonSolver on different types of transformations, which is precisely what I do.

ElTransPairOp is a generalization of ElTransOp which represents the concatenation of two functions: {reference space}->{X(u,v,w), Y(u,v,w)}. The two functions must take in the same reference space. However, there is great freedom: The X function and Y function need not have the same polynomial order, or even the same ShapeOp type. ElTransPairOp::eval() first calls the eval() of X, then the eval() of Y. I use ElTransPairOp to concatenate an element transformation \PHI(u,v,w) with a scalar field F(u,v,w) when I do ray-isosurface intersection.

ElTransRayOp is a special combination of a ray with either ElTransOp or ElTransPairOp (template <class ElTransOpType>). It adds a virtual reference coordinate to the reference space of ElTransOpType. The extra coordinate represents distance along the ray; I define 'evaluating' a ray R at distance s to mean, calculate (dir * s). The function represented by ElTransRayOp is (\PHI(u,v,w) - R(s)). (orig does not appear here, but is supplied by the intersection methods to NewtonSolver). I use ElTransRayOp<ElTransPairOp> to implement ray-isosurface intersection. In this case it is the final interface I hand to NewtonSolver.

**In order to implement ray-boundary intersection**, combine ElTransRayOp<ElTransOp<BernsteinBasis<RefDim=2>, ElTransBdryIter<PhysDim=3>>>


### Other Idiosyncracies

RefDim and PhysDim: RefDim is the dimensionality of reference space. I saw no reason to lock things to 3D when the only material difference between 3D and ND is the number of iterations of a for loop. Sometimes ND means 2D, for example faces in the element boundary intersector. PhysDim is the dimensionality of a vector-valued field or transformation. For scalar-valued fields, PhysDim==1. I probably should have named PhysDim something else.

space_dim and field_dim: They are different instances of PhysDim. Typically space_dim is 3, for the 3D vector-valued element transformation, and field_dim is 1, for scalar fields.
















