// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// COMPONENTS
//
// Newton solve
// GetRayAABBs
// CalcShape functor   : Vector (u,v,w,s) -> Vector (x,y,z,f)
// CalcDShape functor  : Vector (u,v,w,s) -> Matrix d[x,y,z,f]/d[u,v,w,s]
// Invert square matrix<1> <2> <3> _N
//

// Array<Vec<T,S>> dof_values
//
//


// INTERFACES

// MFEMMeshField::Isosurface(const Ray &rays, T isovalue);
