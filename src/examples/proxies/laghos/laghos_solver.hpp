// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_LAGHOS_SOLVER
#define MFEM_LAGHOS_SOLVER

#include "mfem.hpp"
#include "laghos_assembly.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace hydrodynamics
{

/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

// These are defined in laghos.cpp
double rho0(const Vector &);
void v0(const Vector &, Vector &);
double e0(const Vector &);
double gamma(const Vector &);

struct TimingData
{
   // Total times for all major computations:
   // CG solves (H1 and L2) / force RHS assemblies / quadrature computations.
   StopWatch sw_cgH1, sw_cgL2, sw_force, sw_qdata;

   // These accumulate the total processed dofs or quad points:
   // #(CG iterations) for the H1 CG solve.
   // #dofs  * #(CG iterations) for the L2 CG solve.
   // #quads * #(RK sub steps) for the quadrature data computations.
   int H1cg_iter, L2dof_iter, quad_tstep;

   TimingData()
      : H1cg_iter(0), L2dof_iter(0), quad_tstep(0) { }
};

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1FESpace;
   ParFiniteElementSpace &L2FESpace;

   Array<int> &ess_tdofs;

   const int dim, nzones, l2dofs_cnt, h1dofs_cnt, source_type;
   const double cfl;
   const bool use_viscosity, p_assembly;
   const double cg_rel_tol;
   const int cg_max_iter;
   Coefficient *material_pcf;

   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   DenseTensor Me_inv;

   // Integration rule for all assemblies.
   const IntegrationRule &integ_rule;

   // Data associated with each quadrature point in the mesh. These values are
   // recomputed at each time step.
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current;

   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it is used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm Force;

   // Same as above, but done through partial assembly.
   ForcePAOperator ForcePA;

   // Mass matrices done through partial assembly:
   // velocity (coupled H1 assembly) and energy (local L2 assemblies).
   mutable MassPAOperator VMassPA;
   mutable DiagonalSolver VMassPA_prec;
   mutable LocalMassPAOperator locEMassPA;

   // Linear solver for energy.
   CGSolver locCG;

   mutable TimingData timer;

   virtual void ComputeMaterialProperties(int nvalues, const double gamma[],
                                          const double rho[], const double e[],
                                          double p[], double cs[]) const
   {
      for (int v = 0; v < nvalues; v++)
      {
         p[v]  = (gamma[v] - 1.0) * rho[v] * e[v];
         cs[v] = sqrt(gamma[v] * (gamma[v]-1.0) * e[v]);
      }
   }

   void UpdateQuadratureData(const Vector &S) const;

public:
   LagrangianHydroOperator(int size, ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &l2_fes,
                           Array<int> &essential_tdofs, ParGridFunction &rho0,
                           int source_type_, double cfl_,
                           Coefficient *material_, bool visc, bool pa,
                           double cgt, int cgiter);

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_estimate.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const { quad_data_is_current = false; }

   // The density values, which are stored only at some quadrature points, are
   // projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho);

   void PrintTimingData(bool IamRoot, int steps);

   ~LagrangianHydroOperator();
};

class TaylorCoefficient : public Coefficient
{
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
   }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
