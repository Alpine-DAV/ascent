/*--------------------------------------------------------------------------
 * Header file for the Input_Variables structure
 *--------------------------------------------------------------------------*/

#ifndef KRIPKE_INPUT_VARIABLES_H__
#define KRIPKE_INPUT_VARIABLES_H__

#include<Kripke.h>

/**
 * This structure defines the input parameters to setup a problem.
 */

struct Input_Variables {
  int npx, npy, npz;            // The number of processors in x,y,z
  int nx, ny, nz;               // Number of spatial zones in x,y,z
  int num_dirsets_per_octant;
  int num_dirs_per_dirset;
  int num_groupsets;
  int num_groups_per_groupset;  //
  int num_zonesets_dim[3];      // number of zoneset in x, y, z
  int niter;                    // number of solver iterations to run
  int legendre_order;           // Scattering order (number Legendre coeff's - 1)
  int layout_pattern;           // Which subdomain/task layout to use
  int quad_num_polar;           // Number of polar quadrature points
  int quad_num_azimuthal;       // Number of azimuthal quadrature points
  ParallelMethod parallel_method;
  double sigt[3];               // total cross section for 3 materials
  double sigs[3];               // total scattering cross section for 3 materials
#ifdef KRIPKE_USE_SILO
  std::string silo_basename;    // name prefix for silo output files
#endif

  Nesting_Order nesting;        // Data layout and loop ordering (of Psi)
};

#endif
