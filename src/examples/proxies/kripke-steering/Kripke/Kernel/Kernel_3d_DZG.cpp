#include<Kripke/Kernel/Kernel_3d_DZG.h>
#include<Kripke/Grid.h>
#include<Kripke/SubTVec.h>

Nesting_Order Kernel_3d_DZG::nestingPsi(void) const {
  return NEST_DZG;
}

Nesting_Order Kernel_3d_DZG::nestingPhi(void) const {
  return NEST_DZG;
}

Nesting_Order Kernel_3d_DZG::nestingSigt(void) const {
  return NEST_DZG;
}

Nesting_Order Kernel_3d_DZG::nestingEll(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_DZG::nestingEllPlus(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_DZG::nestingSigs(void) const {
  return NEST_DGZ;
}


void Kernel_3d_DZG::LTimes(Grid_Data *grid_data) {
  // Outer parameters
  int nidx = grid_data->total_num_moments;

  for(int ds = 0;ds < grid_data->num_zone_sets;++ ds){
    grid_data->phi[ds]->clear(0.0);
  }

  // Loop over Subdomains
  int num_subdomains = grid_data->subdomains.size();
  for (int sdom_id = 0; sdom_id < num_subdomains; ++ sdom_id){
    Subdomain &sdom = grid_data->subdomains[sdom_id];

    // Get dimensioning
    int num_zones = sdom.num_zones;
    int num_groups = sdom.phi->groups;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;

    /* 3D Cartesian Geometry */
    double * KRESTRICT ell = sdom.ell->ptr();
    double * KRESTRICT phi_ptr = sdom.phi->ptr(group0, 0, 0);
    for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
      double * KRESTRICT psi_ptr = sdom.psi->ptr();

      for (int d = 0; d < num_local_directions; d++) {
        double ell_nm_d = ell[d];


#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
        for (int z = 0;z < num_zones;++ z){
          double * KRESTRICT phi = phi_ptr + num_groups*z;
          double * KRESTRICT psi = psi_ptr + num_local_groups*z;

          for(int group = 0;group < num_local_groups; ++ group){
            phi[group] += ell_nm_d * psi[group];
          }

        }
        psi_ptr += num_zones*num_local_groups;
      }
      ell += num_local_directions;
      phi_ptr += num_groups*num_zones;
    }

  } // Subdomain
}

void Kernel_3d_DZG::LPlusTimes(Grid_Data *grid_data) {
  // Outer parameters
  int nidx = grid_data->total_num_moments;

  // Loop over Subdomains
  int num_subdomains = grid_data->subdomains.size();
  for (int sdom_id = 0; sdom_id < num_subdomains; ++ sdom_id){
    Subdomain &sdom = grid_data->subdomains[sdom_id];

    // Get dimensioning
    int num_zones = sdom.num_zones;
    int num_groups = sdom.phi_out->groups;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;
    int num_groups_zones = num_local_groups*num_zones;

    // Get Variables
    sdom.rhs->clear(0.0);

    /* 3D Cartesian Geometry */
    double * KRESTRICT ell_plus = sdom.ell_plus->ptr();

    for (int d = 0; d < num_local_directions; d++) {
      double * KRESTRICT phi_out_ptr = sdom.phi_out->ptr(group0, 0, 0);
      double * KRESTRICT rhs_ptr = sdom.rhs->ptr(0, d, 0);

      for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
        double ell_plus_d_nm = ell_plus[nm_offset];

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
        for(int z = 0;z < num_zones;++ z){
          double * KRESTRICT rhs = rhs_ptr + num_local_groups*z;
          double * KRESTRICT phi_out = phi_out_ptr + num_groups*z;


          for(int group = 0;group < num_local_groups;++ group){
            rhs[group] += ell_plus_d_nm * phi_out[group];
          }
        }
        phi_out_ptr += num_groups*num_zones;
      }
      ell_plus += nidx;
    }
  } // Subdomain
}


/**
  Compute scattering source term phi_out from flux moments in phi.
  phi_out(gp,z,nm) = sum_g { sigs(g, n, gp) * phi(g,z,nm) }

  we are mapping sigs(g,d,z) to mean:
    g=source group
    d=legendre coeff
    z=destination group
*/
void Kernel_3d_DZG::scattering(Grid_Data *grid_data){
  // Loop over zoneset subdomains
  for(int zs = 0;zs < grid_data->num_zone_sets;++ zs){
    // get the phi and phi out references
    SubTVec &phi = *grid_data->phi[zs];
    SubTVec &phi_out = *grid_data->phi_out[zs];
    SubTVec &sigs0 = *grid_data->sigs[0];
    SubTVec &sigs1 = *grid_data->sigs[1];
    SubTVec &sigs2 = *grid_data->sigs[2];

    // get material mix information
    int sdom_id = grid_data->zs_to_sdomid[zs];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    int const * KRESTRICT mixed_to_zones = &sdom.mixed_to_zones[0];
    int const * KRESTRICT mixed_material = &sdom.mixed_material[0];
    double const * KRESTRICT mixed_fraction = &sdom.mixed_fraction[0];

    // Zero out source terms
    phi_out.clear(0.0);

    // grab dimensions
    int num_mixed = sdom.mixed_to_zones.size();
    int num_zones = sdom.num_zones;
    int num_groups = phi.groups;
    int num_moments = grid_data->total_num_moments;
    int const * KRESTRICT moment_to_coeff = &grid_data->moment_to_coeff[0];

    double *phi_nm = phi.ptr();
    double *phi_out_nm = phi_out.ptr();
    for(int nm = 0;nm < num_moments;++ nm){
      // map nm to n
      int n = moment_to_coeff[nm];
      double *sigs_n[3] = {
          sigs0.ptr() + n*num_groups*num_groups,
          sigs1.ptr() + n*num_groups*num_groups,
          sigs2.ptr() + n*num_groups*num_groups
      };

      for(int mix = 0;mix < num_mixed;++ mix){
        int zone = mixed_to_zones[mix];
        int material = mixed_material[mix];
        double fraction = mixed_fraction[mix];
        double *sigs_n_g = sigs_n[material];
        double *phi_nm_z = phi_nm + zone*num_groups;
        double *phi_out_nm_z = phi_out_nm + zone*num_groups;

        for(int g = 0;g < num_groups;++ g){
          double phi_nm_z_g = phi_nm_z[g];

          for(int gp = 0;gp < num_groups;++ gp){
            phi_out_nm_z[gp] += sigs_n_g[gp] * phi_nm_z_g * fraction;
          }
          sigs_n_g += num_groups;
        }
      }
      phi_nm += num_zones*num_groups;
      phi_out_nm += num_zones*num_groups;
    }
  }
}

/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */
void Kernel_3d_DZG::source(Grid_Data *grid_data){
  // Loop over zoneset subdomains
  for(int zs = 0;zs < grid_data->num_zone_sets;++ zs){
    // get the phi and phi out references
    SubTVec &phi_out = *grid_data->phi_out[zs];

    // get material mix information
    int sdom_id = grid_data->zs_to_sdomid[zs];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    int const * KRESTRICT mixed_to_zones = &sdom.mixed_to_zones[0];
    int const * KRESTRICT mixed_material = &sdom.mixed_material[0];
    double const * KRESTRICT mixed_fraction = &sdom.mixed_fraction[0];

    // grab dimensions
    int num_mixed = sdom.mixed_to_zones.size();
    int num_zones = sdom.num_zones;
    int num_groups = phi_out.groups;
    int num_moments = grid_data->total_num_moments;

    double *phi_out_nm0 = phi_out.ptr();
    for(int mix = 0;mix < num_mixed;++ mix){
      int zone = mixed_to_zones[mix];
      int material = mixed_material[mix];
      double fraction = mixed_fraction[mix];
      double *phi_out_nm0_z = phi_out_nm0 + zone*num_groups;

      if(material == 0){
        for(int g = 0;g < num_groups;++ g){
          phi_out_nm0_z[g] += 1.0 * fraction;
        }
      }
    }
  }
}


/* Sweep routine for Diamond-Difference */
/* Macros for offsets with fluxes on cell faces */
#define I_PLANE_INDEX(j, k) (k)*(local_jmax) + (j)
#define J_PLANE_INDEX(i, k) (k)*(local_imax) + (i)
#define K_PLANE_INDEX(i, j) (j)*(local_imax) + (i)
#define Zonal_INDEX(i, j, k) (i) + (local_imax)*(j) \
  + (local_imax)*(local_jmax)*(k)

void Kernel_3d_DZG::sweep(Subdomain *sdom) {
  int num_directions = sdom->num_directions;
  int num_groups = sdom->num_groups;
  int num_zones = sdom->num_zones;

  Directions *direction = sdom->directions;

  int local_imax = sdom->nzones[0];
  int local_jmax = sdom->nzones[1];
  int local_kmax = sdom->nzones[2];
  int local_imax_1 = local_imax + 1;
  int local_jmax_1 = local_jmax + 1;

  double *dx = &sdom->deltas[0][0];
  double *dy = &sdom->deltas[1][0];
  double *dz = &sdom->deltas[2][0];

  // Upwind/Downwind face flux data
  SubTVec &i_plane = *sdom->plane_data[0];
  SubTVec &j_plane = *sdom->plane_data[1];
  SubTVec &k_plane = *sdom->plane_data[2];

  // All directions have same id,jd,kd, since these are all one Direction Set
  // So pull that information out now
  Grid_Sweep_Block const &extent = sdom->sweep_block;

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
  for (int d = 0; d < num_directions; ++d) {
    double xcos = direction[d].xcos;
    double ycos = direction[d].ycos;
    double zcos = direction[d].zcos;

    /*  Perform transport sweep of the grid 1 cell at a time.   */
    for (int k = extent.start_k; k != extent.end_k; k += extent.inc_k) {
      double dzk = dz[k + 1];
      double zcos_dzk = 2.0 * zcos / dzk;
      for (int j = extent.start_j; j != extent.end_j; j += extent.inc_j) {
        double dyj = dy[j + 1];
        double ycos_dyj = 2.0 * ycos / dyj;
        for (int i = extent.start_i; i != extent.end_i; i += extent.inc_i) {
          double dxi = dx[i + 1];
          double xcos_dxi = 2.0 * xcos / dxi;

          int z = Zonal_INDEX(i, j, k);
          double * KRESTRICT psi_d_z = sdom->psi->ptr(0, d, z);
          double * KRESTRICT rhs_d_z = sdom->rhs->ptr(0, d, z);

          double * KRESTRICT psi_lf_d_z = i_plane.ptr(0, d, I_PLANE_INDEX(j, k));
          double * KRESTRICT psi_fr_d_z = j_plane.ptr(0, d, J_PLANE_INDEX(i, k));
          double * KRESTRICT psi_bo_d_z = k_plane.ptr(0, d, K_PLANE_INDEX(i, j));

          double * KRESTRICT sigt_z = sdom->sigt->ptr(0, 0, z);

          for (int group = 0; group < num_groups; ++group) {
            /* Calculate new zonal flux */
            double psi_d_z_g = (rhs_d_z[group]
                + psi_lf_d_z[group] * xcos_dxi
                + psi_fr_d_z[group] * ycos_dyj
                + psi_bo_d_z[group] * zcos_dzk)
                / (xcos_dxi + ycos_dyj + zcos_dzk + sigt_z[group]);

            psi_d_z[group] = psi_d_z_g;

            /* Apply diamond-difference relationships */
            psi_lf_d_z[group] = 2.0 * psi_d_z_g - psi_lf_d_z[group];
            psi_fr_d_z[group] = 2.0 * psi_d_z_g - psi_fr_d_z[group];
            psi_bo_d_z[group] = 2.0 * psi_d_z_g - psi_bo_d_z[group];
          }
        }
      }
    }
  }
}

