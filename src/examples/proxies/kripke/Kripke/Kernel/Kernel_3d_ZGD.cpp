#include<Kripke/Kernel/Kernel_3d_ZGD.h>
#include<Kripke/Grid.h>
#include<Kripke/SubTVec.h>

Nesting_Order Kernel_3d_ZGD::nestingPsi(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_ZGD::nestingPhi(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_ZGD::nestingSigt(void) const {
  return NEST_DZG;
}

Nesting_Order Kernel_3d_ZGD::nestingEll(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZGD::nestingEllPlus(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZGD::nestingSigs(void) const {
  return NEST_GZD;
}

void Kernel_3d_ZGD::LTimes(Grid_Data *grid_data) {
  // Outer parameters
  int nidx = grid_data->total_num_moments;

  // Clear phi
  for(int ds = 0;ds < grid_data->num_zone_sets;++ ds){
    grid_data->phi[ds]->clear(0.0);
  }
  // Loop over Subdomains
  int num_subdomains = grid_data->subdomains.size();
  for (int sdom_id = 0; sdom_id < num_subdomains; ++ sdom_id){
    Subdomain &sdom = grid_data->subdomains[sdom_id];

    // Get dimensioning
    int num_zones = sdom.num_zones;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;

    /* 3D Cartesian Geometry */
    double * KRESTRICT ell_ptr = sdom.ell->ptr();

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int z = 0;z < num_zones;++ z){
      double * KRESTRICT psi = sdom.psi->ptr(0, 0, z);
      double * KRESTRICT phi = sdom.phi->ptr(group0, 0, z);
      for(int group = 0;group < num_local_groups;++ group){
        double * KRESTRICT ell_d = ell_ptr;

        for (int d = 0; d < num_local_directions; d++) {
          double psi_d = psi[d];

          for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
            phi[nm_offset] += ell_d[nm_offset] * psi_d;
          }
          ell_d += nidx;
        }

        psi += num_local_directions;
        phi += nidx;
      }
    }

  } // Subdomain
}

void Kernel_3d_ZGD::LPlusTimes(Grid_Data *grid_data) {
  // Outer parameters
  int nidx = grid_data->total_num_moments;

  // Loop over Subdomains
  int num_subdomains = grid_data->subdomains.size();
  for (int sdom_id = 0; sdom_id < num_subdomains; ++ sdom_id){
    Subdomain &sdom = grid_data->subdomains[sdom_id];

    // Get dimensioning
    int num_zones = sdom.num_zones;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;

    // Get Variables
    sdom.rhs->clear(0.0);

    /* 3D Cartesian Geometry */
    double * KRESTRICT ell_plus_ptr = sdom.ell_plus->ptr();


#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int z = 0;z < num_zones;++ z){
      double * KRESTRICT rhs = sdom.rhs->ptr(0, 0, z);
      double * KRESTRICT phi_out = sdom.phi_out->ptr(group0,0, z);
      for(int group = 0;group < num_local_groups;++ group){
        double * KRESTRICT ell_plus_d = ell_plus_ptr;

        for (int d = 0; d < num_local_directions; d++) {
          double rhs_acc = 0.0;

          for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
             rhs_acc += ell_plus_d[nm_offset] * phi_out[nm_offset];
          }
          rhs[d] += rhs_acc;

          ell_plus_d += nidx;
        }
        rhs += num_local_directions;
        phi_out += nidx;
      }
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
void Kernel_3d_ZGD::scattering(Grid_Data *grid_data){
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
    int num_coeff = grid_data->legendre_order+1;
    int num_moments = grid_data->total_num_moments;
    int const * KRESTRICT moment_to_coeff = &grid_data->moment_to_coeff[0];

    double *sigs[3] = {
        sigs0.ptr(),
        sigs1.ptr(),
        sigs2.ptr()
    };

    for(int mix = 0;mix < num_mixed;++ mix){
      int zone = mixed_to_zones[mix];
      int material = mixed_material[mix];
      double fraction = mixed_fraction[mix];
      double *sigs_g_gp = sigs[material];
      double *phi_z_g = phi.ptr() + zone*num_groups*num_moments;

      for(int g = 0;g < num_groups;++ g){
        double *phi_out_z_gp = phi_out.ptr() + zone*num_groups*num_moments;

        for(int gp = 0;gp < num_groups;++ gp){
          for(int nm = 0;nm < num_moments;++ nm){
            // map nm to n
            int n = moment_to_coeff[nm];

            phi_out_z_gp[nm] += sigs_g_gp[n] * phi_z_g[nm] * fraction;
          }
          sigs_g_gp += num_coeff;
          phi_out_z_gp += num_moments;
        }
        phi_z_g += num_moments;
      }
    }
  }
}

/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */
void Kernel_3d_ZGD::source(Grid_Data *grid_data){
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

    for(int mix = 0;mix < num_mixed;++ mix){
      int zone = mixed_to_zones[mix];
      int material = mixed_material[mix];
      double fraction = mixed_fraction[mix];

      if(material == 0){
        double *phi_out_z = phi_out.ptr() + zone*num_moments*num_groups;
        for(int g = 0;g < num_groups;++ g){
          phi_out_z[g*num_moments] += 1.0 * fraction;
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

void Kernel_3d_ZGD::sweep(Subdomain *sdom) {
  int num_directions = sdom->num_directions;
  int num_groups = sdom->num_groups;
  int num_zones = sdom->num_zones;

  Directions *direction = sdom->directions;

  int local_imax = sdom->nzones[0];
  int local_jmax = sdom->nzones[1];
  int local_kmax = sdom->nzones[2];

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

  /*  Perform transport sweep of the grid 1 cell at a time.   */

  for (int k = extent.start_k; k != extent.end_k; k += extent.inc_k) {
    double dzk = dz[k + 1];
    double two_dz = 2.0 / dzk;
    for (int j = extent.start_j; j != extent.end_j; j += extent.inc_j) {
      double dyj = dy[j + 1];
      double two_dy = 2.0 / dyj;
      for (int i = extent.start_i; i != extent.end_i; i += extent.inc_i) {
        double dxi = dx[i + 1];
        double two_dx = 2.0 / dxi;

        int z = Zonal_INDEX(i, j, k);
        double * sigt_z = sdom->sigt->ptr(0, 0, z);
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
        for (int group = 0; group < num_groups; ++group) {
          double * KRESTRICT psi_z_g = sdom->psi->ptr(group, 0, z);
          double * KRESTRICT rhs_z_g = sdom->rhs->ptr(group, 0, z);

          double * KRESTRICT psi_lf_z_g = i_plane.ptr(group, 0, I_PLANE_INDEX(j, k));
          double * KRESTRICT psi_fr_z_g = j_plane.ptr(group, 0, J_PLANE_INDEX(i, k));
          double * KRESTRICT psi_bo_z_g = k_plane.ptr(group, 0, K_PLANE_INDEX(i, j));

          for (int d = 0; d < num_directions; ++d) {

            double xcos = direction[d].xcos;
            double ycos = direction[d].ycos;
            double zcos = direction[d].zcos;

            double zcos_dzk = zcos * two_dz;
            double ycos_dyj = ycos * two_dy;
            double xcos_dxi = xcos * two_dx;

            /* Calculate new zonal flux */
            double psi_z_g_d = (rhs_z_g[d]
                + psi_lf_z_g[d] * xcos_dxi
                + psi_fr_z_g[d] * ycos_dyj
                + psi_bo_z_g[d] * zcos_dzk)
                / (xcos_dxi + ycos_dyj + zcos_dzk + sigt_z[group]);

            psi_z_g[d] = psi_z_g_d;

            /* Apply diamond-difference relationships */
            psi_lf_z_g[d] = 2.0 * psi_z_g_d - psi_lf_z_g[d];
            psi_fr_z_g[d] = 2.0 * psi_z_g_d - psi_fr_z_g[d];
            psi_bo_z_g[d] = 2.0 * psi_z_g_d - psi_bo_z_g[d];
          }
        }
      }
    }
  }
}

