#include<Kripke/Kernel/Kernel_3d_ZDG.h>
#include<Kripke/Grid.h>
#include<Kripke/SubTVec.h>

Nesting_Order Kernel_3d_ZDG::nestingPsi(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZDG::nestingPhi(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZDG::nestingSigt(void) const {
  return NEST_DZG;
}

Nesting_Order Kernel_3d_ZDG::nestingEll(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZDG::nestingEllPlus(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_ZDG::nestingSigs(void) const {
  return NEST_DGZ;
}


void Kernel_3d_ZDG::LTimes(Grid_Data *grid_data) {
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
    int num_groups = sdom.phi->groups;
    int num_zones = sdom.num_zones;
    int num_local_groups = sdom.num_groups;
    int group0 = sdom.group0;
    int num_local_directions = sdom.num_directions;

    /* 3D Cartesian Geometry */
    double * KRESTRICT ell_d_ptr = sdom.ell->ptr();

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for (int z = 0; z < num_zones; z++) {
      double * KRESTRICT psi = sdom.psi->ptr(0, 0, z);
      double * KRESTRICT ell_d = ell_d_ptr;

      for (int d = 0; d < num_local_directions; d++) {
        double * KRESTRICT phi = sdom.phi->ptr(group0, 0, z);

        for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
          double ell_d_nm = ell_d[nm_offset];

          for (int group = 0; group < num_local_groups; ++group) {
            phi[group] += ell_d_nm * psi[group];
          }
          phi += num_groups;
        }
        ell_d += nidx;
        psi += num_local_groups;
      }
    }

  } // Subdomain
}

void Kernel_3d_ZDG::LPlusTimes(Grid_Data *grid_data) {
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

    // Get Variables
    sdom.rhs->clear(0.0);

    /* 3D Cartesian Geometry */
    double * KRESTRICT ell_plus_ptr = sdom.ell_plus->ptr();

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for (int z = 0; z < num_zones; z++) {
      double * KRESTRICT rhs = sdom.rhs->ptr(0, 0, z);

      double * ell_plus_d = ell_plus_ptr;
      for (int d = 0; d < num_local_directions; d++) {

        double * KRESTRICT phi_out = sdom.phi_out->ptr(group0, 0, z);

        for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
          double ell_plus_d_n_m = ell_plus_d[nm_offset];

          for (int group = 0; group < num_local_groups; ++group) {
            rhs[group] += ell_plus_d_n_m * phi_out[group];
          }
          phi_out += num_groups;
        }
        rhs += num_local_groups;
        ell_plus_d += nidx;
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
void Kernel_3d_ZDG::scattering(Grid_Data *grid_data){
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
      double *sigs_mat = sigs[material];
      double *phi_z_nm = phi.ptr() + zone*num_groups*num_moments;
      double *phi_out_z_nm = phi_out.ptr() + zone*num_groups*num_moments;

      for(int nm = 0;nm < num_moments;++ nm){
        // map nm to n
        int n = moment_to_coeff[nm];
        double *sigs_n_g = sigs_mat + n*num_groups*num_groups;

        for(int g = 0;g < num_groups;++ g){
          double *phi_out_z_gp = phi_out.ptr() + zone*num_groups*num_moments;

          for(int gp = 0;gp < num_groups;++ gp){
            phi_out_z_nm[gp] += sigs_n_g[gp] * phi_z_nm[g] * fraction;
          }
          sigs_n_g += num_groups;
        }
        phi_z_nm += num_groups;
        phi_out_z_nm += num_groups;
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
void Kernel_3d_ZDG::source(Grid_Data *grid_data){
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
        double *phi_out_z_nm0 = phi_out.ptr() + zone*num_moments*num_groups;
        for(int g = 0;g < num_groups;++ g){
          phi_out_z_nm0[g] += 1.0 * fraction;
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

void Kernel_3d_ZDG::sweep(Subdomain *sdom) {
  int num_directions = sdom->num_directions;
  int num_groups = sdom->num_groups;
  int num_zones = sdom->num_zones;

  Directions *direction = sdom->directions;

  int local_imax = sdom->nzones[0];
  int local_jmax = sdom->nzones[1];
  int local_kmax = sdom->nzones[2];

  double * dx = &sdom->deltas[0][0];
  double * dy = &sdom->deltas[1][0];
  double * dz = &sdom->deltas[2][0];

  // Upwind/Downwind face flux data
  SubTVec &i_plane = *sdom->plane_data[0];
  SubTVec &j_plane = *sdom->plane_data[1];
  SubTVec &k_plane = *sdom->plane_data[2];

  // All directions have same id,jd,kd, since these are all one Direction Set
  // So pull that information out now
  Grid_Sweep_Block const &extent = sdom->sweep_block;

  for (int k = extent.start_k; k != extent.end_k; k += extent.inc_k) {
    double dzk = dz[k + 1];
    for (int j = extent.start_j; j != extent.end_j; j += extent.inc_j) {
      double dyj = dy[j + 1];
      for (int i = extent.start_i; i != extent.end_i; i += extent.inc_i) {
        double dxi = dx[i + 1];

        int z = Zonal_INDEX(i, j, k);
        double * KRESTRICT sigt_z = sdom->sigt->ptr(0, 0, z);

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
        for (int d = 0; d < num_directions; ++d) {
          double xcos = direction[d].xcos;
          double ycos = direction[d].ycos;
          double zcos = direction[d].zcos;

          double zcos_dzk = 2.0 * zcos / dzk;
          double ycos_dyj = 2.0 * ycos / dyj;
          double xcos_dxi = 2.0 * xcos / dxi;

          double * KRESTRICT psi_z_d = sdom->psi->ptr(0, d, z);
          double * KRESTRICT rhs_z_d = sdom->rhs->ptr(0, d, z);

          double * KRESTRICT psi_lf_z_d = i_plane.ptr(0, d, I_PLANE_INDEX(j, k));
          double * KRESTRICT psi_fr_z_d = j_plane.ptr(0, d, J_PLANE_INDEX(i, k));
          double * KRESTRICT psi_bo_z_d = k_plane.ptr(0, d, K_PLANE_INDEX(i, j));

          for (int group = 0; group < num_groups; ++group) {
            /* Calculate new zonal flux */
            double psi_z_d_g = (rhs_z_d[group]
                + psi_lf_z_d[group] * xcos_dxi
                + psi_fr_z_d[group] * ycos_dyj
                + psi_bo_z_d[group] * zcos_dzk)
                / (xcos_dxi + ycos_dyj + zcos_dzk + sigt_z[group]);

            psi_z_d[group] = psi_z_d_g;

            /* Apply diamond-difference relationships */
            psi_z_d_g *= 2.0;
            psi_lf_z_d[group] = psi_z_d_g - psi_lf_z_d[group];
            psi_fr_z_d[group] = psi_z_d_g - psi_fr_z_d[group];
            psi_bo_z_d[group] = psi_z_d_g - psi_bo_z_d[group];
          }
        }
      }
    }
  }
}

