#include<Kripke/Kernel/Kernel_3d_GDZ.h>
#include<Kripke/Grid.h>
#include<Kripke/SubTVec.h>

Nesting_Order Kernel_3d_GDZ::nestingPsi(void) const {
  return NEST_GDZ;
}

Nesting_Order Kernel_3d_GDZ::nestingPhi(void) const {
  return NEST_GDZ;
}

Nesting_Order Kernel_3d_GDZ::nestingSigt(void) const {
  return NEST_DGZ;
}

Nesting_Order Kernel_3d_GDZ::nestingEll(void) const {
  return NEST_ZGD;
}

Nesting_Order Kernel_3d_GDZ::nestingEllPlus(void) const {
  return NEST_ZDG;
}

Nesting_Order Kernel_3d_GDZ::nestingSigs(void) const {
  return NEST_GZD;
}


void Kernel_3d_GDZ::LTimes(Grid_Data *grid_data) {
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
    double * KRESTRICT phi = sdom.phi->ptr(group0, 0, 0);
    double * KRESTRICT psi_ptr = sdom.psi->ptr();

    for (int group = 0; group < num_local_groups; ++group) {
      double * KRESTRICT ell_nm = sdom.ell->ptr();

      for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
        double * KRESTRICT psi = psi_ptr;

        for (int d = 0; d < num_local_directions; d++) {
          double ell_nm_d = ell_nm[d];

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
          for(int z = 0;z < num_zones; ++ z){
            phi[z] += ell_nm_d * psi[z];
          }

          psi += num_zones;
        }
        ell_nm += num_local_directions;
        phi += num_zones;
      }
      psi_ptr += num_zones*num_local_directions;
    }
  } // Subdomain
}

void Kernel_3d_GDZ::LPlusTimes(Grid_Data *grid_data) {
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
    double *ell_plus_ptr = sdom.ell_plus->ptr();

    double * KRESTRICT phi_out_ptr = sdom.phi_out->ptr(group0, 0, 0);
    double * KRESTRICT rhs = sdom.rhs->ptr();

    for (int group = 0; group < num_local_groups; ++group) {
      double *ell_plus = ell_plus_ptr;

      for (int d = 0; d < num_local_directions; d++) {
        double * KRESTRICT phi_out = phi_out_ptr;

        for(int nm_offset = 0;nm_offset < nidx;++nm_offset){
          double ell_plus_d_nm = ell_plus[nm_offset];

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
          for(int z = 0;z < num_zones; ++ z){
            rhs[z] += ell_plus_d_nm * phi_out[z];
          }
          phi_out += num_zones;
        }
        ell_plus += nidx;
        rhs += num_zones;
      }
      phi_out_ptr += num_zones*nidx;
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
void Kernel_3d_GDZ::scattering(Grid_Data *grid_data){
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
    int num_coeff = grid_data->legendre_order+1;
    int const * KRESTRICT moment_to_coeff = &grid_data->moment_to_coeff[0];

    double *phi_g = phi.ptr();
    double *sigs0_g_gp = sigs0.ptr();
    double *sigs1_g_gp = sigs1.ptr();
    double *sigs2_g_gp = sigs2.ptr();
    for(int g = 0;g < num_groups;++ g){

      double *phi_out_gp_nm = phi_out.ptr();
      for(int gp = 0;gp < num_groups;++ gp){

        double *phi_g_nm = phi_g;
        for(int nm = 0;nm < num_moments;++ nm){
          // map nm to n
          int n = moment_to_coeff[nm];
          double sigs_g_gp_n[3] = {sigs0_g_gp[n], sigs1_g_gp[n], sigs2_g_gp[n]};

          for(int mix = 0;mix < num_mixed;++ mix){
            int zone = mixed_to_zones[mix];
            int material = mixed_material[mix];
            double fraction = mixed_fraction[mix];
            double sigs_value = sigs_g_gp_n[material];

            phi_out_gp_nm[zone] +=  sigs_value * phi_g_nm[zone] * fraction;
          }

          phi_g_nm += num_zones;
          phi_out_gp_nm += num_zones;
        }
        sigs0_g_gp += num_coeff;
        sigs1_g_gp += num_coeff;
        sigs2_g_gp += num_coeff;
      }
      phi_g +=  num_moments*num_zones;
    }
  }
}


/**
 * Add an isotropic source, with flux of 1, to every zone with Region 1
 * (or material 0).
 *
 * Since it's isotropic, we're just adding this to nm=0.
 */
void Kernel_3d_GDZ::source(Grid_Data *grid_data){
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

    double *phi_out_g_nm0 = phi_out.ptr();
    for(int g = 0;g < num_groups;++ g){
      for(int mix = 0;mix < num_mixed;++ mix){
        int zone = mixed_to_zones[mix];
        int material = mixed_material[mix];
        double fraction = mixed_fraction[mix];

        if(material == 0){
          phi_out_g_nm0[zone] += 1.0 * fraction;
        }
      }
      phi_out_g_nm0 += num_moments * num_zones;
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

void Kernel_3d_GDZ::sweep(Subdomain *sdom) {
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

#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
  for (int group = 0; group < num_groups; ++group) {

    std::vector<double> xcos_dxi_all(local_imax);
    std::vector<double> ycos_dyj_all(local_jmax);
    std::vector<double> zcos_dzk_all(local_kmax);
    double * KRESTRICT sigt_g = sdom->sigt->ptr(group, 0, 0);

    for (int d = 0; d < num_directions; ++d) {
      double * KRESTRICT psi_g_d = sdom->psi->ptr(group, d, 0);
      double * KRESTRICT rhs_g_d = sdom->rhs->ptr(group, d, 0);
      double * KRESTRICT i_plane_g_d = i_plane.ptr(group, d, 0);
      double * KRESTRICT j_plane_g_d = j_plane.ptr(group, d, 0);
      double * KRESTRICT k_plane_g_d = k_plane.ptr(group, d, 0);

      double xcos = direction[d].xcos;
      double ycos = direction[d].ycos;
      double zcos = direction[d].zcos;
      for (int i = 0; i < local_imax; ++i) {
        double dxi = dx[i + 1];
        xcos_dxi_all[i] = 2.0 * xcos / dxi;
      }

      for (int j = 0; j < local_jmax; ++j) {
        double dyj = dy[j + 1];
        ycos_dyj_all[j] = 2.0 * ycos / dyj;
      }

      for (int k = 0; k < local_kmax; ++k) {
        double dzk = dz[k + 1];
        zcos_dzk_all[k] = 2.0 * zcos / dzk;
      }

      /*  Perform transport sweep of the grid 1 cell at a time.   */
      for (int k = extent.start_k; k != extent.end_k; k += extent.inc_k) {
        double zcos_dzk = zcos_dzk_all[k];
        for (int j = extent.start_j; j != extent.end_j; j += extent.inc_j) {
          double ycos_dyj = ycos_dyj_all[j];
          int z_idx = Zonal_INDEX(extent.start_i, j, k);
          for (int i = extent.start_i; i != extent.end_i; i += extent.inc_i) {
            double xcos_dxi = xcos_dxi_all[i];

            /* Calculate new zonal flux */
            double psi_g_d_z = (rhs_g_d[z_idx]
                + i_plane_g_d[I_PLANE_INDEX(j, k)] * xcos_dxi
                + j_plane_g_d[J_PLANE_INDEX(i, k)] * ycos_dyj
                + k_plane_g_d[K_PLANE_INDEX(i, j)] * zcos_dzk)
                / (xcos_dxi + ycos_dyj + zcos_dzk
                    + sigt_g[z_idx]);
            psi_g_d[z_idx] = psi_g_d_z;

            /* Apply diamond-difference relationships */
            i_plane_g_d[I_PLANE_INDEX(j, k)] = 2.0 * psi_g_d_z
                - i_plane_g_d[I_PLANE_INDEX(j, k)];
            j_plane_g_d[J_PLANE_INDEX(i, k)] = 2.0 * psi_g_d_z
                - j_plane_g_d[J_PLANE_INDEX(i, k)];
            k_plane_g_d[K_PLANE_INDEX(i, j)] = 2.0 * psi_g_d_z
                - k_plane_g_d[K_PLANE_INDEX(i, j)];

            z_idx += extent.inc_i;
          }
        }
      }
    }
  }

}

