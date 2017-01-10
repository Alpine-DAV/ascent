#ifndef KRIPKE_SUBTVEC_H__
#define KRIPKE_SUBTVEC_H__

#include <Kripke/Kernel.h>
#include <algorithm>
#include <vector>
#include <stdlib.h>

/**
 *  A transport vector (used for Psi and Phi, RHS, etc.)
 *
 *  This provides the inner most three strides of
 *    Psi[GS][DS][G][D][Z]
 *  but in whatever nesting order is specified.
 */
struct SubTVec {
  SubTVec(Nesting_Order nesting, int ngrps, int ndir_mom, int nzones):
    groups(ngrps),
    directions(ndir_mom),
    zones(nzones),
    elements(groups*directions*zones),
    data_linear(elements)
  {
    setupIndices(nesting, &data_linear[0]);
  }


  /**
   * ALIASING version of constructor.
   * Use this when you have a data buffer already, and don't want this class
   * to do any memory management.
   */
  SubTVec(Nesting_Order nesting, int ngrps, int ndir_mom, int nzones, double *ptr):
    groups(ngrps),
    directions(ndir_mom),
    zones(nzones),
    elements(groups*directions*zones),
    data_linear(0)
  {
    setupIndices(nesting, ptr);
  }

  ~SubTVec(){
  }

  void setupIndices(Nesting_Order nesting, double *ptr){
    // setup nesting order
    switch(nesting){
      case NEST_GDZ:
        ext_to_int[0] = 0;
        ext_to_int[1] = 1;
        ext_to_int[2] = 2;
        break;
      case NEST_GZD:
        ext_to_int[0] = 0;
        ext_to_int[2] = 1;
        ext_to_int[1] = 2;
        break;
      case NEST_DZG:
        ext_to_int[1] = 0;
        ext_to_int[2] = 1;
        ext_to_int[0] = 2;
        break;
      case NEST_DGZ:
        ext_to_int[1] = 0;
        ext_to_int[0] = 1;
        ext_to_int[2] = 2;
        break;
      case NEST_ZDG:
        ext_to_int[2] = 0;
        ext_to_int[1] = 1;
        ext_to_int[0] = 2;
        break;
      case NEST_ZGD:
        ext_to_int[2] = 0;
        ext_to_int[0] = 1;
        ext_to_int[1] = 2;
        break;
    }

    // setup dimensionality
    int size_ext[3];
    size_ext[0] = groups;
    size_ext[1] = directions;
    size_ext[2] = zones;

    // map to internal indices
    for(int i = 0; i < 3; ++i){
      size_int[ext_to_int[i]] = size_ext[i];
    }

    data_pointer = ptr;
  }

  inline double* ptr(void){
    return data_pointer;
  }

  inline double* ptr(int g, int d, int z){
    return &(*this)(g,d,z);
  }

  // These are NOT efficient.. just used to re-stride data for comparisons
  inline double &operator()(int g, int d, int z) {
    int idx[3];
    idx[ext_to_int[0]] = g;
    idx[ext_to_int[1]] = d;
    idx[ext_to_int[2]] = z;
    int offset = idx[0] * size_int[1]*size_int[2] +
                 idx[1] * size_int[2] +
                 idx[2];
    return data_pointer[offset];
  }
  inline double operator()(int g, int d, int z) const {
    return (*const_cast<SubTVec*>(this))(g,d,z);
  }

  inline double sum(void) const {
    double s = 0.0;
    for(size_t i = 0;i < elements;++ i){
      s+= data_linear[i];
    }
    return s;
  }

  inline void clear(double v){
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0;i < elements;++ i){
      data_linear[i] = v;
    }
  }

  inline void randomizeData(void){
    for(int i = 0;i < elements;++ i){
      data_linear[i] = drand48();
    }
  }

  inline void copy(SubTVec const &b){
    for(int g = 0;g < groups;++ g){
      for(int d = 0;d < directions; ++ d){
        for(int z = 0;z < zones;++ z){
          // Copy using abstract indexing
          (*this)(g,d,z) = b(g,d,z);
        }
      }
    }
  }

  inline bool compare(std::string const &name, SubTVec const &b,
      double tol, bool verbose){

    bool is_diff = false;
    int num_wrong = 0;
    for(int g = 0;g < groups;++ g){
      for(int d = 0;d < directions; ++ d){
        for(int z = 0;z < zones;++ z){
          // Copy using abstract indexing
          double err = std::abs((*this)(g,d,z) - b(g,d,z));
          if(err > tol){
            is_diff = true;
            if(verbose){
              printf("%s[g=%d, d=%d, z=%d]: |%e - %e| = %e\n",
                  name.c_str(), g,d,z, (*this)(g,d,z), b(g,d,z), err);
              num_wrong ++;
              if(num_wrong > 100){
                return true;
              }
            }
          }
        }
      }
    }
    return is_diff;
  }

  int ext_to_int[3]; // external index to internal index mapping
  int size_int[3]; // size of each dimension in internal indices

  int groups, directions, zones, elements;
  double *data_pointer;
  std::vector<double> data_linear;
};


#endif
