#ifndef KRIPKE_LAYOUT_H__
#define KRIPKE_LAYOUT_H__

#include<algorithm>

// foreward decl
struct Input_Variables;

/**
  Describes a neighboring Subdomain using both mpi-rank and subdomin id
*/
struct Neighbor{
  int mpi_rank;     // Neighbors MPI rank, or -1 for boundary condition
  int subdomain_id; // Subdomain ID of neighbor
};



/**
   Describes relationships between MPI-ranks and subdomains.
   This is an interface, allowing different layout schemes to be implemented as derived types.
 */
class Layout {
  public:
    explicit Layout(Input_Variables *input_vars);
    virtual ~Layout();

    virtual int setIdToSubdomainId(int gs, int ds, int zs) const;
    virtual int subdomainIdToZoneSetDim(int sdom_id, int dim) const;
    virtual void subdomainIdToSetId(int sdom_id, int &gs, int &ds, int &zs) const;
    virtual Neighbor getNeighbor(int our_sdom_id, int dim, int dir) const = 0;
    virtual std::pair<double, double> getSpatialExtents(int sdom_id, int dim) const = 0;
    virtual int getNumZones(int sdom_id, int dim) const;

  protected:
    int num_group_sets;      // Number of group sets
    int num_direction_sets;  // Number of direction sets
    int num_zone_sets;       // Number of zone sets
    int num_zone_sets_dim[3];// Number of zone sets in each dimension

    int total_zones[3];      // Total number of zones in each dimension

    int num_procs[3];        // Number of MPI ranks in each dimensions
    int our_rank[3];         // Our mpi indices in xyz
};

class BlockLayout : public Layout {
  public:
    explicit BlockLayout(Input_Variables *input_vars);
    virtual ~BlockLayout();

    virtual Neighbor getNeighbor(int our_sdom_id, int dim, int dir) const;
    virtual std::pair<double, double> getSpatialExtents(int sdom_id, int dim) const;
};

class ScatterLayout : public Layout {
  public:
    explicit ScatterLayout(Input_Variables *input_vars);
    virtual ~ScatterLayout();

    virtual Neighbor getNeighbor(int our_sdom_id, int dim, int dir) const;
    virtual std::pair<double, double> getSpatialExtents(int sdom_id, int dim) const;
};


// Factory to create layout object
Layout *createLayout(Input_Variables *input_vars);

#endif
