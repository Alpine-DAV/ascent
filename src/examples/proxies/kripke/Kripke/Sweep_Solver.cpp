/*--------------------------------------------------------------------------
 * Sweep-based solver routine.
 *--------------------------------------------------------------------------*/

#include <Kripke.h>
#include <Kripke/Subdomain.h>
#include <Kripke/SubTVec.h>
#include <Kripke/ParallelComm.h>
#include <Kripke/Grid.h>
#include <vector>
#include <stdio.h>

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------
 * Begin Alpine Integration
 *--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#include <conduit_blueprint.hpp>
#include <alpine.hpp>

using namespace conduit;
using alpine::Alpine;
static int count = 0;
static int max_backlog = 0;

void writeAlpineData(Alpine &alpine, Grid_Data *grid_data, int timeStep)
{
  
  grid_data->kernel->LTimes(grid_data);
  conduit::Node data;
  
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int num_zone_sets = grid_data->zs_to_sdomid.size();

  // TODO: we don't support domain overloading ... 
  for(int sdom_idx = 0; sdom_idx < grid_data->num_zone_sets; ++sdom_idx)
  {
    ALPINE_BLOCK_TIMER(COPY_DATA);
    
    int sdom_id =  grid_data->zs_to_sdomid[sdom_idx];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    //create coords array
    conduit::float64 *coords[3];

    data["state/time"]   = (conduit::float64)3.1415;
    data["state/domain"] = (conduit::uint64) myid;
    data["state/cycle"]  = (conduit::uint64) timeStep;

    data["state/performance/incomingRequests"] = ParallelComm::getIncomingRequests();
    data["state/performance/outgointRequests"] = ParallelComm::getOutgoingRequests();
    data["state/performance/loops"] = count;
    data["state/performance/max_backlog"] = max_backlog;
    ParallelComm::resetRequests();

    data["coordsets/coords/type"]  = "rectilinear";
    data["coordsets/coords/values/x"].set(conduit::DataType::float64(sdom.nzones[0]+1));
    coords[0] = data["coordsets/coords/values/x"].value();
    data["coordsets/coords/values/y"].set(conduit::DataType::float64(sdom.nzones[1]+1));
    coords[1] = data["coordsets/coords/values/y"].value();
    data["coordsets/coords/values/z"].set(conduit::DataType::float64(sdom.nzones[2]+1));
    coords[2] = data["coordsets/coords/values/z"].value();

    data["topologies/mesh/type"]      = "rectilinear";
    data["topologies/mesh/coordset"]  = "coords";

    for(int dim = 0; dim < 3;++ dim)
    {
      coords[dim][0] = sdom.zeros[dim];
      for(int z = 0;z < sdom.nzones[dim]; ++z)
      {
        coords[dim][1+z] = coords[dim][z] + sdom.deltas[dim][z];
      }
    }
    data["fields/phi/association"] = "element";
    data["fields/phi/topology"] = "mesh";
    data["fields/phi/type"] = "scalar";
  
    data["fields/phi/values"].set(conduit::DataType::float64(sdom.num_zones));
    conduit::float64 * phi_scalars = data["fields/phi/values"].value();

    // TODO can we do this with strides and not copy?
    for(int i = 0; i < sdom.num_zones; i++)
    {
      phi_scalars[i] = (*sdom.phi)(0,0,i);  
    } 

  }//each sdom
  
   //------- end wrapping with Conduit here -------//
  conduit::Node verify_info;
  if(!conduit::blueprint::mesh::verify(data,verify_info))
  {
      CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
  }
  else
  {
      CONDUIT_INFO("blueprint verify succeeded");
  }

  conduit::Node actions;   
  conduit::Node scenes;
  scenes["s1/plots/p1/type"]         = "volume";
  scenes["s1/plots/p1/params/field"] = "phi";


  conduit::Node &add_plots = actions.append();
  add_plots["action"] = "add_scenes";
  add_plots["scenes"] = scenes;

  actions.append()["action"] = "execute";  
  actions.append()["action"] = "reset";
  
  alpine.publish(data);
  alpine.execute(actions);
}

/**
  Run solver iterations.
*/
int SweepSolver (Grid_Data *grid_data, bool block_jacobi)
{
  conduit::Node alpine_opts;
  alpine_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  alpine_opts["runtime/type"] = "ascent";

  Alpine alpine;
  alpine.open(alpine_opts);

  conduit::Node testNode;
  Kernel *kernel = grid_data->kernel;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  BLOCK_TIMER(grid_data->timing, Solve);
  {
  
  // Loop over iterations
  double part_last = 0.0;
 for(int iter = 0;iter < grid_data->niter;++ iter){
   
   {//alpine block timer
     ALPINE_BLOCK_TIMER(KRIPKE_MAIN_LOOP);
    /*
     * Compute the RHS:  rhs = LPlus*S*L*psi + Q
     */

    // Discrete to Moments transformation (phi = L*psi)
    {
      BLOCK_TIMER(grid_data->timing, LTimes);
      kernel->LTimes(grid_data);
    }

    // Compute Scattering Source Term (psi_out = S*phi)
    {
      BLOCK_TIMER(grid_data->timing, Scattering);
      kernel->scattering(grid_data);
    }

    // Compute External Source Term (psi_out = psi_out + Q)
    {
      BLOCK_TIMER(grid_data->timing, Source);
      kernel->source(grid_data);
    }

    // Moments to Discrete transformation (rhs = LPlus*psi_out)
    {
      BLOCK_TIMER(grid_data->timing, LPlusTimes);
      kernel->LPlusTimes(grid_data);
    }

    /*
     * Sweep (psi = Hinv*rhs)
     */
    {
      BLOCK_TIMER(grid_data->timing, Sweep);

      if(true){
        // Create a list of all groups
        std::vector<int> sdom_list(grid_data->subdomains.size());
        for(int i = 0;i < grid_data->subdomains.size();++ i){
          sdom_list[i] = i;
        }

        // Sweep everything
        SweepSubdomains(sdom_list, grid_data, block_jacobi);
      }
      // This is the ARDRA version, doing each groupset sweep independently
      else{
        for(int group_set = 0;group_set < grid_data->num_group_sets;++ group_set){
          std::vector<int> sdom_list;
          // Add all subdomains for this groupset
          for(int s = 0;s < grid_data->subdomains.size();++ s){
            if(grid_data->subdomains[s].idx_group_set == group_set){
              sdom_list.push_back(s);
            }
          }

          // Sweep the groupset
          SweepSubdomains(sdom_list, grid_data, block_jacobi);
        }
      }
    }
   }//end main loop timing
    double part = grid_data->particleEdit();
    writeAlpineData(alpine, grid_data, iter);
    if(mpi_rank==0){
      printf("iter %d: particle count=%e, change=%e\n", iter, part, (part-part_last)/part);
    }
    part_last = part;
  }
  
  alpine.close();
  }//Solve block
  
  //Alpine: we don't want to execute all loop orderings, so we will just exit;
  MPI_Finalize();
  exit(0);
  return(0);
}   
    
/*  --------------------------------------------------------------------------
 *  --------------------------------------------------------------------------
 *   End Alpine Integration
 *  --------------------------------------------------------------------------
 *  --------------------------------------------------------------------------*/
    
    
/**
  Perform full parallel sweep algorithm on subset of subdomains.
*/  
void SweepSubdomains (std::vector<int> subdomain_list, Grid_Data *grid_data, bool block_jacobi)
{   
  // Create a new sweep communicator object
  ParallelComm *comm = NULL;
  if(block_jacobi){
    comm = new BlockJacobiComm(grid_data);
  }
  else {
    comm = new SweepComm(grid_data);
  }

  // Add all subdomains in our list
  for(int i = 0;i < subdomain_list.size();++ i){
    int sdom_id = subdomain_list[i];
    comm->addSubdomain(sdom_id, grid_data->subdomains[sdom_id]);
  }
  count = 0;
  max_backlog = 0;
  /* Loop until we have finished all of our work */
  while(comm->workRemaining()){
    count++;
    // Get a list of subdomains that have met dependencies
    std::vector<int> sdom_ready = comm->readySubdomains();
    int backlog = sdom_ready.size();
    max_backlog = max_backlog < backlog ? backlog : max_backlog;
    // Run top of list
    if(backlog > 0){
      int sdom_id = sdom_ready[0];
      Subdomain &sdom = grid_data->subdomains[sdom_id];
      // Clear boundary conditions
      for(int dim = 0;dim < 3;++ dim){
        if(sdom.upwind[dim].subdomain_id == -1){
          sdom.plane_data[dim]->clear(0.0);
        }
      }
      {
        BLOCK_TIMER(grid_data->timing, Sweep_Kernel);
        // Perform subdomain sweep
        grid_data->kernel->sweep(&sdom);
      }

      // Mark as complete (and do any communication)
      comm->markComplete(sdom_id);
    }
  }
  delete comm;
}


