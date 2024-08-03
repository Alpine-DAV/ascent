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
 * Begin Ascent Integration
 *--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <unistd.h>

using namespace conduit;
using ascent::Ascent;
static int count = 0;
static int max_backlog = 0;

// Global Conduit nodes
conduit::Node ascent_opts;
conduit::Node data;

///
/// Begin Ascent Callback Prototypes
///

void intro_message(conduit::Node &params, conduit::Node &output);
void mpi_example(conduit::Node &params, conduit::Node &output);
void conduit_example(conduit::Node &params, conduit::Node &output);
void render_data(conduit::Node &params, conduit::Node &output);
void run_laghos(conduit::Node &params, conduit::Node &output);

///
/// End Ascent Callback Prototypes
///

void writeAscentData(Ascent &ascent, Grid_Data *grid_data, int timeStep)
{

  grid_data->kernel->LTimes(grid_data);

  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int num_zone_sets = grid_data->zs_to_sdomid.size();

  // TODO: we don't support domain overloading ...
  for(int sdom_idx = 0; sdom_idx < grid_data->num_zone_sets; ++sdom_idx)
  {
    ASCENT_BLOCK_TIMER(COPY_DATA);

    int sdom_id =  grid_data->zs_to_sdomid[sdom_idx];
    Subdomain &sdom = grid_data->subdomains[sdom_id];
    //create coords array
    conduit::float64 *coords[3];

    data["state/time"]   = (conduit::float64)3.1415;
    data["state/domain_id"] = (conduit::uint64) myid;
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
      CONDUIT_INFO("blueprint verify failed!" + verify_info.to_yaml());
  }
  else
  {
      CONDUIT_INFO("blueprint verify succeeded");
  }

  // We pull our actions from ascent_actions.yaml
  conduit::Node actions;
  conduit::Node extracts;
  extracts["e1/type"] = "steering";

  conduit::Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  ascent.publish(data);
  ascent.execute(actions);
}

/**
  Run solver iterations.
*/
int SweepSolver (Grid_Data *grid_data, bool block_jacobi)
{
  ascent::register_callback("1_intro", intro_message);
  ascent::register_callback("2_params", conduit_example);
  ascent::register_callback("3_mpi", mpi_example);
  ascent::register_callback("4_render", render_data);
  // ascent::register_callback("laghos", run_laghos);

  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_opts["runtime/type"] = "ascent";

  Ascent ascent;
  ascent.open(ascent_opts);

  conduit::Node testNode;
  Kernel *kernel = grid_data->kernel;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  BLOCK_TIMER(grid_data->timing, Solve);
  {

  // Loop over iterations
  double part_last = 0.0;
 for(int iter = 0;iter < grid_data->niter;++ iter){

   {//ascent block timer
     ASCENT_BLOCK_TIMER(KRIPKE_MAIN_LOOP);
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
    writeAscentData(ascent, grid_data, iter);
    if(mpi_rank==0){
      printf("iter %d: particle count=%e, change=%e\n", iter, part, (part-part_last)/part);
    }
    part_last = part;
  }

  ascent.close();
  }//Solve block

  //Ascent: we don't want to execute all loop orderings, so we will just exit;
  MPI_Finalize();
  exit(0);
  return(0);
}

/*  --------------------------------------------------------------------------
 *  --------------------------------------------------------------------------
 *   End Ascent Integration
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

///
/// Begin Ascent Callback Definitions
///

void intro_message(conduit::Node &params, conduit::Node &output)
{
  if (params["mpi_rank"].as_int32() == 0)
  {
    // Credit: https://www.asciiart.eu/toys/balloons
    std::cout << std::endl;
    std::cout << "             .#############. " << std::endl;
    std::cout << "          .###################. " << std::endl;
    std::cout << "       .####%####################.,::;;;;;;;;;;, " << std::endl;
    std::cout << "      .####%###############%######:::;;;;;;;;;;;;;, " << std::endl;
    std::cout << "      ####%%################%######:::;;;;;;;;@;;;;;;, " << std::endl;
    std::cout << "      ####%%################%%#####:::;;;;;;;;;@;;;;;;, " << std::endl;
    std::cout << "      ####%%################%%#####:::;;;;;;;;;@@;;;;;; " << std::endl;
    std::cout << "      `####%################%#####:::;;;;;;;;;;@@;;;;;; " << std::endl;
    std::cout << "        `###%##############%####:::;;;;;;;;;;;;@@;;;;;; " << std::endl;
    std::cout << "           `#################'::%%%%%%%%%%%%;;;@;;;;;;' " << std::endl;
    std::cout << "             `#############'.%%%%%%%%%%%%%%%%%%;;;;;' " << std::endl;
    std::cout << "               `#########'%%%%#%%%%%%%%%%%%%%%%%%%, " << std::endl;
    std::cout << "                 `#####'.%%%%#%%%%%%%%%%%%%%#%%%%%%, " << std::endl;
    std::cout << "                   `##' %%%%##%%%%%%%%%%%%%%%##%%%%% " << std::endl;
    std::cout << "                   ###  %%%%##%%%%%%%%%%%%%%%##%%%%% " << std::endl;
    std::cout << "                    '   %%%%##%%%%%%%%%%%%%%%##%%%%% " << std::endl;
    std::cout << "                   '    `%%%%#%%%%%%%%%%%%%%%#%%%%%' " << std::endl;
    std::cout << "                  '       `%%%#%%%%%%%%%%%%%#%%%%' " << std::endl;
    std::cout << "                  `         `%%%%%%%%%%%%%%%%%%' " << std::endl;
    std::cout << "                   `          `%%%%%%%%%%%%%%' " << std::endl;
    std::cout << "                    `           `%%%%%%%%%%'  ' " << std::endl;
    std::cout << "                     '            `%%%%%%'   ' " << std::endl;
    std::cout << "                    '              `%%%'    ' " << std::endl;
    std::cout << "                   '               .%%      ` " << std::endl;
    std::cout << "                  `                %%%       ' " << std::endl;
    std::cout << "                   `                '       ' " << std::endl;
    std::cout << "                    `              '      ' " << std::endl;
    std::cout << "                    '            '      ' " << std::endl;
    std::cout << "                   '           '       ` " << std::endl;
    std::cout << "                  '           '        ' " << std::endl;
    std::cout << "                              `       ' " << std::endl;
    std::cout << "                               ' " << std::endl;
    std::cout << "         Congraulations!      ' " << std::endl;
    std::cout << "                             ' " << std::endl;
    std::cout << std::endl;
    std::cout << "If you can see this message, you have successfully used Ascent to execute arbitrary code while a simulation is still running." << std::endl;
    std::cout << "To be more specific, the simulation is effectively paused right now, waiting patiently for us (the user) to yield control back to it." << std::endl;
    std::cout << std::endl;
    std::cout << "The coolest part is that this simulation *doesn't* expose native steering controls." << std::endl;
    std::cout << "Our approach is simulation-agnostic, meaning that existing codes (even the ones you use) can be enhanced with interactivity." << std::endl;
    std::cout << "Since we're leveraging Ascent's infrastructure, this steering interface also works at scale." << std::endl;
    std::cout << std::endl;
    std::cout << "Sure, we're only printing a message to the terminal right now, but we could have executed any valid C++ code here instead." << std::endl;
    std::cout << "This code lives alongside the simulation, so there's nothing stopping us from exposing the simulation state and changing it." << std::endl;
    std::cout << "We could also execute bash, python, or anything else that lives externally to the simulation." << std::endl;
  }
}

void conduit_example(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  if (mpi_rank == 0)
  {
    if(!params.has_path("cool_variable"))
    {
      std::cout << "We require Ascent callbacks to take references to two Conduit nodes." << std::endl;
      std::cout << "This enables us to pass arbitrary parameters to callbacks and retrieve arbitrary output from them." << std::endl;
      std::cout << "Here's an example: 'void my_fancy_callback(conduit::Node &params, conduit::Node &output)'." << std::endl;
      std::cout << std::endl;
      std::cout << "Once inside an Ascent callback, we can check the params for specific variables that we expect." << std::endl;
      std::cout << "This is how we can manage control flow." << std::endl;
      std::cout << "If we don't find what we're looking for, we could use a default value, or simply add a message to the output node to tell the user what params need to be set like so:" << std::endl;

      output["success"] = "no";
      output["error_message"] = "Try setting a param called 'cool_variable' (value doesn't matter) and run this callback again.";
    }
    else
    {
      std::cout << "Yay! You correctly set 'cool_variable' to something." << std::endl;
      std::cout << "Here's your reward, an ASCII representation of you following along with the tutorial:" << std::endl;
      std::cout << " ______________" << std::endl;
      std::cout << "||            ||" << std::endl;
      std::cout << "||  Ascent    ||" << std::endl;
      std::cout << "||  Tutorial  ||" << std::endl;
      std::cout << "||            ||" << std::endl;
      std::cout << "||____________||" << std::endl;
      std::cout << "|______________|" << std::endl;
      std::cout << " \\\\############\\\\" << std::endl;
      std::cout << "  \\\\############\\\\" << std::endl;
      std::cout << "   \\      ____    \\ " << std::endl;
      std::cout << "    \\_____\\___\\____\\ " << std::endl;

      output["success"] = "yes";
      output["simulation_steering_is_cool"] = "also yes";
    }
  }
}

void mpi_example(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  if (mpi_rank == 0)
  {
    std::cout << "This steering interface also plays nicely with MPI." << std::endl;
    std::cout << "For convenience, we automatically pass the MPI communicator and MPI ranks with the params." << std::endl;
    std::cout << "That's how I'm able to print these lines only once, by checking the params to make sure that I'm rank 0." << std::endl;
    std::cout << "Here's proof that this callback is actually getting executed by all of the MPI ranks:" << std::endl << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  char hostname[1024];
  gethostname(hostname, sizeof(hostname));
  std::cout << "I'm rank " << mpi_rank << " and my hostname is " << hostname << std::endl;
  output["hostname"] = hostname;
}

void render_data(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  if (mpi_rank == 0)
  {
    std::cout << "Suppose we want to interactively adjust our visualizations to find the best settings." << std::endl;
    std::cout << "With some effort, it's possible to create a callback which helps you do that." << std::endl;
    std::cout << std::endl;
    std::cout << "After this callback runs, check the working directory to see a rendered image of the data." << std::endl;
    std::cout << "Try setting a new param called 'color_table' to 'Jet' or 'Viridis' and run this callback again." << std::endl;
    std::cout << "You should see a similar rendered image, but using a different color table." << std::endl;
  }

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "volume";
  scenes["s1/image_prefix"] = "volume_phi";
  output["color_table_used"] = "default";
  if(params.has_path("color_table"))
  {
    std::string color_table = params["color_table"].as_string();
    scenes["s1/plots/p1/color_table/name"] = color_table;
    output["color_table_used"] = color_table;
  }
  scenes["s1/plots/p1/field"] = "phi";

  conduit::Node actions;
  // add the scenes
  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  Ascent ascent;
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
}

void run_laghos(conduit::Node &params, conduit::Node &output)
{
  // A stretch goal, but it would be cool to run a totally different simulation from a callback just to prove that it can be done
}

///
/// End Ascent Callback Definitions
///

