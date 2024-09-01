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
void handle_errors(conduit::Node &params, conduit::Node &output);
void render_data(conduit::Node &params, conduit::Node &output);
void run_lulesh(conduit::Node &params, conduit::Node &output);
void progress_sim(conduit::Node &params, conduit::Node &output);

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
  ascent::register_callback("2_mpi", mpi_example);
  ascent::register_callback("3_params", conduit_example);
  ascent::register_callback("4_shapes", handle_errors);
  ascent::register_callback("5_render", render_data);
  ascent::register_callback("6_lulesh", run_lulesh);
  ascent::register_callback("7_end", progress_sim);

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
    std::cout << "If you can see this message, you have successfully used Ascent to execute arbitrary code while a simulation (Kripke) is still running." << std::endl;
    std::cout << "To be more specific, the simulation is effectively paused right now, waiting patiently for you to yield control back to it." << std::endl;
    std::cout << std::endl;
    std::cout << "The coolest part is that this simulation *doesn't* expose native steering controls." << std::endl;
    std::cout << "Our approach is simulation-agnostic, meaning that existing codes (even the ones you use) can be enhanced with interactivity." << std::endl;
    std::cout << "Since we're leveraging Ascent's infrastructure, this steering interface also works at scale." << std::endl;
    std::cout << std::endl;
    std::cout << "Sure, we're only printing a message to the terminal right now, but we could have executed any valid C++ code here instead." << std::endl;
    std::cout << "This code lives alongside the simulation, so there's nothing stopping us from exposing the simulation state and changing it." << std::endl;
    std::cout << "We could also execute bash, python, or anything else that lives externally to the simulation." << std::endl;
    output["status"] = "We're learning!";
  }
}

void mpi_example(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();
  MPI_Comm mpi_comm = MPI_Comm_f2c(params["mpi_comm"].as_int32());

  if (mpi_rank == 0)
  {
    // Credit: https://patorjk.com/software/taag/#p=display&f=Doh&t=MPI
    std::cout << "MMMMMMMM               MMMMMMMMPPPPPPPPPPPPPPPPP   IIIIIIIIII" << std::endl;
    std::cout << "M:::::::M             M:::::::MP::::::::::::::::P  I::::::::I" << std::endl;
    std::cout << "M::::::::M           M::::::::MP::::::PPPPPP:::::P I::::::::I" << std::endl;
    std::cout << "M:::::::::M         M:::::::::MPP:::::P     P:::::PII::::::II" << std::endl;
    std::cout << "M::::::::::M       M::::::::::M  P::::P     P:::::P  I::::I  " << std::endl;
    std::cout << "M:::::::::::M     M:::::::::::M  P::::P     P:::::P  I::::I  " << std::endl;
    std::cout << "M:::::::M::::M   M::::M:::::::M  P::::PPPPPP:::::P   I::::I  " << std::endl;
    std::cout << "M::::::M M::::M M::::M M::::::M  P:::::::::::::PP    I::::I  " << std::endl;
    std::cout << "M::::::M  M::::M::::M  M::::::M  P::::PPPPPPPPP      I::::I  " << std::endl;
    std::cout << "M::::::M   M:::::::M   M::::::M  P::::P              I::::I  " << std::endl;
    std::cout << "M::::::M    M:::::M    M::::::M  P::::P              I::::I  " << std::endl;
    std::cout << "M::::::M     MMMMM     M::::::M  P::::P              I::::I  " << std::endl;
    std::cout << "M::::::M               M::::::MPP::::::PP          II::::::II" << std::endl;
    std::cout << "M::::::M               M::::::MP::::::::P          I::::::::I" << std::endl;
    std::cout << "M::::::M               M::::::MP::::::::P          I::::::::I" << std::endl;
    std::cout << "MMMMMMMM               MMMMMMMMPPPPPPPPPP          IIIIIIIIII" << std::endl;
    std::cout << std::endl;
    std::cout << "This steering interface also works seamlessly with MPI." << std::endl;
    std::cout << "For convenience, we automatically pass the MPI communicator and MPI ranks with the params." << std::endl;
    std::cout << "Here's some proof that this callback is actually being executed by all of the MPI ranks:" << std::endl << std::endl;
  }

  MPI_Barrier(mpi_comm);

  char hostname[1024];
  gethostname(hostname, sizeof(hostname));
  std::cout << "I'm rank " << mpi_rank << " and my hostname is " << hostname << std::endl;
  output["hostname"] = hostname;
  output["note"] = "Currently we only print rank 0's output.";
}

void conduit_example(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  bool success = false;

  if (mpi_rank == 0)
  {
    if(!params.has_path("cool_variable"))
    {
      std::cout << "We require Ascent callbacks to take references to two Conduit nodes." << std::endl;
      std::cout << "This enables us to pass arbitrary parameters to callbacks and retrieve arbitrary output from them." << std::endl;
      std::cout << std::endl;
      std::cout << "Once inside an Ascent callback, we can check the params node for any specific variables that we want to make use of." << std::endl;
      std::cout << "If we don't find what we're looking for, we could use a default value or simply add a message to the output node to tell the user what params need to be set, like so:" << std::endl;
      output["message"] = "Try setting a param called 'cool_variable' (its value can be anything) and run this callback again.";
    }
    else
    {
      // Credit: https://www.asciiart.eu/computers/computers
      std::cout << "You successfully created a param called 'cool_variable'." << std::endl;
      std::cout << "As a reward, here's an ASCII representation of you following along with the tutorial:" << std::endl;
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
      output["simulation_steering_is_cool"] = "yes";
      success = true;
    }
  }

  if (success)
  {
    output["success"] = "yes";
  }
  else
  {
    output["success"] = "no";
  }
}

void handle_errors(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  bool success = false;

  if (mpi_rank == 0)
  {
    // Credit: https://www.asciiart.eu/space/planets
    std::cout << "              _-o#&&*''''?d:>b\\_                           " << std::endl;
    std::cout << "          _o/\"`''  '',, dMF9MMMMMHo_                       " << std::endl;
    std::cout << "       .o&#'        `\"MbHMMMMMMMMMMMHo.                   " << std::endl;
    std::cout << "     .o\"\" '         vodM*$&&HMMMMMMMMMM?.                 " << std::endl;
    std::cout << "    ,'              $M&ood,~'`(&##MMMMMMH\\               " << std::endl;
    std::cout << "   /               ,MMMMMMM#b?#bobMMMMHMMML              " << std::endl;
    std::cout << "  &              ?MMMMMMMMMMMMMMMMM7MMM$R*Hk             " << std::endl;
    std::cout << " ?$.            :MMMMMMMMMMMMMMMMMMM/HMMM|`*L            " << std::endl;
    std::cout << "|               |MMMMMMMMMMMMMMMMMMMMbMH'   T,           " << std::endl;
    std::cout << "$H#:            `*MMMMMMMMMMMMMMMMMMMMb#}'  `?           " << std::endl;
    std::cout << "]MMH#             \"\"*\"\"\"\"*#MMMMMMMMMMMMM'    -       " << std::endl;
    std::cout << "MMMMMb_                   |MMMMMMMMMMMP'     :           " << std::endl;
    std::cout << "HMMMMMMMHo                 `MMMMMMMMMT       .           " << std::endl;
    std::cout << "?MMMMMMMMP                  9MMMMMMMM}       -           " << std::endl;
    std::cout << "-?MMMMMMM                  |MMMMMMMMM?,d-    '           " << std::endl;
    std::cout << " :|MMMMMM-                 `MMMMMMMT .M|.   :           " << std::endl;
    std::cout << "  .9MMM[                    &MMMMM*' `'    .            " << std::endl;
    std::cout << "   :9MMk                    `MMM#\"        -           " << std::endl;
    std::cout << "     &M}                     `          .-             " << std::endl;
    std::cout << "      `&.                             .                " << std::endl;
    std::cout << "        `~,   .                     ./                 " << std::endl;
    std::cout << "            . _                  .-                    " << std::endl;
    std::cout << "              '`--._,dd###pp=\"\"'                        " << std::endl;
    std::cout << std::endl;

    if (!params.has_path("shape"))
    {
      std::cout << "This callback will give you practice with setting several params at once." << std::endl;
      std::cout << "Let's pretend that we're inserting a 3D shape at a specific (x, y, z) coordinate within our mesh." << std::endl;
      std::cout << "Start by setting a new param called 'shape' to your favorite shape." << std::endl;
      output["message"] = "Set a new param called 'shape' to your favorite shape.";
    }
    else if (!params["shape"].dtype().is_string())
    {
      std::cout << "You must have set 'shape' to a numeric value. Try setting it to a string value, like 'cube' or 'sphere'." << std::endl;
      output["message"] = "Make sure that 'shape' is set to a string and not a number.";
    }
    else
    {
      std::string shape = params["shape"].as_string();

      bool has_x = false;
      if (!params.has_path("x"))
      {
        output["message"].append() = "You are missing an 'x' param.";
      }
      else if (!params["x"].dtype().is_double())
      {
        output["message"].append() = "You have an 'x' param, but it is not set to a numeric value.";
      }
      else
      {
        has_x = true;
      }

      bool has_y = false;
      if (!params.has_path("y"))
      {
        output["message"].append() = "You are missing an 'y' param.";
      }
      else if (!params["y"].dtype().is_double())
      {
        output["message"].append() = "You have an 'y' param, but it is not set to a numeric value.";
      }
      else
      {
        has_y = true;
      }

      bool has_z = false;
      if (!params.has_path("z"))
      {
        output["message"].append() = "You are missing an 'z' param.";
      }
      else if (!params["z"].dtype().is_double())
      {
        output["message"].append() = "You have an 'z' param, but it is not set to a numeric value.";
      }
      else
      {
        has_z = true;
      }

      if (has_x && has_y && has_z)
      {
        double x = params["x"].as_double();
        double y = params["y"].as_double();
        double z = params["z"].as_double();
        std::cout << "Great job! You have inserted a " << shape << " at coordinate (" << x << ", " << y << ", " << z << ")" << std::endl;
        success = true;
      }
      else
      {
        std::cout << "Great! Your favorite shape is a " << shape << "." << std::endl;
        std::cout << "Now create 3 more params called 'x', 'y', and 'z'. Read the output message to see what still needs to be fixed." << std::endl;
      }
    }
  }

  if (success)
  {
    output["success"] = "yes";
  }
  else
  {
    output["success"] = "no";
  }
}

void render_data(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  if (mpi_rank == 0)
  {
    std::cout << "                                  \\                                  " << std::endl;
    std::cout << "                                  `\\,/                              " << std::endl;
    std::cout << "                                  .-'-.                             " << std::endl;
    std::cout << "                                 '     `                            " << std::endl;
    std::cout << "                                 `.   .'                            " << std::endl;
    std::cout << "                          `._  .-~     ~-.   _,'                   " << std::endl;
    std::cout << "                           ( )'           '.( )                    " << std::endl;
    std::cout << "             `._    _       /               .'                      " << std::endl;
    std::cout << "              ( )--' `-.  .'                 ;                      " << std::endl;
    std::cout << "         .    .'        '.;                  ()                     " << std::endl;
    std::cout << "          `.-.`           '                 .'                      " << std::endl;
    std::cout << "----*-----;                                .'                       " << std::endl;
    std::cout << "          .`-'.           ,                `.                       " << std::endl;
    std::cout << "         '    '.        .';                  ()                     " << std::endl;
    std::cout << "              (_)-   .-'  `.                 ;                      " << std::endl;
    std::cout << "             ,'   `-'       \\               `.                      " << std::endl;
    std::cout << "                           (_).           .'(_)                    " << std::endl;
    std::cout << "                          .'   '-._   _.-'    `.                   " << std::endl;
    std::cout << "                                 .'   `.                            " << std::endl;
    std::cout << "                                 '     ;                            " << std::endl;
    std::cout << "                                  `-,-'                            " << std::endl;
    std::cout << "                                   /`\\                              " << std::endl;
    std::cout << "                                 /`                                 " << std::endl;
    std::cout << std::endl;
    std::cout << "Suppose we want to experiment with visualization parameters to help us find ideal settings." << std::endl;
    std::cout << "With some effort, it's possible to create a callback which helps you do precisely that." << std::endl;
    std::cout << std::endl;
    std::cout << "After this callback runs, check the kripke-steering directory to see a rendered image of the data." << std::endl;
    std::cout << "The output will tell you which variables can be controlled." << std::endl;
  }

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "volume";
  scenes["s1/plots/p1/field"] = "phi";

  output["color_table"] = "Viridis, Jet";
  if(params.has_path("color_table") && params["color_table"].dtype().is_string())
  {
    std::string color_table = params["color_table"].as_string();
    scenes["s1/plots/p1/color_table/name"] = color_table;
  }

  output["image_width"] = "Numeric";
  if(params.has_path("image_width") && params["image_width"].dtype().is_double())
  {
    int image_width = params["image_width"].as_double();
    scenes["s1/renders/r1/image_width"] = image_width;
  }

  output["image_height"] = "Numeric";
  if(params.has_path("image_height") && params["image_height"].dtype().is_double())
  {
    int image_height = params["image_height"].as_double();
    scenes["s1/renders/r1/image_height"] = image_height;
  }

  output["fov"] = "Numeric";
  if(params.has_path("fov") && params["fov"].dtype().is_double())
  {
    double fov = params["fov"].as_double();
    scenes["s1/renders/r1/camera/fov"] = fov;
  }

  output["azimuth"] = "Numeric";
  if(params.has_path("azimuth") && params["azimuth"].dtype().is_double())
  {
    int azimuth = params["azimuth"].as_double();
    scenes["s1/renders/r1/camera/azimuth"] = azimuth;
  }

  output["elevation"] = "Numeric";
  if(params.has_path("elevation") && params["elevation"].dtype().is_double())
  {
    int elevation = params["elevation"].as_double();
    scenes["s1/renders/r1/camera/elevation"] = elevation;
  }

  scenes["s1/renders/r1/image_prefix"] = "kripke";

  conduit::Node actions;
  conduit::Node &add_scenes= actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  Ascent ascent;
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
}

void run_lulesh(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  if (mpi_rank == 0)
  {
    // Credit: https://patorjk.com/software/taag/#p=display&f=Doh&t=Lulesh
    std::cout << "LLLLLLLLLLL                               lllllll                                     hhhhhhh             " << std::endl;
    std::cout << "L:::::::::L                               l:::::l                                     h:::::h             " << std::endl;
    std::cout << "L:::::::::L                               l:::::l                                     h:::::h             " << std::endl;
    std::cout << "LL:::::::LL                               l:::::l                                     h:::::h             " << std::endl;
    std::cout << "  L:::::L               uuuuuu    uuuuuu   l::::l     eeeeeeeeeeee        ssssssssss   h::::h hhhhh       " << std::endl;
    std::cout << "  L:::::L               u::::u    u::::u   l::::l   ee::::::::::::ee    ss::::::::::s  h::::hh:::::hhh    " << std::endl;
    std::cout << "  L:::::L               u::::u    u::::u   l::::l  e::::::eeeee:::::eess:::::::::::::s h::::::::::::::hh  " << std::endl;
    std::cout << "  L:::::L               u::::u    u::::u   l::::l e::::::e     e:::::es::::::ssss:::::sh:::::::hhh::::::h " << std::endl;
    std::cout << "  L:::::L               u::::u    u::::u   l::::l e:::::::eeeee::::::e s:::::s  ssssss h::::::h   h::::::h" << std::endl;
    std::cout << "  L:::::L               u::::u    u::::u   l::::l e:::::::::::::::::e    s::::::s      h:::::h     h:::::h" << std::endl;
    std::cout << "  L:::::L               u::::u    u::::u   l::::l e::::::eeeeeeeeeee        s::::::s   h:::::h     h:::::h" << std::endl;
    std::cout << "  L:::::L         LLLLLLu:::::uuuu:::::u   l::::l e:::::::e           ssssss   s:::::s h:::::h     h:::::h" << std::endl;
    std::cout << "LL:::::::LLLLLLLLL:::::Lu:::::::::::::::uul::::::le::::::::e          s:::::ssss::::::sh:::::h     h:::::h" << std::endl;
    std::cout << "L::::::::::::::::::::::L u:::::::::::::::ul::::::l e::::::::eeeeeeee  s::::::::::::::s h:::::h     h:::::h" << std::endl;
    std::cout << "L::::::::::::::::::::::L  uu::::::::uu:::ul::::::l  ee:::::::::::::e   s:::::::::::ss  h:::::h     h:::::h" << std::endl;
    std::cout << "LLLLLLLLLLLLLLLLLLLLLLLL    uuuuuuuu  uuuullllllll    eeeeeeeeeeeeee    sssssssssss    hhhhhhh     hhhhhhh" << std::endl;
    std::cout << std::endl;
    std::cout << "Ascent ships with another example simulation called Lulesh. Let's try running it from within an callback." << std::endl;
    std::cout << "In other words, we're about to perform simulation-ception, remember that our original simulation is still waiting in the background." << std::endl;
    std::cout << "Note: this probably has no practical application, beyond demonstrating the flexibility of Ascent callbacks." << std::endl;
    std::cout << std::endl;
    std::cout << "Press enter to proceed..." << std::endl;
    std::cin.get();
    system("mkdir -p lulesh && cd lulesh && mpiexec -n 8 ../../lulesh/lulesh_par -i 100 -s 32");
    output["message"] = "Check the kripke-steering directory, you should now see a new lulesh folder full of png files.";
    output["success"] = "yes";
  }
}

void progress_sim(conduit::Node &params, conduit::Node &output)
{
  int mpi_rank = params["mpi_rank"].as_int32();

  if (mpi_rank == 0)
  {
    std::cout << "The simulation is still running in the background." << std::endl;
    std::cout << "Type 'exit' to yield control back to the simulation." << std::endl;
    std::cout << "Thanks for participating!" << std::endl;
  }
  output["tutorial"] = "completed";
}

///
/// End Ascent Callback Definitions
///

