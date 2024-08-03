#include<Kripke.h>
#include<Kripke/Input_Variables.h>
#include<Kripke/Grid.h>
#include"testKernels.h"
#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include<algorithm>
#include<string>
#include<sstream>

#ifdef KRIPKE_USE_OPENMP
#include<omp.h>
#endif

#ifdef KRIPKE_USE_TCMALLOC
#include<gperftools/malloc_extension.h>
#endif

#ifdef KRIPKE_USE_PERFTOOLS
#include<google/profiler.h>
#endif

#ifdef __bgq__
#include </bgsys/drivers/ppcfloor/spi/include/kernel/location.h>
#include </bgsys/drivers/ppcfloor/spi/include/kernel/memory.h>
#endif

typedef std::pair<int, int> IntPair;

std::vector<std::string> papi_names;

void usage(void){
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid == 0){
    printf("Usage:  [srun ...] kripke [options...]\n");
    printf("Where options are:\n");
    printf("  --dir [D:d,D:d,...]    List of dirsets and dirs/set pairs\n");
    printf("                         Default:  --dir 1:1\n");
    printf("                         Example:  --dir 1:4,2:2,4:1\n");
    printf("  --grp [G:g,G:g,...]    List of grpsets and groups/set pairs\n");
    printf("                         Default:  --grp 1:1\n");
    printf("  --layout <lout>        Layout of spatial subdomains and mpi ranks\n");
    printf("                         0: Blocked layout, subdomains on local rank are adjacent\n");
    printf("                         1: Scattered layout, ranks with same subdomain are adjacent\n");
    printf("                         Default: --layout 0\n");
    printf("  --legendre <lorder>    Scattering Legendre Expansion Order (0, 1, ...)\n");
    printf("                         Default:  --legendre 2\n");
    printf("  --nest [n,n,...]       List of data nestings\n");
    printf("                         Default:  --nest DGZ,DZG,GDZ,GZD,ZDG,ZGD\n");
    printf("  --niter <NITER>        Number of solver iterations to run (default: 10)\n");
    printf("  --out <OUTFILE>        Optional output file (default: none)\n");
    printf("  --gperf                Turn on Google Perftools profiling\n");
    printf("  --pmethod <method>     Parallel solver method\n");
    printf("                         sweep: Full up-wind sweep (wavefront algorithm)\n");
    printf("                         bj: Block Jacobi\n");
    printf("                         Default: --pmethod sweep\n");
    printf("  --procs <npx,npy,npz>  MPI task spatial decomposition\n");
    printf("                         Default:  --procs 1,1,1\n");
    printf("  --quad <polar:azim>    Use a Gauss-Legendre Product Quadrature\n");
    printf("                         with the specified number of polar and azimuthal points\n");
    printf("                         Default:  --quad 0,0  [disabled, use dummy S2]\n");
    printf("  --restart <point>      Restart at given point\n");
#ifdef KRIPKE_USE_SILO
    printf("  --silo <BASENAME>      Create SILO output files\n");
#endif
    printf("  --test                 Run Kernel Test instead of solver\n");
    printf("  --zset [x:y:z, ...]    Number of zonesets in x:y:z\n");
    printf("                         Default:  --zst 1:1:1\n");
    printf("  --zones <x,y,z>        Number of zones in x,y,z\n");
    printf("                         Default:  --zones 12,12,12\n");
    printf("\n");
  }
  MPI_Finalize();
  exit(1);
}

struct CmdLine {
  CmdLine(int argc, char **argv) :
    size(argc-1),
    cur(0),
    args()
  {
    for(int i = 0;i < size;++ i){
      args.push_back(argv[i+1]);
    }
  }

  std::string pop(void){
    if(atEnd())
      usage();
    return args[cur++];
  }

  bool atEnd(void){
    return(cur >= size);
  }

  int size;
  int cur;
  std::vector<std::string> args;
};

std::vector<std::string> split(std::string const &str, char delim){
  std::vector<std::string> elem;
  std::stringstream ss(str);
  std::string e;
  while(std::getline(ss, e, delim)){
    elem.push_back(e);
  }
  return elem;
}


namespace {
  template<typename T>
  std::string toString(T const &val){
    std::stringstream ss;
    ss << val;
    return ss.str();
  }
}


void runPoint(int point, int num_tasks, int num_threads, Input_Variables &input_variables, FILE *out_fp, std::string const &run_name){

  /* Allocate problem */
  Grid_Data *grid_data = new Grid_Data(&input_variables);

  grid_data->timing.setPapiEvents(papi_names);

  /* Run the solver */
  SweepSolver(grid_data, input_variables.parallel_method == PMETHOD_BJ);

#ifdef KRIPKE_USE_SILO
  /* output silo data, if requested */
  if(input_variables.silo_basename != ""){
    std::string silo_name = input_variables.silo_basename + toString(point);
    grid_data->writeSilo(silo_name);
  }
#endif

  std::string nesting = nestingString(input_variables.nesting);

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid == 0){
    std::vector<std::string> headers;
    std::vector<std::string> values;

    headers.push_back("run_name");
    values.push_back(run_name);

    headers.push_back("point");
    values.push_back(toString(point));

    headers.push_back("nesting");
    values.push_back(nesting);

    headers.push_back("num_tasks");
    values.push_back(toString(num_tasks));

    headers.push_back("num_threads");
    values.push_back(toString(num_threads));

    headers.push_back("D");
    values.push_back(toString(input_variables.num_dirsets_per_octant));

    headers.push_back("d");
    values.push_back(toString(input_variables.num_dirs_per_dirset));

    headers.push_back("dirs");
    values.push_back(toString(8*input_variables.num_dirsets_per_octant*input_variables.num_dirs_per_dirset));

    headers.push_back("G");
    values.push_back(toString(input_variables.num_groupsets));

    headers.push_back("g");
    values.push_back(toString(input_variables.num_groups_per_groupset));

    headers.push_back("groups");
    values.push_back(toString(input_variables.num_groupsets * input_variables.num_groups_per_groupset));


    if(out_fp != NULL){
      grid_data->timing.printTabular(point == 1, headers, values, out_fp);
      fflush(out_fp);
    }
    grid_data->timing.print();
    printf("\n\n");
  }

  /* Cleanup */
  delete grid_data;
}

int main(int argc, char **argv) {
  /*
   * Initialize MPI
   */
  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int num_tasks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  if (myid == 0) {
    /* Print out a banner message along with a version number. */
    printf("\n");
    printf("---------------------------------------------------------\n");
    printf("------------------- KRIPKE VERSION 1.0 ------------------\n");
    printf("---------------------------------------------------------\n");

    /* Print out some information about how OpenMP threads are being mapped
     * to CPU cores.
     */
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
#ifdef __bgq__
      int core = Kernel_ProcessorCoreID();
#else
      int core = sched_getcpu();
#endif
      printf("Rank: %d Thread %d: Core %d\n", myid, tid, core);
    }
#endif
  }


  /*
   * Default input parameters
   */
  std::string run_name = "kripke";
  std::vector<IntPair> grp_list;
  grp_list.push_back(IntPair(1,1));
  std::vector<IntPair> dir_list;
  dir_list.push_back(IntPair(1,1));
  std::string outfile;
  int nprocs[3] = {1, 1, 1};
  int zset[3] = {1,1,1};
  int layout = 0;
  int nzones[3] = {12, 12, 12};
  int lorder = 4;
  int num_polar = 0;
  int num_azimuthal = 0;
  int niter = 10;
  double sigt[3] = {0.10, 0.0001, 0.10};
  double sigs[3] = {0.05, 0.00005, 0.05};
  bool test = false;
  bool perf_tools = false;
  int restart_point = 0;
  ParallelMethod parallel_method = PMETHOD_SWEEP;
#ifdef KRIPKE_USE_SILO
  std::string silo_basename = "";
#endif


  std::vector<Nesting_Order> nest_list;
  nest_list.push_back(NEST_DGZ);
  nest_list.push_back(NEST_DZG);
  nest_list.push_back(NEST_GDZ);
  nest_list.push_back(NEST_GZD);
  nest_list.push_back(NEST_ZDG);
  nest_list.push_back(NEST_ZGD);

  /*
   * Parse command line
   */
  CmdLine cmd(argc, argv);
  while(!cmd.atEnd()){
    std::string opt = cmd.pop();
    if(opt == "-h" || opt == "--help"){usage();}
    else if(opt == "--out"){outfile = cmd.pop();}
    else if(opt == "--name"){run_name = cmd.pop();}
    else if(opt == "--zset"){
      std::vector<std::string> nz = split(cmd.pop(), ':');
      if(nz.size() != 3) usage();
      zset[0] = std::atoi(nz[0].c_str());
      zset[1] = std::atoi(nz[1].c_str());
      zset[2] = std::atoi(nz[2].c_str());
      if(zset[0] <= 0 || zset[1] <= 0 || zset[2] <= 0){usage();}
    }
    else if(opt == "--layout"){
      layout = std::atoi(cmd.pop().c_str());
      if(layout < 0 || layout > 1){usage();}
    }
    else if(opt == "--zones"){
      std::vector<std::string> nz = split(cmd.pop(), ',');
      if(nz.size() != 3) usage();
      nzones[0] = std::atoi(nz[0].c_str());
      nzones[1] = std::atoi(nz[1].c_str());
      nzones[2] = std::atoi(nz[2].c_str());
    }
    else if(opt == "--procs"){
      std::vector<std::string> np = split(cmd.pop(), ',');
      if(np.size() != 3) usage();
      nprocs[0] = std::atoi(np[0].c_str());
      nprocs[1] = std::atoi(np[1].c_str());
      nprocs[2] = std::atoi(np[2].c_str());
    }
    else if(opt == "--pmethod"){
      std::string method = cmd.pop();
      if(!strcasecmp(method.c_str(), "sweep")){
        parallel_method = PMETHOD_SWEEP;
      }
      else if(!strcasecmp(method.c_str(), "bj")){
        parallel_method = PMETHOD_BJ;
      }
      else{
        usage();
      }
    }
    else if(opt == "--grp"){
      std::vector<std::string> sets = split(cmd.pop(), ',');
      if(sets.size() < 1)usage();
      grp_list.clear();
      for(int i = 0;i < sets.size();++ i){
        std::vector<std::string> p = split(sets[i], ':');
        if(p.size() != 2)usage();
        grp_list.push_back(IntPair(std::atoi(p[0].c_str()), std::atoi(p[1].c_str())));
      }
    }
    else if(opt == "--dir"){
      std::vector<std::string> sets = split(cmd.pop(), ',');
      if(sets.size() < 1)usage();
      dir_list.clear();
      for(int i = 0;i < sets.size();++ i){
        std::vector<std::string> p = split(sets[i], ':');
        if(p.size() != 2)usage();
        dir_list.push_back(IntPair(std::atoi(p[0].c_str()), std::atoi(p[1].c_str())));
      }
    }
    else if(opt == "--legendre"){
      lorder = std::atoi(cmd.pop().c_str());
    }
    else if(opt == "--quad"){
      std::vector<std::string> values = split(cmd.pop(), ':');
      if(values.size()!=2)usage();
      num_polar = std::atoi(values[0].c_str());
      num_azimuthal = std::atoi(values[1].c_str());
    }
    else if(opt == "--sigs"){
      std::vector<std::string> values = split(cmd.pop(), ',');
      if(values.size()!=3)usage();
      for(int mat = 0;mat < 3;++ mat){
        sigs[mat] = std::atof(values[mat].c_str());
      }
    }
    else if(opt == "--sigt"){
      std::vector<std::string> values = split(cmd.pop(), ',');
      if(values.size()!=3)usage();
      for(int mat = 0;mat < 3;++ mat){
        sigt[mat] = std::atof(values[mat].c_str());
      }
    }
    else if(opt == "--niter"){
      niter = std::atoi(cmd.pop().c_str());
    }
    else if(opt == "--nest"){
      std::vector<std::string> sets = split(cmd.pop(), ',');
      if(sets.size() < 1)usage();
      nest_list.clear();
      for(int i = 0;i < sets.size();++ i){
        Nesting_Order n = nestingFromString(sets[i]);
        if(n < 0)usage();
        nest_list.push_back(n);
      }
    }
#ifdef KRIPKE_USE_SILO
    else if(opt == "--silo"){
      silo_basename = cmd.pop();
    }
#endif
    else if(opt == "--test"){
      test = true;
    }
    else if(opt == "--papi"){
      papi_names = split(cmd.pop(), ',');
    }
    else if(opt == "--gperf"){
      perf_tools = true;
    }
    else if(opt == "--restart"){
      restart_point = std::atoi(cmd.pop().c_str());
    }
    else{
      printf("Unknwon options %s\n", opt.c_str());
      usage();
    }
  }

  /*
   * Display Options
   */
  int nsearches = grp_list.size() * dir_list.size() * nest_list.size();
  int num_threads=1;
  if (myid == 0) {
    printf("Number of MPI tasks:   %d\n", num_tasks);
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
      if(omp_get_thread_num() == 0){
          printf("OpenMP threads/task:   %d\n", num_threads);
          printf("OpenMP total threads:  %d\n", num_threads*num_tasks);
        }
    }
#endif
    printf("Output File:           %s\n", outfile.c_str());
    printf("Processors:            %d x %d x %d\n", nprocs[0], nprocs[1], nprocs[2]);
    printf("Zones:                 %d x %d x %d\n", nzones[0], nzones[1], nzones[2]);
    printf("Legendre Order:        %d\n", lorder);
    printf("Total X-Sec:           sigt=[%lf, %lf, %lf]\n", sigt[0], sigt[1], sigt[2]);
    printf("Scattering X-Sec:      sigs=[%lf, %lf, %lf]\n", sigs[0], sigs[1], sigs[2]);
    printf("Quadrature Set:        ");
    if(num_polar == 0){
      printf("Dummy S2\n");
    }
    else {
      printf("Gauss-Legendre, %d polar, %d azimuthal\n", num_polar, num_azimuthal);
    }
    printf("Parallel method:       ");
    if(parallel_method == PMETHOD_SWEEP){
      printf("Sweep\n");
    }
    else if(parallel_method == PMETHOD_BJ){
      printf("Block Jacobi\n");
    }
    printf("Number iterations:     %d\n", niter);

    if(grp_list.size() == 0){
      printf("No GroupSet/Groups defined (--grp)\n");
      usage();
    }
    printf("GroupSet/Groups:       ");
    for(int i = 0;i < grp_list.size();++ i){
      printf("%s%d:%d", (i==0 ? "" : ", "), grp_list[i].first, grp_list[i].second);
    }
    printf("\n");

    if(dir_list.size() == 0){
      printf("No DirSets/Directions defined (--dir)\n");
      usage();
    }
    printf("DirSets/Directions:    ");
    for(int i = 0;i < dir_list.size();++ i){
      printf("%s%d:%d", (i==0 ? "" : ", "), dir_list[i].first, dir_list[i].second);
    }
    printf("\n");

    printf("Zone Sets:             ");
    printf("%d:%d:%d\n", zset[0], zset[1], zset[2]);

    printf("Nestings:              ");
    for(int i = 0;i < nest_list.size();++ i){
      printf("%s%s", (i==0 ? "" : ", "), nestingString(nest_list[i]).c_str());
    }
    printf("\n");
    printf("Search space size:     %d points\n", nsearches);
    if(perf_tools){
      printf("Using Google Perftools\n");
    }
  }

  /*
   * Execute the Search Space
   */
  FILE *outfp = NULL;
  if(outfile != "" && myid == 0){
    if(restart_point == 0){
      outfp = fopen(outfile.c_str(), "wb");
    }
    else{
      outfp = fopen(outfile.c_str(), "ab");
    }
  }
#ifdef KRIPKE_USE_PERFTOOLS
  if(perf_tools){
    std::stringstream pfname;
    pfname << "profile." << myid;
    ProfilerStart(pfname.str().c_str());
    ProfilerRegisterThread();
  }
#endif
  Input_Variables ivars;
  ivars.nx = nzones[0];
  ivars.ny = nzones[1];
  ivars.nz = nzones[2];
  ivars.npx = nprocs[0];
  ivars.npy = nprocs[1];
  ivars.npz = nprocs[2];
  ivars.legendre_order = lorder;
  ivars.niter = niter;
  ivars.num_zonesets_dim[0] = zset[0];
  ivars.num_zonesets_dim[1] = zset[1];
  ivars.num_zonesets_dim[2] = zset[2];
  ivars.parallel_method = parallel_method;

  for(int mat = 0;mat < 3;++ mat){
    ivars.sigt[mat] = sigt[mat];
    ivars.sigs[mat] = sigs[mat];
  }
  ivars.layout_pattern = layout;
  ivars.quad_num_polar = num_polar;
  ivars.quad_num_azimuthal = num_azimuthal;
#ifdef KRIPKE_USE_SILO
  ivars.silo_basename = silo_basename;
#endif
  int point = 0;
  for(int d = 0;d < dir_list.size();++ d){
    for(int g = 0;g < grp_list.size();++ g){
      for(int n = 0;n < nest_list.size();++ n){

        if(restart_point <= point+1){
          if(myid == 0){
            printf("Running point %d/%d: D:d=%d:%d, G:g=%d:%d, Nest=%s\n",
                point+1, nsearches,
                dir_list[d].first,
                dir_list[d].second,
                grp_list[g].first,
                grp_list[g].second,
                nestingString(nest_list[n]).c_str());
          }
          // Setup Current Search Point
          ivars.num_dirsets_per_octant = dir_list[d].first;
          ivars.num_dirs_per_dirset = dir_list[d].second;
          ivars.num_groupsets = grp_list[g].first;
          ivars.num_groups_per_groupset = grp_list[g].second;
          ivars.nesting = nest_list[n];

          // Run the point
          if(test){
            // Invoke Kernel testing
            testKernels(ivars);
          }
          else{
            // Just run the "solver"
            runPoint(point+1, num_tasks, num_threads, ivars, outfp, run_name);
          }


          // Gather post-point memory info
          double heap_mb = -1.0;
          double hwm_mb = -1.0;
#ifdef KRIPKE_USE_TCMALLOC
          // If we are using tcmalloc, we need to use it's interface
          MallocExtension *mext = MallocExtension::instance();
          size_t bytes;

          mext->GetNumericProperty("generic.current_allocated_bytes", &bytes);
          heap_mb = ((double)bytes)/1024.0/1024.0;

          mext->GetNumericProperty("generic.heap_size", &bytes);
          hwm_mb = ((double)bytes)/1024.0/1024.0;
#else
#ifdef __bgq__
          // use BG/Q specific calls (if NOT using tcmalloc)
          uint64_t bytes;

          int rc = Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &bytes);
          heap_mb = ((double)bytes)/1024.0/1024.0;

          rc = Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPMAX, &bytes);
          hwm_mb = ((double)bytes)/1024.0/1024.0;
#endif
#endif
          // Print memory info
          if(myid == 0 && heap_mb >= 0.0){
            printf("Bytes allocated: %lf MB\n", heap_mb);
            printf("Heap Size      : %lf MB\n", hwm_mb);

          }
        }
        point ++;

      }
    }
  }
  if(outfp != NULL){
    fclose(outfp);
  }

  /*
   * Cleanup and exit
   */
  MPI_Finalize();
#ifdef KRIPKE_USE_PERFTOOLS
  if(perf_tools){
    ProfilerFlush();
    ProfilerStop();
  }
#endif
  return (0);
}
