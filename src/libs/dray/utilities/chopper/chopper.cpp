#include <dray/dray.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/utils/png_encoder.hpp>

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

int main (int argc, char *argv[])
{
#ifdef MPI_ENABLED
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));
#endif

  std::string config_file = "";

  if (argc != 2)
  {
    std::cout << "Missing configure file name\n";
    exit (1);
  }

  config_file = argv[1];

  conduit::Node options;
  options.load(config_file,"yaml");

  if(dray::dray::mpi_rank() == 0)
  {
    options.print();
  }

  if(!options.has_path("root_file"))
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"Missing root file\n";
    }
    exit(1);
  }
  if(!options.has_path("axis"))
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"Missing axis\n";
    }
    exit(1);
  }

  if(!options.has_path("direction"))
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"Missing direction\n";
    }
    exit(1);
  }

  if(!options.has_path("location"))
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"Missing locaton\n";
    }
    exit(1);
  }

  int axis = options["axis"].to_int32();

  if(axis < 0 || axis > 2)
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"Bad axis "<<axis<<"\n";
    }
    exit(1);
  }

  int direction = options["direction"].to_int32();;
  double location = options["location"].to_float64();

  std::string root_file = options["root_file"].as_string();

  conduit::Node dataset;
  dray::BlueprintReader::load_blueprint(root_file, dataset);

  conduit::Node chopped;
  const int domains = dataset.number_of_children();

  for(int i = 0; i < domains; ++i)
  {
    conduit::Node &dom = dataset.child(i);

    //if(i == 0) dom.print();

    conduit::Node &coords = dom["coordsets"].child(0);

    if(coords["type"].as_string() != "explicit")
    {
      std::cout<<"only explicit coordinates supported\n";
    }

    const int total_size = coords["values/x"].dtype().number_of_elements();;
    conduit::float64_array array;

    if(axis == 0)
    {
      array = coords["values/x"].value();
    }
    else if(axis == 1)
    {
      array = coords["values/y"].value();
    }
    else
    {
      array = coords["values/z"].value();
    }

    double min_v = 1e32;
    double max_v = -1e32;
    for(int i = 0; i < total_size; ++i)
    {
      double val = array[i];
      min_v = std::min(min_v,val);
      max_v = std::max(max_v,val);
    }

    bool keep = true;

    if(direction < 0 && max_v > location)
    {
      keep = false;
    }

    if(direction > 0 && min_v < location)
    {
      keep = false;
    }

    if(keep)
    {
      chopped.append().set_external(dom);
    }

  }

  // this needs to be a multi-domain data set
  dray::BlueprintReader::save_blueprint("chopped",chopped);


#ifdef MPI_ENABLED
  MPI_Finalize();
#endif
}
