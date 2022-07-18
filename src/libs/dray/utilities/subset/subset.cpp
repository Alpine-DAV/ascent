#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/utils/png_encoder.hpp>

#include <set>

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

  if(!options.has_path("root_file"))
  {
    std::cout<<"Missing root file\n";
    exit(1);
  }
  if(!options.has_path("domains"))
  {
    std::cout<<"Missing domain list\n";
    exit(1);
  }
  if(dray::dray::mpi_rank() == 0)
  {
    options.print();
    options.schema().print();
  }
  //if(!options["domains"].schema().dtype().is_list())
  //{
  //  if(dray::dray::mpi_rank() == 0)
  //  {
  //    std::cout<<"domain list is not a list\n";
  //  }
  //  exit(1);
  //}

  std::set<int> domain_set;
  const int set_size = options["domains"].dtype().number_of_elements();
  conduit::int64* list = options["domains"].value();

  for(int i = 0; i < set_size; ++i)
  {
    int id = list[i];
    domain_set.insert(id);
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"adding domain id "<<id<<"\n";
    }
  }

  std::string root_file = options["root_file"].as_string();

  conduit::Node dataset;
  dray::BlueprintReader::load_blueprint(root_file, dataset);

  conduit::Node subset;
  const int domains = dataset.number_of_children();

  for(int i = 0; i < domains; ++i)
  {
    conduit::Node &dom = dataset.child(i);

    int domain_id = dom["state/domain_id"].to_int32();

    if(domain_set.find(domain_id) != domain_set.end())
    {
      subset.append().set_external(dom);
    }

  }

  try
  {
    // this needs to be a multi-domain data set
    dray::BlueprintReader::save_blueprint("subset",subset);
  }
  catch(const dray::DRayError &e)
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"failed: "<<e.what()<<"\n";
    }
  }
  catch(const conduit::Error &e)
  {
    if(dray::dray::mpi_rank() == 0)
    {
      std::cout<<"failed: "<<e.what()<<"\n";
    }
  }

#ifdef MPI_ENABLED
  MPI_Finalize();
#endif
}
