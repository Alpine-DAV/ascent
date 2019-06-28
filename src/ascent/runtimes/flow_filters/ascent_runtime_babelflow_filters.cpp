//
// Created by Li, Jixian on 2019-06-04.
//

#include "ascent_runtime_babelflow_filters.h"

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>


void ascent::runtime::filter::BabelFlow::declare_interface(conduit::Node &i)
{
  i["type_name"] = "babelflow";
  i["port_names"].append() = "in";
  i["output_port"] = "false";
}


void ascent::runtime::filter::BabelFlow::execute()
{

  this->mpi_rank = params()["mpi_rank"].as_int();
  this->mpi_size = params()["mpi_size"].as_int();
  if (op == PMT) {
    std::cout << "Rank:" << mpi_rank << " is executing Parallel Merge Tree" << std::endl;
    auto *in = input<conduit::Node>("in");
    auto &data_node = in->children().next();
    conduit::Node p = params();

    conduit::DataArray<double> arr = data_node[p["data_path"].as_string()].as_float64_array();
    int n = arr.number_of_elements();
    std::vector<FunctionType> vdata;
    vdata.resize(n);
    for (int i = 0; i < n; ++i) {
      vdata[i] = static_cast<FunctionType>(arr[i]);
    }

#ifdef BABELFLOW_DEBUG
    if (mpi_rank == 0) {
      std::ofstream ofs("data.txt");
      std::stringstream ss;
      for (auto &&v: vdata) ss << v << " ";
      ofs << ss.str();
      ofs.flush();
      ofs.close();

      std::ofstream bofs("data.bin", std::ios::out | std::ios::binary);
      FunctionType *flt_buffer = vdata.data();
      size_t size = vdata.size() * sizeof(FunctionType);
      auto *buffer = reinterpret_cast<char *>(flt_buffer);
      bofs.write(buffer, size);
      bofs.flush();
      bofs.close();
    }
#endif

    int dim[3] = {p["xdim"].as_int(), p["ydim"].as_int(), p["zdim"].as_int()};
    int block[3] = {p["bxdim"].as_int(), p["bydim"].as_int(), p["bzdim"].as_int()};
    ParallelMergeTree pmt(reinterpret_cast<FunctionType *>(vdata.data()), dim,
                          block, p["fanin"].as_int(), p["threshold"].as_float(),
                          MPI_Comm_f2c(p["mpi_comm"].as_int()));
    pmt.Initialize();
    pmt.Execute();
  } else {
    return;
  }
}

bool ascent::runtime::filter::BabelFlow::verify_params(const conduit::Node &params, conduit::Node &info)
{
  if (params.has_child("task")) {
    std::string task_str(params["task"].as_string());
    if (task_str == "pmt") {
      this->op = PMT;
    } else {
      std::cerr << "[Error] ascent::BabelFlow\nUnknown task \"" << task_str << "\"" << std::endl;
      return false;
    }
  } else {
    std::cerr << "[Error] ascent::BabelFlow\ntask need to be specified" << std::endl;
    return false;
  }
  return true;
}
