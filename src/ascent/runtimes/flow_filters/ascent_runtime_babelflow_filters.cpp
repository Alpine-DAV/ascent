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
  if (op == PMT) {
    // connect to the input port and get the parameters
    conduit::Node p = params();
    auto *in = input<conduit::Node>("in");
    auto &data_node = in->children().next();

    // get the data handle
    conduit::DataArray<float> array = data_node[p["data_path"].as_string()].as_float32_array();

    // get the parameters
    MPI_Comm comm = MPI_Comm_f2c(p["mpi_comm"].as_int());
    uint32_t *data_size = p["data_size"].as_uint32_ptr();
    uint32_t *low = p["low"].as_uint32_ptr();
    uint32_t *high = p["high"].as_uint32_ptr();
    uint32_t *n_blocks = p["n_blocks"].as_uint32_ptr();
    uint32_t task_id = p["task_id"].as_uint32();
    uint32_t fanin = p["fanin"].as_uint32();
    FunctionType threshold = p["threshold"].as_float();

    // create ParallelMergeTree instance and run
    ParallelMergeTree pmt(reinterpret_cast<FunctionType *>(array.data_ptr()), task_id, data_size, n_blocks, low, high,
                          fanin, threshold, comm);
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
