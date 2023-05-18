#include <iostream>
#include <fstream>
#include <string>

#include <ascent.h>

#include <catalyst.h>
#include <catalyst_conduit.hpp>
#include <catalyst_conduit_blueprint.hpp>
#include <catalyst_stub.h>

#include <conduit.hpp>
#include <conduit_cpp_to_c.hpp>

#include "catalyst_impl_ascent.h"

#ifdef _USE_MPI
#include <mpi.h>
#endif

namespace detail{

class Instance
{
protected:
    Instance() = default;

private:
    //pointer to the ascent instance
    void* ascent = nullptr;
    std::vector<conduit::Node> actions;
    //instance pointer;
    static Instance* instance;

public:
    // deleting copy constructor
    Instance(const Instance& obj) = delete;

    static Instance* GetInstance()
    {
      if(instance == nullptr)
      {
        instance = new Instance();
        return instance;
      }
      else
      {
        return instance;
      }
    }

    void SetAscent(void* _ascent) { this->ascent = _ascent; }
    void* GetAscent() { return this->ascent; }

    void AddActions(conduit::Node& node)
    {
      actions.emplace_back(node);
    }
    std::vector<conduit::Node> GetActions()
    {
      return this->actions;
    }

    void ClearActions()
    {
      this->actions.clear();
    }
};

}

detail::Instance* detail::Instance::instance = nullptr;

enum catalyst_status catalyst_initialize_ascent(const conduit_node* params)
{
  std::cout << "[pre] Executing Initialize" << std::endl;
  //Convert params to C++ node:
  const conduit_cpp::Node cpp_params = conduit_cpp::cpp_node(const_cast<conduit_node*>(params));
  auto instance = detail::Instance::GetInstance();
  void* ascent = ascent_create();
  if(!ascent)
  {
    std::cerr << "Error creating ascent" << std::endl;
  }
  instance->SetAscent(ascent);
  // Assuming params contain information to initialize ascent
  conduit::Node options;
#ifdef _USE_MPI
  options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
  options["runtime/type"] = "ascent";
  conduit_node* options_c = conduit::c_node(&options);
  ascent_open(ascent, options_c);

  if (cpp_params.has_path("catalyst/scripts"))
  {
    auto& scripts = cpp_params["catalyst/scripts"];
    conduit_index_t nchildren = scripts.number_of_children();
    for (conduit_index_t i = 0; i < nchildren; ++i)
    {
      auto script = scripts.child(i);
      const auto fname =
        script.dtype().is_string() ? script.as_string() : script["filename"].as_string();
      std::cout << "Ascent script : " << fname << std::endl;
      std::ifstream ifs(fname);
      std::string content;
      content.assign((std::istreambuf_iterator<char>(ifs)),
                     (std::istreambuf_iterator<char>()));
      conduit::Node node = conduit::Node();
      node.parse(content, "yaml");
      instance->AddActions(node);
    }
  }

  std::cout << "[post] Executing Initialize" << std::endl;
  return catalyst_status_ok;
}

enum catalyst_status catalyst_execute_ascent(const conduit_node* params)
{
  std::cout << "[pre] Executing Execute" << std::endl;
  const conduit_cpp::Node cpp_params = conduit_cpp::cpp_node(const_cast<conduit_node*>(params));
  conduit_cpp::Node verify_info;
  conduit_cpp::Node data = cpp_params["catalyst/channels/grid/data"];
  if(!conduit_cpp::BlueprintMesh::verify(data,verify_info))
  {
    // show details of what went awry
    verify_info.print();
  }
  // First publish the data using ascent_publish
  auto instance = detail::Instance::GetInstance();
  auto ascent = instance->GetAscent();
  conduit_node* data_c = conduit_cpp::c_node(&data);
  ascent_publish(ascent, const_cast<conduit_node*>(data_c));
  // Then use catalyst_script to convert a ascent_actions.yaml to a conduit node
  // Finally, call ascent_execute with the actions and the data
  std::vector<conduit::Node> actions = instance->GetActions();
  for(const conduit::Node& node : actions)
  {
    const conduit_node* node_c = conduit::c_node(&node);
    ascent_execute(ascent, const_cast<conduit_node*>(node_c));
  }
  std::cout << "[post] Executing Execute" << std::endl;
  return catalyst_status_ok;
}

enum catalyst_status catalyst_finalize_ascent(const conduit_node* params)
{
  std::cout << "[pre] Executing Finalize" << std::endl;
  auto instance = detail::Instance::GetInstance();
  auto ascent = instance->GetAscent();
  if(ascent)
  {
    ascent_close(ascent);
    ascent_destroy(ascent);
    instance->SetAscent(nullptr);
  }
  std::cout << "[post] Executing Finalize" << std::endl;
  return catalyst_status_ok;
}

enum catalyst_status catalyst_about_ascent(conduit_node* params)
{
  std::cout << "[pre] Executing About" << std::endl;
  ascent_about(params);
  std::cout << "[post] Executing About" << std::endl;
  return catalyst_status_ok;
}

enum catalyst_status catalyst_results_ascent(conduit_node* params)
{
  std::cout << "[per] Executing Results" << std::endl;
  // TODO: is this method required for ascent?
  std::cout << "[post] Executing Results" << std::endl;
  return catalyst_status_ok;
}
