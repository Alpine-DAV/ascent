//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_vtkh_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_vtkh_utils.hpp"
#include <ascent_runtime_utils.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

namespace detail
{


void field_error(const std::string field_name,
                 const std::string filter_name,
                 std::shared_ptr<VTKHCollection> collection,
                 bool error)
{
  std::string fpath = filter_to_path(filter_name);
  std::vector<std::string> names = collection->field_names();
  std::stringstream ss;
  ss<<" field names: ";
  for(int i = 0; i < names.size(); ++i)
  {
    ss<<"'"<<names[i]<<"'";
    if(i != names.size() - 1)
    {
      ss<<", ";
    }
  }
  if(error)
  {
    ASCENT_ERROR("("<<fpath<<") unknown field '"<<field_name<<"'"
                 <<ss.str());
  }
  else
  {
    ASCENT_INFO("("<<fpath<<") unknown field '"<<field_name<<"'"
                <<ss.str());
  }
}

std::string possible_topologies(std::shared_ptr<VTKHCollection> collection)
{
   std::stringstream ss;
   ss<<" topology names: ";
   std::vector<std::string> names = collection->topology_names();
   for(int i = 0; i < names.size(); ++i)
   {
     ss<<"'"<<names[i]<<"'";
     if(i != names.size() -1)
     {
       ss<<", ";
     }
   }
   return ss.str();
}

std::string resolve_topology(const conduit::Node &params,
                             const std::string filter_name,
                             std::shared_ptr<VTKHCollection> collection,
                             bool error)
{
  int num_topologies = collection->number_of_topologies();
  std::string topo_name;
  std::string fpath = filter_to_path(filter_name);
  if(num_topologies > 1)
  {
    if(!params.has_path("topology"))
    {
      std::string topo_names = detail::possible_topologies(collection);;
      if(error)
      {
        ASCENT_ERROR(fpath<<": data set has multiple topologies "
                     <<"and no topology is specified. "<<topo_names);
      }
      else
      {
        ASCENT_INFO(fpath<<": data set has multiple topologies "
                     <<"and no topology is specified. "<<topo_names);
        return topo_name;
      }
    }

    topo_name = params["topology"].as_string();
    if(!collection->has_topology(topo_name))
    {
      std::string topo_names = detail::possible_topologies(collection);;
      if(error)
      {
        ASCENT_ERROR(fpath<<": no topology named '"<<topo_name<<"'."
                     <<topo_names);
      }
      else
      {
        ASCENT_INFO(fpath<<": no topology named '"<<topo_name<<"'."
                   <<topo_names);
        return topo_name;
      }

    }
  }
  else
  {
    topo_name = collection->topology_names()[0];

    if(params.has_path("topology"))
    {
      std::string provided = params["topology"].as_string();
      if(topo_name != provided)
      {
        if(error)
        {
          ASCENT_ERROR(fpath<<": provided topology parameter '"<<provided<<"' "
                       <<"does not match the name of the only topology '"
                       <<topo_name<<"'.");
        }
        else
        {
          ASCENT_INFO(fpath<<": provided topology parameter '"<<provided<<"' "
                      <<"does not match the name of the only topology '"
                      <<topo_name<<"'.");
          return "";
        }
      }
    }
  }

  return topo_name;
}

} // namespace detail
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
