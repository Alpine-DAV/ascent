// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/data_set.hpp>
#include <dray/error.hpp>
#include <algorithm>
#include <sstream>

namespace dray
{

DataSet::DataSet(std::shared_ptr<Mesh> mesh)
 : m_is_valid(true),
   m_domain_id(0)
{
  m_meshes.push_back(mesh);
}

DataSet::DataSet()
 : m_is_valid(false),
   m_domain_id(0)
{
}

void DataSet::domain_id(const int32 id)
{
  m_domain_id = id;
}

int32 DataSet::domain_id() const
{
  return m_domain_id;
}

void DataSet::clear_meshes()
{
  m_meshes.clear();
  m_is_valid = false;
  // should this invalidate the fields? Maybe we have a consistency check
}

void DataSet::clear_fields()
{
  m_fields.clear();
}

void DataSet::add_mesh(std::shared_ptr<Mesh> mesh)
{
  m_meshes.push_back(mesh);
  m_is_valid = true;
}

bool DataSet::has_mesh(const std::string &mesh_name) const
{
  bool found = false;

  for(int32 i = 0; i < m_meshes.size(); ++i)
  {
    if(m_meshes[i]->name() == mesh_name)
    {
      found = true;
      break;
    }
  }
  return found;
}

int32 DataSet::number_of_meshes() const
{
  return static_cast<int32>(m_meshes.size());
}

std::vector<std::string> DataSet::meshes() const
{

  std::vector<std::string> names;
  for(int32 i = 0; i < m_meshes.size(); ++i)
  {
    names.push_back(m_meshes[i]->name());
  }
  return names;
}

int32 DataSet::number_of_fields() const
{
  return m_fields.size();
}

bool DataSet::has_field(const std::string &field_name) const
{
  bool found = false;

  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    if(m_fields[i]->name() == field_name)
    {
      found = true;
      break;
    }
  }
  return found;
}

std::vector<std::string> DataSet::fields() const
{

  std::vector<std::string> names;
  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    names.push_back(m_fields[i]->name());
  }
  return names;
}

Field* DataSet::field(const int &index)
{
  return field_shared(index).get();
}

std::shared_ptr<Field> DataSet::field_shared(const int &index)
{
  if (index < 0 || index >= this->number_of_fields())
  {
    std::stringstream ss;
    ss<<"DataSet: Bad field index "<<index;
    DRAY_ERROR(ss.str());
  }
  return m_fields[index];
}

std::shared_ptr<Field> DataSet::field_shared(const std::string &field_name)
{
  bool found = false;
  int32 index = -1;
  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    if(m_fields[i]->name() == field_name)
    {
      found = true;
      index = i;
      break;
    }
  }

  if (!found)
  {
    std::stringstream ss;
    ss << "Known fields: ";
    std::vector<std::string> names = this->fields();
    for (auto it = names.begin (); it != names.end (); ++it)
    {
      ss << "[" << *it << "] ";
    }

    DRAY_ERROR ("No field named '" + field_name + "' " + ss.str ());
  }

  return m_fields[index];
}

Field* DataSet::field(const std::string &field_name)
{
  return field_shared(field_name).get();
}

Mesh* DataSet::mesh(const int32 mesh_index)
{
  if(!m_is_valid)
  {
    DRAY_ERROR ("Need to set the mesh before asking for it.");
  }

  if(mesh_index < 0 || mesh_index >= m_meshes.size())
  {
    DRAY_ERROR ("Invalid mesh index: "<<mesh_index);
  }
  return m_meshes[mesh_index].get();
}

Mesh* DataSet::mesh(const std::string mesh_name)
{
  int32 index = -1;

  for(int32 i = 0; i < m_meshes.size(); ++i)
  {
    if(m_meshes[i]->name() == mesh_name)
    {
      index = i;
      break;
    }
  }

  if(index == -1)
  {
    DRAY_ERROR ("Unknown mesh '"<<mesh_name<<"'");
  }

  return m_meshes[index].get();
}

void DataSet::add_field(std::shared_ptr<Field> field)
{
  m_fields.push_back(field);
}

std::string DataSet::field_info()
{
  std::stringstream ss;
  for(int i = 0; i < m_fields.size(); ++i)
  {
    ss<<m_fields[i]->name()<<" "<<m_fields[i]->type_name()<<"\n";
  }
  return ss.str();
}

void DataSet::to_node(conduit::Node &n_dataset)
{
  n_dataset.reset();
  conduit::Node &n_meshes = n_dataset["meshes"];
  for(int32 i = 0; i < m_meshes.size(); ++i)
  {
    // Make a name. Make sure it is valid in case the mesh does not have a name.
    std::string name(m_meshes[i]->name());
    if(name.empty())
    {
      if(m_meshes.size() > 1)
      {
        std::ostringstream s;
        s << "mesh" << i;
        name = s.str();
      }
      else
      {
        name = "mesh";
      }
    }
    conduit::Node &n_mesh = n_meshes[name];
    m_meshes[i]->to_node(n_mesh);
  }

  const int32 num_fields = m_fields.size();
  for(int32 i = 0; i < num_fields; ++i)
  {
    std::string name = m_fields[i]->name();
    conduit::Node &n_field = n_dataset["fields/"+name];
    m_fields[i]->to_node(n_field);
  }
}

} // namespace dray
