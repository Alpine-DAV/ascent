// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/error.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/utils/data_logger.hpp>

#include <mfem/fem/conduitdatacollection.hpp>

namespace dray
{

namespace detail
{

mfem::DataCollection *load_collection (const std::string root_file, const int32 cycle)
{
  // start with visit
  mfem::VisItDataCollection *vcol = new mfem::VisItDataCollection (root_file);
  try
  {
    vcol->Load (cycle);
    // apparently failing to open is just a warning...
    if (vcol->GetMesh () == nullptr)
    {
      DRAY_ERROR ("Failed");
    }
    DRAY_INFO ("Load succeeded 'visit data collection'");
    return vcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'visit data collection'");
  }
  delete vcol;

  // now try conduit
  mfem::ConduitDataCollection *dcol = new mfem::ConduitDataCollection (root_file);
  try
  {
    dcol->SetProtocol ("conduit_bin");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'conduit_bin'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'conduit_bin'");
  }

  try
  {
    dcol->SetProtocol ("conduit_json");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'conduit_json'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'conduit_json'");
  }

  try
  {
    dcol->SetProtocol ("json");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'json'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'json'");
  }

  try
  {
    dcol->SetProtocol ("hdf5");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'hdf5'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'hdf5'");
  }


  delete dcol;

  return nullptr;
}

Collection load(const std::string &root_file,
                const int32 cycle,
                const ImportOrderPolicy &import_order_policy)
{

  mfem::DataCollection *dcol = load_collection (root_file, cycle);
  if (dcol == nullptr)
  {
    DRAY_ERROR ("Failed to open file '" + root_file + "'");
  }

  mfem::Mesh *mfem_mesh_ptr;

  mfem_mesh_ptr = dcol->GetMesh ();

  if (mfem_mesh_ptr->NURBSext)
  {
    mfem_mesh_ptr->SetCurvature (2);
  }

  mfem::Geometry::Type geom_type = mfem_mesh_ptr->GetElementBaseGeometry(0);

  if(geom_type != mfem::Geometry::CUBE && geom_type != mfem::Geometry::SQUARE)
  {
    DRAY_ERROR("Only hex and quad imports implemented");
  }

  mfem_mesh_ptr->GetNodes ();

  DataSet dataset = import_mesh(*mfem_mesh_ptr);

  auto field_map = dcol->GetFieldMap ();
  for (auto it = field_map.begin (); it != field_map.end (); ++it)
  {
    const std::string field_name = it->first;
    mfem::GridFunction *grid_ptr = dcol->GetField (field_name);
    const int components = grid_ptr->VectorDim ();

    const mfem::FiniteElementSpace *fespace = grid_ptr->FESpace ();
    const int32 P = fespace->GetOrder (0);
    if (P == 0)
    {
      DRAY_INFO ("Field has unsupported order " << P);
      continue;
    }
    if (components == 1)
    {
      import_field(dataset, *grid_ptr, geom_type, field_name);
    }
    else if (components == 2)
    {
      import_field(dataset, *grid_ptr, geom_type, field_name + "_x", 0);
      import_field(dataset, *grid_ptr, geom_type, field_name + "_y", 1);
    }
    else if (components == 3)
    {
      import_field(dataset, *grid_ptr, geom_type, field_name + "_x", 0);
      import_field(dataset, *grid_ptr, geom_type, field_name + "_y", 1);
      import_field(dataset, *grid_ptr, geom_type, field_name + "_z", 2);
    }
    else
    {
      std::cout << "Import field: number of components = " << components << " not supported \n";
    }
  }

  delete dcol;
  Collection collection;
  collection.add_domain(dataset);
  return collection;
}


} // namespace detail

Collection
MFEMReader::load (const std::string &root_file,
                  const int32 cycle,
                  const ImportOrderPolicy &import_order_policy)
{
  try
  {
    return detail::load (root_file, cycle, import_order_policy);
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'mfem data collection'");
  }
  try
  {
    return BlueprintReader::load (root_file, cycle);
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'blueprint reader'");
  }

  DRAY_ERROR ("Failed to open file '" + root_file + "'");
}
// TODO triangle, 2d, etc.

} // namespace dray
