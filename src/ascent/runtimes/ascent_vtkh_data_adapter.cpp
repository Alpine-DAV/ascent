//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_data_adapter.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_vtkh_data_adapter.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>
#include <type_traits>

// third party includes

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

// VTKm includes
#define VTKM_USE_DOUBLE_PRECISION
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandleExtractComponent.h>
#include <vtkh/DataSet.hpp>
// other ascent includes
#include <ascent_logging.hpp>
#include <ascent_block_timer.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>
#include <vtkh/utils/vtkm_dataset_info.hpp>

#include <conduit_blueprint.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin detail:: --
//-----------------------------------------------------------------------------
namespace detail
{

vtkm::Id3 topo_origin(const conduit::Node &n_topo)
{
  vtkm::Id3 topo_origin(0,0,0);
  // maintain backwards compatibility between
  // i and i0 versions
  if(n_topo.has_path("elements/origin"))
  {
    const conduit::Node &origin = n_topo["elements/origin"];

    if(origin.has_path("i"))
    {
      topo_origin[0] = n_topo["elements/origin/i"].to_int32();
    }
    if(origin.has_path("i0"))
    {
      topo_origin[0] = n_topo["elements/origin/i0"].to_int32();
    }

    if(origin.has_path("j"))
    {
      topo_origin[1] = n_topo["elements/origin/j"].to_int32();
    }
    if(origin.has_path("j0"))
    {
      topo_origin[1] = n_topo["elements/origin/j0"].to_int32();
    }

    if(origin.has_path("k"))
    {
      topo_origin[2] = n_topo["elements/origin/k"].to_int32();
    }
    if(origin.has_path("k0"))
    {
      topo_origin[2] = n_topo["elements/origin/k0"].to_int32();
    }
  }

  return topo_origin;
}

template<typename T>
const T* GetNodePointer(const conduit::Node &node);

template<>
const float64* GetNodePointer<float64>(const conduit::Node &node)
{
  return node.as_float64_ptr();
}

template<>
const float32* GetNodePointer<float32>(const conduit::Node &node)
{
  return node.as_float32_ptr();
}

template<typename T>
void CopyArray(vtkm::cont::ArrayHandle<T> &vtkm_handle, const T* vals_ptr, const int size, bool zero_copy)
{
  vtkm::CopyFlag copy = vtkm::CopyFlag::On;
  if(zero_copy)
  {
    copy = vtkm::CopyFlag::Off;
  }

  vtkm_handle = vtkm::cont::make_ArrayHandle(vals_ptr, size, copy);
}

template<typename T>
vtkm::cont::CoordinateSystem
GetExplicitCoordinateSystem(const conduit::Node &n_coords,
                            const std::string name,
                            int &ndims,
                            bool zero_copy)
{
    int nverts = n_coords["values/x"].dtype().number_of_elements();
    bool is_interleaved = blueprint::mcarray::is_interleaved(n_coords["values"]);

    // some interleaved cases aren't working
    // disabling this path until we find out what is going wrong.
    is_interleaved = false;

    ndims = 2;

    // n_coords_conv holds contig data if we have stride-ed but
    // non-interleaved values
    Node n_coords_conv;

    const T* x_coords_ptr = NULL;
    const T* y_coords_ptr = NULL;
    const T *z_coords_ptr = NULL;

    // if we are an interleaved mcarray, or compact we can
    // directly use the pointer with vtk-m.
    // otherwise, we need to compact.

    if(is_interleaved || n_coords["values/x"].is_compact())
    {
        x_coords_ptr = GetNodePointer<T>(n_coords["values/x"]);
    }
    else
    {
        n_coords["values/x"].compact_to(n_coords_conv["x"]);
        x_coords_ptr = GetNodePointer<T>(n_coords_conv["x"]);
        // since we had to copy and compact the data, we can't zero copy
        zero_copy = false;
    }

    if(is_interleaved || n_coords["values/y"].is_compact())
    {
        y_coords_ptr = GetNodePointer<T>(n_coords["values/y"]);
    }
    else
    {
        n_coords["values/y"].compact_to(n_coords_conv["y"]);
        y_coords_ptr = GetNodePointer<T>(n_coords_conv["y"]);
        // since we had to copy and compact the data, we can't zero copy
        zero_copy = false;
    }

    if(n_coords.has_path("values/z"))
    {
        ndims = 3;
        if(is_interleaved || n_coords["values/z"].is_compact())
        {
            z_coords_ptr = GetNodePointer<T>(n_coords["values/z"]);
        }
        else
        {
            n_coords["values/z"].compact_to(n_coords_conv["z"]);
            z_coords_ptr = GetNodePointer<T>(n_coords_conv["z"]);
            // since we had to copy and compact the data, we can't zero copy
            zero_copy = false;
        }
    }

    if(!is_interleaved)
    {
      vtkm::cont::ArrayHandle<T> x_coords_handle;
      vtkm::cont::ArrayHandle<T> y_coords_handle;
      vtkm::cont::ArrayHandle<T> z_coords_handle;

      detail::CopyArray(x_coords_handle, x_coords_ptr, nverts, zero_copy);
      detail::CopyArray(y_coords_handle, y_coords_ptr, nverts, zero_copy);

      if(ndims == 3)
      {
        detail::CopyArray(z_coords_handle, z_coords_ptr, nverts, zero_copy);
      }
      else
      {
          z_coords_handle.Allocate(nverts);
          // This does not get initialized to zero
          T *z = vtkh::GetVTKMPointer(z_coords_handle);
          memset(z, 0.0, nverts * sizeof(T));
      }

      return vtkm::cont::CoordinateSystem(name,
                                          make_ArrayHandleSOA(x_coords_handle,
                                                              y_coords_handle,
                                                              z_coords_handle));
    }
    else // NOTE: This case is disabled.
    {
      // we have interleaved coordinates x0,y0,z0,x1,y1,z1...
      const T* coords_ptr = GetNodePointer<T>(n_coords["values/x"]);
      vtkm::cont::ArrayHandle<vtkm::Vec<T, 3>> coords;
      // we cannot zero copy 2D interleaved arrays into vtkm
      if(ndims == 3 || true) // TODO: need way to detect 3d interleaved components that has
                             //       only has xy in conduit
      {
        // this case was failing from Nyx + AMReX
        // still haven't been able to reproduce with a simpler test
        detail::CopyArray(coords, (vtkm::Vec<T,3>*)coords_ptr, nverts, zero_copy);
      }
      else
      {
        // 2D interleaved array case
        vtkm::cont::ArrayHandle<T> x_coords_handle;
        vtkm::cont::ArrayHandle<T> y_coords_handle;
        vtkm::cont::ArrayHandle<T> z_coords_handle;

        x_coords_handle.Allocate(nverts);
        y_coords_handle.Allocate(nverts);
        z_coords_handle.Allocate(nverts);

        auto x_portal = x_coords_handle.WritePortal();
        auto y_portal = y_coords_handle.WritePortal();

        const T* coords_ptr = GetNodePointer<T>(n_coords["values/x"]);

        T *z = (T*) vtkh::GetVTKMPointer(z_coords_handle);
        memset(z, 0.0, nverts * sizeof(T));

        for(int i = 0; i < nverts; ++i)
        {
          x_portal.Set(i, coords_ptr[i*2+0]);
          y_portal.Set(i, coords_ptr[i*2+1]);
        }

        return vtkm::cont::CoordinateSystem(name,
                                            make_ArrayHandleSOA(x_coords_handle,
                                                                y_coords_handle,
                                                                z_coords_handle));
      }

      return vtkm::cont::CoordinateSystem(name, coords);
    }

}

template<typename T>
vtkm::cont::Field GetField(const conduit::Node &node,
                           const std::string field_name,
                           const std::string assoc_str,
                           const std::string topo_str,
                           bool zero_copy)
{
  vtkm::CopyFlag copy = vtkm::CopyFlag::On;
  if(zero_copy)
  {
    copy = vtkm::CopyFlag::Off;
  }
  vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::ANY;
  if(assoc_str == "vertex")
  {
    vtkm_assoc = vtkm::cont::Field::Association::POINTS;
  }
  else if(assoc_str == "element")
  {
    vtkm_assoc = vtkm::cont::Field::Association::CELL_SET;
  }
  else
  {
    ASCENT_ERROR("Cannot add field association "<<assoc_str<<" from field "<<field_name);
  }

  int num_vals = node.dtype().number_of_elements();

  const T *values_ptr = node.value();

  vtkm::cont::Field field;
  field = vtkm::cont::make_Field(field_name,
                                 vtkm_assoc,
                                 values_ptr,
                                 num_vals,
                                 copy);
  return field;
}

template<typename T>
vtkm::cont::Field GetVectorField(T *values_ptr,
                                 const int num_vals,
                                 const std::string field_name,
                                 const std::string assoc_str,
                                 const std::string topo_str,
                                 bool zero_copy)
{
  vtkm::CopyFlag copy = vtkm::CopyFlag::On;
  if(zero_copy)
  {
    copy = vtkm::CopyFlag::Off;
  }
  vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::ANY;
  if(assoc_str == "vertex")
  {
    vtkm_assoc = vtkm::cont::Field::Association::POINTS;
  }
  else if(assoc_str == "element")
  {
    vtkm_assoc = vtkm::cont::Field::Association::CELL_SET;
  }
  else
  {
    ASCENT_ERROR("Cannot add vector field with association "
                 <<assoc_str<<" field_name "<<field_name);
  }

  vtkm::cont::Field field;
  field = vtkm::cont::make_Field(field_name,
                                 vtkm_assoc,
                                 values_ptr,
                                 num_vals,
                                 copy);

  return field;
}

//
// extract a vector from 3 separate arrays
//
template<typename T>
void ExtractVector(vtkm::cont::DataSet *dset,
                   const conduit::Node &u,
                   const conduit::Node &v,
                   const conduit::Node &w,
                   const int num_vals,
                   const int dims,
                   const std::string field_name,
                   const std::string assoc_str,
                   const std::string topo_name,
                   bool zero_copy)
{
  // TODO: Do we need to fix this for striding?
  // GetField<T> expects compact
  if(dims != 2 && dims != 3)
  {
    ASCENT_ERROR("Extract vector: only 2 and 3 dims supported given "<<dims);
  }

  vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::ANY;
  if(assoc_str == "vertex")
  {
    vtkm_assoc = vtkm::cont::Field::Association::POINTS;
  }
  else if (assoc_str == "element")
  {
    vtkm_assoc = vtkm::cont::Field::Association::CELL_SET;
  }
  else
  {
    ASCENT_ERROR("Cannot add vector field with association "
                 <<assoc_str<<" field_name "<<field_name);
  }

  if(dims == 2)
  {
    const T *x_ptr = GetNodePointer<T>(u);
    const T *y_ptr = GetNodePointer<T>(v);

    vtkm::cont::ArrayHandle<T> x_handle;
    vtkm::cont::ArrayHandle<T> y_handle;

    // always zero copy because we are about to make a copy
    detail::CopyArray(x_handle, x_ptr, num_vals, true);
    detail::CopyArray(y_handle, y_ptr, num_vals, true);


    auto composite  = make_ArrayHandleSOA(x_handle,
                                          y_handle);

    vtkm::cont::ArrayHandle<vtkm::Vec<T,2>> interleaved_handle;
    interleaved_handle.Allocate(num_vals);
    // Calling this without forcing serial could cause serious problems
    {
      vtkm::cont::ScopedRuntimeDeviceTracker tracker(vtkm::cont::DeviceAdapterTagSerial{});
      vtkm::cont::ArrayCopy(composite, interleaved_handle);
    }

    vtkm::cont::Field field(field_name, vtkm_assoc, interleaved_handle);
    dset->AddField(field);
  }

  if(dims == 3)
  {
    const T *x_ptr = GetNodePointer<T>(u);
    const T *y_ptr = GetNodePointer<T>(v);
    const T *z_ptr = GetNodePointer<T>(w);

    vtkm::cont::ArrayHandle<T> x_handle;
    vtkm::cont::ArrayHandle<T> y_handle;
    vtkm::cont::ArrayHandle<T> z_handle;

    // always zero copy because we are about to make a copy
    detail::CopyArray(x_handle, x_ptr, num_vals, true);
    detail::CopyArray(y_handle, y_ptr, num_vals, true);
    detail::CopyArray(z_handle, z_ptr, num_vals, true);

    auto composite  = make_ArrayHandleSOA(x_handle,
                                          y_handle,
                                          z_handle);

    vtkm::cont::ArrayHandle<vtkm::Vec<T,3>> interleaved_handle;
    interleaved_handle.Allocate(num_vals);
    // Calling this without forcing serial could cause serious problems
    {
      vtkm::cont::ScopedRuntimeDeviceTracker tracker(vtkm::cont::DeviceAdapterTagSerial{});
      vtkm::cont::ArrayCopy(composite, interleaved_handle);
    }

    vtkm::cont::Field field(field_name, vtkm_assoc, interleaved_handle);
    dset->AddField(field);
  }
}


void VTKmCellShape(const std::string shape_type,
                   vtkm::UInt8 &shape_id,
                   vtkm::IdComponent &num_indices)
{
  shape_id = 0;
  num_indices = 0;
  if(shape_type == "tri")
  {
      shape_id = 5;
      num_indices = 3;
  }
  else if(shape_type == "quad")
  {
      shape_id = 9;
      num_indices = 4;
  }
  else if(shape_type == "tet")
  {
      shape_id = 10;
      num_indices = 4;
  }
  else if(shape_type == "hex")
  {
      shape_id = 12;
      num_indices = 8;
  }
  else if(shape_type == "point")
  {
      shape_id = 1;
      num_indices = 1;
  }
  else if(shape_type == "line")
  {
      shape_id = 3;
      num_indices = 2;
  }
  else
  {
    ASCENT_ERROR("Unsupported cell type "<<shape_type);
  }
}

};
//-----------------------------------------------------------------------------
// -- end detail:: --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// VTKHDataAdapter public methods
//-----------------------------------------------------------------------------

VTKHCollection*
VTKHDataAdapter::BlueprintToVTKHCollection(const conduit::Node &n,
                                           bool zero_copy)
{
    // We must separate different topologies into
    // different vtkh data sets

    const int num_domains = n.number_of_children();

    VTKHCollection *res = new VTKHCollection();
    std::map<std::string, vtkh::DataSet> datasets;
    vtkm::UInt64 cycle = 0;
    double time = 0;

    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = n.child(i);
      const std::vector<std::string> topo_names  = dom["topologies"].child_names();

      if(!dom.has_path("state/domain_id"))
      {
        ASCENT_ERROR("Must have a domain_id to convert blueprint to vtkh");
      }

      int domain_id = dom["state/domain_id"].to_int();

      if(dom.has_path("state/cycle"))
      {
        cycle = dom["state/cycle"].to_uint64();
      }

      if(dom.has_path("state/time"))
      {
        time = dom["state/time"].to_float64();
      }

      for(int t = 0; t < topo_names.size(); ++t)
      {
        const std::string topo_name = topo_names[t];
        vtkm::cont::DataSet *dset = BlueprintToVTKmDataSet(dom, zero_copy, topo_name);
        datasets[topo_name].AddDomain(*dset,domain_id);
        delete dset;
      }

    }

    for(auto dset_it : datasets)
    {
      res->add(dset_it.second, dset_it.first);
    }

    return res;
}

//-----------------------------------------------------------------------------
vtkh::DataSet *
VTKHDataAdapter::BlueprintToVTKHDataSet(const Node &node,
                                        const std::string &topo_name,
                                        bool zero_copy)
{

    // treat everything as a multi-domain data set

    vtkh::DataSet *res = new vtkh::DataSet;

    int num_domains = 0;

    // get the number of domains and check for id consistency
    num_domains = node.number_of_children();

    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = node.child(i);
      vtkm::cont::DataSet *dset = VTKHDataAdapter::BlueprintToVTKmDataSet(dom,
                                                                          zero_copy,
                                                                          topo_name);
      int domain_id = dom["state/domain_id"].to_int();

      if(dom.has_path("state/cycle"))
      {
        vtkm::UInt64 cycle = dom["state/cycle"].to_uint64();
        res->SetCycle(cycle);
      }

      res->AddDomain(*dset,domain_id);
      // vtk-m will shallow copy the data assoced with dset
      // clean up our copy
      delete dset;

    }
    return res;
}

//-----------------------------------------------------------------------------
vtkh::DataSet *
VTKHDataAdapter::VTKmDataSetToVTKHDataSet(vtkm::cont::DataSet *dset)
{
    // wrap a single VTKm data set into a VTKH dataset
    vtkh::DataSet   *res = new  vtkh::DataSet;
    int domain_id = 0; // TODO, MPI_TASK_ID ?
    res->AddDomain(*dset,domain_id);
    return res;
}

//-----------------------------------------------------------------------------
vtkm::cont::DataSet *
VTKHDataAdapter::BlueprintToVTKmDataSet(const Node &node,
                                        bool zero_copy,
                                        const std::string &topo_name_str)
{
    vtkm::cont::DataSet * result = NULL;

    std::string topo_name = topo_name_str;

    // we must find the topolgy they asked for
    if(!node["topologies"].has_child(topo_name))
    {
        ASCENT_ERROR("Invalid topology name: " << topo_name);
    }

    // as long as mesh blueprint verify true, we access data without fear.

    const Node &n_topo   = node["topologies"][topo_name];
    string mesh_type     = n_topo["type"].as_string();

    string coords_name   = n_topo["coordset"].as_string();
    const Node &n_coords = node["coordsets"][coords_name];

    int neles  = 0;
    int nverts = 0;

    if( mesh_type ==  "uniform")
    {
        result = UniformBlueprintToVTKmDataSet(coords_name,
                                               n_coords,
                                               topo_name,
                                               n_topo,
                                               neles,
                                               nverts);
    }
    else if(mesh_type == "rectilinear")
    {
        result = RectilinearBlueprintToVTKmDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts,
                                                   zero_copy);

    }
    else if(mesh_type == "structured")
    {
        result =  StructuredBlueprintToVTKmDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts,
                                                   zero_copy);
    }
    else if( mesh_type ==  "unstructured")
    {
        result =  UnstructuredBlueprintToVTKmDataSet(coords_name,
                                                     n_coords,
                                                     topo_name,
                                                     n_topo,
                                                     neles,
                                                     nverts,
                                                     zero_copy);
    }
    else
    {
        ASCENT_ERROR("Unsupported topology/type:" << mesh_type);
    }


    if(node.has_child("fields"))
    {
        // add all of the fields:
        NodeConstIterator itr = node["fields"].children();
        while(itr.has_next())
        {

            const Node &n_field = itr.next();
            std::string field_name = itr.name();
            if(n_field["topology"].as_string() != topo_name)
            {
              // these are not the fields we are looking for
              continue;
            }

            // skip vector fields for now, we need to add
            // more logic to AddField
            const int num_children = n_field["values"].number_of_children();

            if(num_children == 0 || num_children == 1)
            {

                AddField(field_name,
                         n_field,
                         topo_name,
                         neles,
                         nverts,
                         result,
                         zero_copy);
            }
            else if(num_children == 2 )
            {
              AddVectorField(field_name,
                             n_field,
                             topo_name,
                             neles,
                             nverts,
                             result,
                             2,
                             zero_copy);
            }
            else if(num_children == 3 )
            {
              AddVectorField(field_name,
                             n_field,
                             topo_name,
                             neles,
                             nverts,
                             result,
                             3,
                             zero_copy);
            }
            else
            {
              ASCENT_INFO("skipping field "<<field_name<<" with "<<num_children<<" comps");
            }
        }
    }
    return result;
}


//-----------------------------------------------------------------------------
class ExplicitArrayHelper
{
public:
// Helper function to create explicit coordinate arrays for vtkm data sets
void CreateExplicitArrays(vtkm::cont::ArrayHandle<vtkm::UInt8> &shapes,
                          vtkm::cont::ArrayHandle<vtkm::IdComponent> &num_indices,
                          const std::string &shape_type,
                          const vtkm::Id &conn_size,
                          vtkm::IdComponent &dimensionality,
                          int &neles)
{
    vtkm::UInt8 shape_id = 0;
    vtkm::IdComponent indices= 0;
    if(shape_type == "tri")
    {
        shape_id = 3;
        indices = 3;
        // note: vtkm cell dimensions are topological
        dimensionality = 2;
    }
    else if(shape_type == "quad")
    {
        shape_id = 9;
        indices = 4;
        // note: vtkm cell dimensions are topological
        dimensionality = 2;
    }
    else if(shape_type == "tet")
    {
        shape_id = 10;
        indices = 4;
        dimensionality = 3;
    }
    else if(shape_type == "hex")
    {
        shape_id = 12;
        indices = 8;
        dimensionality = 3;
    }
    else if(shape_type == "points")
    {
        shape_id = 1;
        indices = 1;
        dimensionality = 1;
    }
    // TODO: Not supported in blueprint yet ...
    // else if(shape_type == "wedge")
    // {
    //     shape_id = 13;
    //     indices = 6;
    //     dimensionality = 3;
    // }
    // else if(shape_type == "pyramid")
    // {
    //     shape_id = 14;
    //     indices = 5;
    //     dimensionality = 3;
    // }
    else
    {
        ASCENT_ERROR("Unsupported element shape " << shape_type);
    }

    if(conn_size < indices)
        ASCENT_ERROR("Connectivity array size " <<conn_size << " must be at least size " << indices);
    if(conn_size % indices != 0)
        ASCENT_ERROR("Connectivity array size " <<conn_size << " be evenly divided by indices size" << indices);

    const vtkm::Id num_shapes = conn_size / indices;

    neles = num_shapes;

    shapes.Allocate(num_shapes);
    num_indices.Allocate(num_shapes);

    // We could memset these and then zero copy them but that
    // would make us responsible for the data. If we just create
    // them, smart pointers will automatically delete them.
    // Hopefull the compiler turns this into a memset.

    const vtkm::UInt8 shape_value = shape_id;
    const vtkm::IdComponent indices_value = indices;
    auto shapes_portal = shapes.WritePortal();
    auto num_indices_portal = num_indices.WritePortal();
#ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_shapes; ++i)
    {
        shapes_portal.Set(i, shape_value);
        num_indices_portal.Set(i, indices_value);
    }
}
};
//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::UniformBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed uniform)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts)                    // output, number of verts
{
    //
    // blueprint uniform coord set provides:
    //
    //  dims/{i,j,k}
    //  origin/{x,y,z} (optional)
    //  spacing/{dx,dy,dz} (optional)

    //Create implicit vtkm coordinate system
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    const Node &n_dims = n_coords["dims"];

    int dims_i = n_dims["i"].to_int();
    int dims_j = n_dims["j"].to_int();
    int dims_k = 1;

    bool is_2d = true;

    // check for 3d
    if(n_dims.has_path("k"))
    {
        dims_k = n_dims["k"].to_int();
        is_2d = false;
    }



    float64 origin_x = 0.0;
    float64 origin_y = 0.0;
    float64 origin_z = 0.0;


    float64 spacing_x = 1.0;
    float64 spacing_y = 1.0;
    float64 spacing_z = 1.0;


    if(n_coords.has_child("origin"))
    {
        const Node &n_origin = n_coords["origin"];

        if(n_origin.has_child("x"))
        {
            origin_x = n_origin["x"].to_float64();
        }

        if(n_origin.has_child("y"))
        {
            origin_y = n_origin["y"].to_float64();
        }

        if(n_origin.has_child("z"))
        {
            origin_z = n_origin["z"].to_float64();
        }
    }

    if(n_coords.has_path("spacing"))
    {
        const Node &n_spacing = n_coords["spacing"];

        if(n_spacing.has_path("dx"))
        {
            spacing_x = n_spacing["dx"].to_float64();
        }

        if(n_spacing.has_path("dy"))
        {
            spacing_y = n_spacing["dy"].to_float64();
        }

        if(n_spacing.has_path("dz"))
        {
            spacing_z = n_spacing["dz"].to_float64();
        }
    }

    // todo, should this be float64 -- or should we read float32 above?

    vtkm::Vec<vtkm::Float32,3> origin(origin_x,
                                      origin_y,
                                      origin_z);

    vtkm::Vec<vtkm::Float32,3> spacing(spacing_x,
                                       spacing_y,
                                       spacing_z);

    vtkm::Id3 dims(dims_i,
                   dims_j,
                   dims_k);

    // todo, use actually coordset and topo names?
    result->AddCoordinateSystem( vtkm::cont::CoordinateSystem(coords_name.c_str(),
                                                              dims,
                                                              origin,
                                                              spacing));
    vtkm::Id3 topo_origin = detail::topo_origin(n_topo);
    if(is_2d)
    {
      vtkm::Id2 dims2(dims[0], dims[1]);
      vtkm::cont::CellSetStructured<2> cell_set;
      cell_set.SetPointDimensions(dims2);
      vtkm::Id2 origin2(topo_origin[0], topo_origin[1]);
      cell_set.SetGlobalPointIndexStart(origin2);
      result->SetCellSet(cell_set);
    }
    else
    {
      vtkm::cont::CellSetStructured<3> cell_set;
      cell_set.SetPointDimensions(dims);
      cell_set.SetGlobalPointIndexStart(topo_origin);
      result->SetCellSet(cell_set);
    }

    neles =  (dims_i - 1) * (dims_j - 1);
    if(dims_k > 1)
    {
        neles *= (dims_k - 1);
    }

    nverts =  dims_i * dims_j;
    if(dims_k > 1)
    {
        nverts *= dims_k;
    }

    return result;
}


//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::RectilinearBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts,                    // output, number of verts
     bool zero_copy)                 // attempt to zero copy
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    int x_npts = n_coords["values/x"].dtype().number_of_elements();
    int y_npts = n_coords["values/y"].dtype().number_of_elements();
    int z_npts = 0;

    int32 ndims = 2;

    // todo assumes float64
    const float64 *x_coords_ptr = n_coords["values/x"].as_float64_ptr();
    const float64 *y_coords_ptr = n_coords["values/y"].as_float64_ptr();
    const float64 *z_coords_ptr = NULL;

    if(n_coords.has_path("values/z"))
    {
        ndims = 3;
        z_npts = n_coords["values/z"].dtype().number_of_elements();
        z_coords_ptr = n_coords["values/z"].as_float64_ptr();
    }

    vtkm::cont::ArrayHandle<vtkm::Float64> x_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> y_coords_handle;
    vtkm::cont::ArrayHandle<vtkm::Float64> z_coords_handle;

    if(zero_copy)
    {
      x_coords_handle = vtkm::cont::make_ArrayHandle(x_coords_ptr, x_npts, vtkm::CopyFlag::Off);
      y_coords_handle = vtkm::cont::make_ArrayHandle(y_coords_ptr, y_npts, vtkm::CopyFlag::Off);
    }
    else
    {
      x_coords_handle.Allocate(x_npts);
      y_coords_handle.Allocate(y_npts);

      vtkm::Float64 *x = vtkh::GetVTKMPointer(x_coords_handle);
      memcpy(x, x_coords_ptr, sizeof(float64) * x_npts);
      vtkm::Float64 *y = vtkh::GetVTKMPointer(y_coords_handle);
      memcpy(y, y_coords_ptr, sizeof(float64) * y_npts);
    }

    if(ndims == 3)
    {
      if(zero_copy)
      {
        z_coords_handle = vtkm::cont::make_ArrayHandle(z_coords_ptr, z_npts, vtkm::CopyFlag::Off);
      }
      else
      {
        z_coords_handle.Allocate(z_npts);
        vtkm::Float64 *z = vtkh::GetVTKMPointer(z_coords_handle);
        memcpy(z, z_coords_ptr, sizeof(float64) * z_npts);
      }
    }
    else
    {
        z_coords_handle.Allocate(1);
        z_coords_handle.WritePortal().Set(0, 0.0);
    }

    static_assert(std::is_same<vtkm::FloatDefault, double>::value,
                  "VTK-m needs to be configured with 'VTKm_USE_DOUBLE_PRECISION=ON'");
    vtkm::cont::ArrayHandleCartesianProduct<
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
        vtkm::cont::ArrayHandle<vtkm::FloatDefault> > coords;

    coords = vtkm::cont::make_ArrayHandleCartesianProduct(x_coords_handle,
                                                          y_coords_handle,
                                                          z_coords_handle);

    vtkm::cont::CoordinateSystem coordinate_system(coords_name.c_str(),
                                                  coords);
    result->AddCoordinateSystem(coordinate_system);

    vtkm::Id3 topo_origin = detail::topo_origin(n_topo);

    if (ndims == 2)
    {
      vtkm::cont::CellSetStructured<2> cell_set;
      cell_set.SetPointDimensions(vtkm::make_Vec(x_npts,
                                                 y_npts));
      vtkm::Id2 origin2(topo_origin[0], topo_origin[1]);
      cell_set.SetGlobalPointIndexStart(origin2);
      result->SetCellSet(cell_set);
    }
    else
    {
      vtkm::cont::CellSetStructured<3> cell_set;
      cell_set.SetPointDimensions(vtkm::make_Vec(x_npts,
                                                 y_npts,
                                                 z_npts));
      cell_set.SetGlobalPointIndexStart(topo_origin);
      result->SetCellSet(cell_set);
    }

    nverts = x_npts * y_npts;
    neles = (x_npts - 1) * (y_npts - 1);
    if(ndims > 2)
    {
        nverts *= z_npts;
        neles *= (z_npts - 1);
    }

    return result;
}

//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::StructuredBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts,                    // output, number of verts
     bool zero_copy)                 // attempt to zero copy
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();

    int ndims = 0;

    vtkm::cont::CoordinateSystem coords;
    if(n_coords["values/x"].dtype().is_float64())
    {
      coords = detail::GetExplicitCoordinateSystem<float64>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            zero_copy);
    }
    else if(n_coords["values/x"].dtype().is_float32())
    {
      coords = detail::GetExplicitCoordinateSystem<float32>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            zero_copy);
    }
    else
    {
      ASCENT_ERROR("Coordinate system must be floating point values");
    }

    result->AddCoordinateSystem(coords);

    int32 x_elems = n_topo["elements/dims/i"].as_int32();
    int32 y_elems = n_topo["elements/dims/j"].as_int32();

    vtkm::Id3 topo_origin = detail::topo_origin(n_topo);

    if (ndims == 2)
    {
      vtkm::cont::CellSetStructured<2> cell_set;
      cell_set.SetPointDimensions(vtkm::make_Vec(x_elems+1,
                                                 y_elems+1));
      vtkm::Id2 origin2(topo_origin[0], topo_origin[1]);
      cell_set.SetGlobalPointIndexStart(origin2);
      result->SetCellSet(cell_set);
      neles = x_elems * y_elems;
    }
    else
    {
      int32 z_elems = n_topo["elements/dims/k"].as_int32();
      vtkm::cont::CellSetStructured<3> cell_set;
      cell_set.SetPointDimensions(vtkm::make_Vec(x_elems+1,
                                                 y_elems+1,
                                                 z_elems+1));
      cell_set.SetGlobalPointIndexStart(topo_origin);
      result->SetCellSet(cell_set);
      neles = x_elems * y_elems * z_elems;

    }
    return result;
}



//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::UnstructuredBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed unstructured)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts,                    // output, number of verts
     bool zero_copy)                 // attempt to zero copy
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();

    int32 ndims;
    vtkm::cont::CoordinateSystem coords;
    if(n_coords["values/x"].dtype().is_float64())
    {
      coords = detail::GetExplicitCoordinateSystem<float64>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            zero_copy);
    }
    else if(n_coords["values/x"].dtype().is_float32())
    {
      coords = detail::GetExplicitCoordinateSystem<float32>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            zero_copy);
    }
    else
    {
      ASCENT_ERROR("Coordinate system must be floating point values");
    }

    result->AddCoordinateSystem(coords);

    // shapes, number of indices, and connectivity.
    // Will have to do something different if this is a "zoo"

    // TODO: there is a special data set type for single cell types

    const Node &n_topo_eles = n_topo["elements"];
    std::string ele_shape = n_topo_eles["shape"].as_string();

    // TODO: assumes int32, and contiguous

    const Node &n_topo_conn = n_topo_eles["connectivity"];

    vtkm::cont::ArrayHandle<vtkm::Id> connectivity;

    int conn_size = n_topo_conn.dtype().number_of_elements();

    if( sizeof(vtkm::Id) == 4)
    {
         if(n_topo_conn.is_compact() && n_topo_conn.dtype().is_int32())
         {
           const void *ele_idx_ptr = n_topo_conn.data_ptr();
           detail::CopyArray(connectivity, (const vtkm::Id*)ele_idx_ptr, conn_size,zero_copy);
         }
         else
         {
             // convert to int32
             connectivity.Allocate(conn_size);
             void *ptr = (void*) vtkh::GetVTKMPointer(connectivity);
             Node n_tmp;
             n_tmp.set_external(DataType::int32(conn_size),ptr);
             n_topo_conn.to_int32_array(n_tmp);
        }
    }
    else
    {
        if(n_topo_conn.is_compact() && n_topo_conn.dtype().is_int64())
        {
            const void *ele_idx_ptr = n_topo_conn.data_ptr();
            detail::CopyArray(connectivity, (const vtkm::Id*)ele_idx_ptr, conn_size, zero_copy);
        }
        else
        {
             // convert to int64
             connectivity.Allocate(conn_size);
             void *ptr = (void*) vtkh::GetVTKMPointer(connectivity);
             Node n_tmp;
             n_tmp.set_external(DataType::int64(conn_size),ptr);
             n_topo_conn.to_int64_array(n_tmp);
        }
    }

    vtkm::UInt8 shape_id;
    vtkm::IdComponent indices_per;
    detail::VTKmCellShape(ele_shape, shape_id, indices_per);
    vtkm::cont::CellSetSingleType<> cellset;
    cellset.Fill(nverts, shape_id, indices_per, connectivity);
    neles = cellset.GetNumberOfCells();
    result->SetCellSet(cellset);

    return result;
}

//-----------------------------------------------------------------------------

void
VTKHDataAdapter::AddField(const std::string &field_name,
                          const Node &n_field,
                          const std::string &topo_name,
                          int neles,
                          int nverts,
                          vtkm::cont::DataSet *dset,
                          bool zero_copy)                 // attempt to zero copy
{
    // TODO: how do we deal with vector valued fields?, these will be mcarrays

    string assoc_str = n_field["association"].as_string();

    vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::ANY;
    if(assoc_str == "vertex")
    {
      vtkm_assoc = vtkm::cont::Field::Association::POINTS;
    }
    else if(assoc_str == "element")
    {
      vtkm_assoc = vtkm::cont::Field::Association::CELL_SET;
    }
    else
    {
      ASCENT_INFO("VTKm conversion does not support field assoc "<<assoc_str<<". Skipping");
      return;
    }
    if(n_field["values"].number_of_children() > 1)
    {
      ASCENT_ERROR("Add field can only use zero or one component");
    }

    bool is_values = n_field["values"].number_of_children() == 0;
    const Node &n_vals = is_values ? n_field["values"] : n_field["values"].child(0);
    int num_vals = n_vals.dtype().number_of_elements();

    if(assoc_str == "vertex" && nverts != num_vals)
    {
      ASCENT_INFO("Field '"<<field_name<<"' (topology: '" << topo_name <<
                  "') number of values "<<num_vals<<
                  " does not match the number of points "<<nverts<<". Skipping");
      return;
    }

    if(assoc_str == "element" && neles != num_vals)
    {
      if(field_name != "boundary_attribute")
      {
        ASCENT_INFO("Field '"<<field_name<<"' (topology: '" << topo_name  <<
                    "') number of values "<<num_vals<<
                    " does not match the number of elements " << neles << ". Skipping");
      }
      return;
    }

    try
    {
        bool supported_type = false;

        if(n_vals.is_compact())
        {
            // we compile vtk-h with fp types
            if(n_vals.dtype().is_float32())
            {
                dset->AddField(detail::GetField<float32>(n_vals, field_name, assoc_str, topo_name, zero_copy));
                supported_type = true;
            }
            else if(n_vals.dtype().is_float64())
            {
                dset->AddField(detail::GetField<float64>(n_vals, field_name, assoc_str, topo_name, zero_copy));
                supported_type = true;
            }
        }

        // vtk-m cant support zero copy for this layout or was not compiled to expose this datatype
        // use float64 by default
        if(!supported_type)
        {
            // convert to float64, we use this as a comprise to cover the widest range
            vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
            vtkm_arr.Allocate(num_vals);

            void *ptr = (void*) vtkh::GetVTKMPointer(vtkm_arr);
            Node n_tmp;
            n_tmp.set_external(DataType::float64(num_vals),ptr);
            n_vals.to_float64_array(n_tmp);

            // add field to dataset
            if(assoc_str == "vertex")
            {
                dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                                 vtkm::cont::Field::Association::POINTS,
                                                 vtkm_arr));
            }
            else if( assoc_str == "element")
            {
                dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                                 vtkm::cont::Field::Association::CELL_SET,
                                                 vtkm_arr));
            }
        }
    }
    catch (vtkm::cont::Error error)
    {
        ASCENT_ERROR("VTKm exception:" << error.GetMessage());
    }

}

void
VTKHDataAdapter::AddVectorField(const std::string &field_name,
                                const Node &n_field,
                                const std::string &topo_name,
                                int neles,
                                int nverts,
                                vtkm::cont::DataSet *dset,
                                const int dims,
                                bool zero_copy)                 // attempt to zero copy
{
    string assoc_str = n_field["association"].as_string();

    vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::ANY;
    if(assoc_str == "vertex")
    {
      vtkm_assoc = vtkm::cont::Field::Association::POINTS;
    }
    else if(assoc_str == "element")
    {
      vtkm_assoc = vtkm::cont::Field::Association::CELL_SET;
    }
    else
    {
      ASCENT_INFO("VTKm conversion does not support field assoc "<<assoc_str<<". Skipping");
      return;
    }


    const Node &n_vals = n_field["values"];
    int num_vals = n_vals.child(0).dtype().number_of_elements();
    int num_components = n_field["values"].number_of_children();

    const conduit::Node &u = n_field["values"].child(0);
    bool interleaved = conduit::blueprint::mcarray::is_interleaved(n_vals);
    try
    {
        bool supported_type = false;

        if(interleaved)
        {
            if(dims == 3)
            {
              // we compile vtk-h with fp types
              if(u.dtype().is_float32())
              {

                using Vec3f32 = vtkm::Vec<vtkm::Float32,3>;
                const Vec3f32 *vec_ptr = reinterpret_cast<const Vec3f32*>(u.as_float32_ptr());

                dset->AddField(detail::GetVectorField(vec_ptr,
                                                      num_vals,
                                                      field_name,
                                                      assoc_str,
                                                      topo_name,
                                                      zero_copy));
                supported_type = true;
              }
              else if(u.dtype().is_float64())
              {

                using Vec3f64 = vtkm::Vec<vtkm::Float64,3>;
                const Vec3f64 *vec_ptr = reinterpret_cast<const Vec3f64*>(u.as_float64_ptr());

                dset->AddField(detail::GetVectorField(vec_ptr,
                                                      num_vals,
                                                      field_name,
                                                      assoc_str,
                                                      topo_name,
                                                      zero_copy));
                supported_type = true;
              }
            }
            else if(dims == 2)
            {
              // we compile vtk-h with fp types
              if(u.dtype().is_float32())
              {

                using Vec2f32 = vtkm::Vec<vtkm::Float32,2>;
                const Vec2f32 *vec_ptr = reinterpret_cast<const Vec2f32*>(u.as_float32_ptr());

                dset->AddField(detail::GetVectorField(vec_ptr,
                                                      num_vals,
                                                      field_name,
                                                      assoc_str,
                                                      topo_name,
                                                      zero_copy));
                supported_type = true;
              }
              else if(u.dtype().is_float64())
              {

                using Vec2f64 = vtkm::Vec<vtkm::Float64,2>;
                const Vec2f64 *vec_ptr = reinterpret_cast<const Vec2f64*>(u.as_float64_ptr());

                dset->AddField(detail::GetVectorField(vec_ptr,
                                                      num_vals,
                                                      field_name,
                                                      assoc_str,
                                                      topo_name,
                                                      zero_copy));
                supported_type = true;
              }
            }
            else
            {
              ASCENT_ERROR("Vector unsupported dims " << dims);
            }
        }
        else
        {
          // we have a vector with 2/3 separate arrays
          // While vtkm supports ArrayHandleCompositeVectors for
          // coordinate systems, it does not support composites
          // for fields. Thus we have to copy the data.
          if(dims == 3)
          {
            const conduit::Node &v = n_field["values"].child(1);
            const conduit::Node &w = n_field["values"].child(2);

            if(u.dtype().is_float32())
            {
              detail::ExtractVector<float32>(dset,
                                             u,
                                             v,
                                             w,
                                             num_vals,
                                             dims,
                                             field_name,
                                             assoc_str,
                                             topo_name,
                                             zero_copy);
            }
            else if(u.dtype().is_float64())
            {
              detail::ExtractVector<float64>(dset,
                                             u,
                                             v,
                                             w,
                                             num_vals,
                                             dims,
                                             field_name,
                                             assoc_str,
                                             topo_name,
                                             zero_copy);
            }
          }
          else if(dims == 2)
          {
            const conduit::Node &v = n_field["values"].child(1);
            conduit::Node fake_w;
            if(u.dtype().is_float32())
            {
              detail::ExtractVector<float32>(dset,
                                             u,
                                             v,
                                             fake_w,
                                             num_vals,
                                             dims,
                                             field_name,
                                             assoc_str,
                                             topo_name,
                                             zero_copy);
            }
            else if(u.dtype().is_float64())
            {
              detail::ExtractVector<float64>(dset,
                                             u,
                                             v,
                                             fake_w,
                                             num_vals,
                                             dims,
                                             field_name,
                                             assoc_str,
                                             topo_name,
                                             zero_copy);
            }
          }
          else
          {
            ASCENT_ERROR("Vector unsupported dims " << dims);
          }
        }
    }
    catch (vtkm::cont::Error error)
    {
        ASCENT_ERROR("VTKm exception:" << error.GetMessage());
    }

}

std::string
GetBlueprintCellName(vtkm::UInt8 shape_id)
{
  std::string name;
  if(shape_id == vtkm::CELL_SHAPE_TRIANGLE)
  {
    name = "tri";
  }
  else if(shape_id == vtkm::CELL_SHAPE_VERTEX)
  {
    name = "point";
  }
  else if(shape_id == vtkm::CELL_SHAPE_LINE)
  {
    name = "line";
  }
  else if(shape_id == vtkm::CELL_SHAPE_POLYGON)
  {
    ASCENT_ERROR("Polygon is not supported in blueprint");
  }
  else if(shape_id == vtkm::CELL_SHAPE_QUAD)
  {
    name = "quad";
  }
  else if(shape_id == vtkm::CELL_SHAPE_TETRA)
  {
    name = "tet";
  }
  else if(shape_id == vtkm::CELL_SHAPE_HEXAHEDRON)
  {
    name = "hex";
  }
  else if(shape_id == vtkm::CELL_SHAPE_WEDGE)
  {
    ASCENT_ERROR("Wedge is not supported in blueprint");
  }
  else if(shape_id == vtkm::CELL_SHAPE_PYRAMID)
  {
    ASCENT_ERROR("Pyramid is not supported in blueprint");
  }
  return name;
}

bool
VTKHDataAdapter::VTKmTopologyToBlueprint(conduit::Node &output,
                                         const vtkm::cont::DataSet &data_set,
                                         const std::string topo_name,
                                         bool zero_copy)
{

  int topo_dims;
  bool is_structured = vtkh::VTKMDataSetInfo::IsStructured(data_set, topo_dims);
  bool is_uniform = vtkh::VTKMDataSetInfo::IsUniform(data_set);
  bool is_rectilinear = vtkh::VTKMDataSetInfo::IsRectilinear(data_set);
  vtkm::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
  const std::string coords_name = coords.GetName();
  // we cannot access an empty domain
  bool is_empty = false;

  if(data_set.GetCoordinateSystem().GetData().GetNumberOfValues() == 0 ||
     data_set.GetCellSet().GetNumberOfCells() == 0)
  {
    is_empty = true;
  }

  if(is_empty)
  {
    return is_empty;
  }

  if(is_uniform)
  {
    auto points = coords.GetData().AsArrayHandle<vtkm::cont::ArrayHandleUniformPointCoordinates>();
    auto portal = points.ReadPortal();

    auto origin = portal.GetOrigin();
    auto spacing = portal.GetSpacing();
    auto dims = portal.GetDimensions();

    output["topologies/"+topo_name+"/coordset"] = coords_name;
    output["topologies/"+topo_name+"/type"] = "uniform";

    output["coordsets/"+coords_name+"/type"] = "uniform";
    output["coordsets/"+coords_name+"/dims/i"] = (int) dims[0];
    output["coordsets/"+coords_name+"/dims/j"] = (int) dims[1];
    output["coordsets/"+coords_name+"/dims/k"] = (int) dims[2];
    output["coordsets/"+coords_name+"/origin/x"] = (double) origin[0];
    output["coordsets/"+coords_name+"/origin/y"] = (double) origin[1];
    output["coordsets/"+coords_name+"/origin/z"] = (double) origin[2];
    output["coordsets/"+coords_name+"/spacing/dx"] = (double) spacing[0];
    output["coordsets/"+coords_name+"/spacing/dy"] = (double) spacing[1];
    output["coordsets/"+coords_name+"/spacing/dz"] = (double) spacing[2];
  }
  else if(is_rectilinear)
  {
    typedef vtkm::cont::ArrayHandleCartesianProduct<vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                                    vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                                    vtkm::cont::ArrayHandle<vtkm::FloatDefault>> Cartesian;

    const auto points = coords.GetData().AsArrayHandle<Cartesian>();
    auto portal = points.ReadPortal();
    auto x_portal = portal.GetFirstPortal();
    auto y_portal = portal.GetSecondPortal();
    auto z_portal = portal.GetThirdPortal();

    // work around for conduit not accepting const pointers
    vtkm::FloatDefault *x_ptr = const_cast<vtkm::FloatDefault*>(x_portal.GetArray());
    vtkm::FloatDefault *y_ptr = const_cast<vtkm::FloatDefault*>(y_portal.GetArray());
    vtkm::FloatDefault *z_ptr = const_cast<vtkm::FloatDefault*>(z_portal.GetArray());

    output["topologies/"+topo_name+"/coordset"] = coords_name;
    output["topologies/"+topo_name+"/type"] = "rectilinear";

    output["coordsets/"+coords_name+"/type"] = "rectilinear";
    if(zero_copy)
    {
      output["coordsets/"+coords_name+"/values/x"].set_external(x_ptr, x_portal.GetNumberOfValues());
      output["coordsets/"+coords_name+"/values/y"].set_external(y_ptr, y_portal.GetNumberOfValues());
      output["coordsets/"+coords_name+"/values/z"].set_external(z_ptr, z_portal.GetNumberOfValues());
    }
    else
    {
      output["coordsets/"+coords_name+"/values/x"].set(x_ptr, x_portal.GetNumberOfValues());
      output["coordsets/"+coords_name+"/values/y"].set(y_ptr, y_portal.GetNumberOfValues());
      output["coordsets/"+coords_name+"/values/z"].set(z_ptr, z_portal.GetNumberOfValues());
    }
  }
  else
  {
    int point_dims[3];
    //
    // This still could be structured, but this will always
    // have an explicit coordinate system
    output["coordsets/"+coords_name+"/type"] = "explicit";
    using Coords32 = vtkm::cont::ArrayHandleSOA<vtkm::Vec<vtkm::Float32, 3>>;
    using Coords64 = vtkm::cont::ArrayHandleSOA<vtkm::Vec<vtkm::Float64, 3>>;

    using CoordsVec32 = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>;
    using CoordsVec64 = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>;

    vtkm::cont::VariantArrayHandle coordsHandle(coords.GetData());

    if(coordsHandle.IsType<Coords32>())
    {
      Coords32 points = coordsHandle.Cast<Coords32>();

      auto x_handle = points.GetArray(0);
      auto y_handle = points.GetArray(1);
      auto z_handle = points.GetArray(2);

      point_dims[0] = x_handle.GetNumberOfValues();
      point_dims[1] = y_handle.GetNumberOfValues();
      point_dims[2] = z_handle.GetNumberOfValues();

      if(zero_copy)
      {
        output["coordsets/"+coords_name+"/values/x"].
          set_external(vtkh::GetVTKMPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set_external(vtkh::GetVTKMPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set_external(vtkh::GetVTKMPointer(z_handle), point_dims[2]);
      }
      else
      {
        output["coordsets/"+coords_name+"/values/x"].
          set(vtkh::GetVTKMPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set(vtkh::GetVTKMPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set(vtkh::GetVTKMPointer(z_handle), point_dims[2]);

      }

    }
    else if(coordsHandle.IsType<CoordsVec32>())
    {
      CoordsVec32 points = coordsHandle.Cast<CoordsVec32>();

      const int num_vals = points.GetNumberOfValues();
      vtkm::Float32 *points_ptr = (vtkm::Float32*)vtkh::GetVTKMPointer(points);
      const int byte_size = sizeof(vtkm::Float32);

      if(zero_copy)
      {
        output["coordsets/"+coords_name+"/values/x"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*0,  // byte offset
                                                                  byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/y"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*1,  // byte offset
                                                                  sizeof(vtkm::Float32)*3); // stride
        output["coordsets/"+coords_name+"/values/z"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*2,  // byte offset
                                                                  byte_size*3); // stride
      }
      else
      {
        output["coordsets/"+coords_name+"/values/x"].set(points_ptr,
                                                         num_vals,
                                                         byte_size*0,  // byte offset
                                                         byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/y"].set(points_ptr,
                                                         num_vals,
                                                         byte_size*1,  // byte offset
                                                         sizeof(vtkm::Float32)*3); // stride
        output["coordsets/"+coords_name+"/values/z"].set(points_ptr,
                                                         num_vals,
                                                         byte_size*2,  // byte offset
                                                         byte_size*3); // stride

      }

    }
    else if(vtkm::cont::IsType<Coords64>(coordsHandle))
    {
      Coords64 points = coordsHandle.Cast<Coords64>();

<<<<<<< HEAD
      auto x_handle = vtkm::get<0>(points.GetArrayTuple());
      auto y_handle = vtkm::get<1>(points.GetArrayTuple());
      auto z_handle = vtkm::get<2>(points.GetArrayTuple());
=======
      auto x_handle = points.GetArray(0);
      auto y_handle = points.GetArray(1);
      auto z_handle = points.GetArray(2);
>>>>>>> a9cfa757646d576189c06353db935ebb9c25812a

      point_dims[0] = x_handle.GetNumberOfValues();
      point_dims[1] = y_handle.GetNumberOfValues();
      point_dims[2] = z_handle.GetNumberOfValues();
      if(zero_copy)
      {
        output["coordsets/"+coords_name+"/values/x"].
          set_external(vtkh::GetVTKMPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set_external(vtkh::GetVTKMPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set_external(vtkh::GetVTKMPointer(z_handle), point_dims[2]);
      }
      else
      {
        output["coordsets/"+coords_name+"/values/x"].
          set(vtkh::GetVTKMPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set(vtkh::GetVTKMPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set(vtkh::GetVTKMPointer(z_handle), point_dims[2]);

      }
    }
    else if(coordsHandle.IsType<CoordsVec64>())
    {
     CoordsVec64 points = vtkm::cont::Cast<CoordsVec64>(coordsHandle);

      const int num_vals = points.GetNumberOfValues();
      vtkm::Float64 *points_ptr = (vtkm::Float64*)vtkh::GetVTKMPointer(points);
      const int byte_size = sizeof(vtkm::Float64);

      if(zero_copy)
      {
        output["coordsets/"+coords_name+"/values/x"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*0,  // byte offset
                                                                  byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/y"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*1,  // byte offset
                                                                  byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/z"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*2,  // byte offset
                                                                  byte_size*3); // stride
      }
      else
      {
        output["coordsets/"+coords_name+"/values/x"].set(points_ptr,
                                                         num_vals,
                                                         byte_size*0,  // byte offset
                                                         byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/y"].set(points_ptr,
                                                         num_vals,
                                                         byte_size*1,  // byte offset
                                                         byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/z"].set(points_ptr,
                                                         num_vals,
                                                         byte_size*2,  // byte offset
                                                         byte_size*3); // stride

      }

    }
    else
    {
      // Ok vtkm has handed us something we don't know about, and its really
      // hard to ask vtkm to tell us what it is. Before we give up, we will
      // attempt to copy the data to a known type and copy that copy.
      // We can't avoid the double copy since conduit can't take ownership
      // and we can't seem to write to a zero copied array

      vtkm::cont::ArrayHandle<vtkm::Vec<double,3>> coords_copy;
      vtkm::cont::ArrayCopy(coordsHandle, coords_copy);
      const int num_vals = coords_copy.GetNumberOfValues();
      vtkm::Float64 *points_ptr = (vtkm::Float64*)vtkh::GetVTKMPointer(coords_copy);
      const int byte_size = sizeof(vtkm::Float64);


      output["coordsets/"+coords_name+"/values/x"].set(points_ptr,
                                                       num_vals,
                                                       byte_size*0,  // byte offset
                                                       byte_size*3); // stride
      output["coordsets/"+coords_name+"/values/y"].set(points_ptr,
                                                       num_vals,
                                                       byte_size*1,  // byte offset
                                                       byte_size*3); // stride
      output["coordsets/"+coords_name+"/values/z"].set(points_ptr,
                                                       num_vals,
                                                       byte_size*2,  // byte offset
                                                       byte_size*3); // stride
    }

    vtkm::UInt8 shape_id = 0;
    if(is_structured)
    {
      output["topologies/"+topo_name+"/coordset"] = coords_name;
      output["topologies/"+topo_name+"/type"] = "structured";

      vtkm::cont::DynamicCellSet dyn_cells = data_set.GetCellSet();
      using Structured2D = vtkm::cont::CellSetStructured<2>;
      using Structured3D = vtkm::cont::CellSetStructured<3>;
      if(dyn_cells.IsSameType(Structured2D()))
      {
        Structured2D cells = dyn_cells.Cast<Structured2D>();
        vtkm::Id2 cell_dims = cells.GetCellDimensions();
        output["topologies/"+topo_name+"/elements/dims/i"] = (int) cell_dims[0];
        output["topologies/"+topo_name+"/elements/dims/j"] = (int) cell_dims[1];
      }
      else if(dyn_cells.IsSameType(Structured3D()))
      {
        Structured3D cells = dyn_cells.Cast<Structured3D>();
        vtkm::Id3 cell_dims = cells.GetCellDimensions();
        output["topologies/"+topo_name+"/elements/dims/i"] = (int) cell_dims[0];
        output["topologies/"+topo_name+"/elements/dims/j"] = (int) cell_dims[1];
        output["topologies/"+topo_name+"/elements/dims/k"] = (int) cell_dims[2];
      }
      else
      {
        ASCENT_ERROR("Unknown structured cell set");
      }

    }
    else
    {
      output["topologies/"+topo_name+"/coordset"] = coords_name;
      output["topologies/"+topo_name+"/type"] = "unstructured";
      vtkm::cont::DynamicCellSet dyn_cells = data_set.GetCellSet();

      using SingleType = vtkm::cont::CellSetSingleType<>;
      using MixedType = vtkm::cont::CellSetExplicit<>;

      if(dyn_cells.IsSameType(SingleType()))
      {
        SingleType cells = dyn_cells.Cast<SingleType>();
        vtkm::UInt8 shape_id = cells.GetCellShape(0);
        std::string conduit_name = GetBlueprintCellName(shape_id);
        output["topologies/"+topo_name+"/elements/shape"] = conduit_name;

        std::cerr<<"BAD TYPE!!! "<<__FILE__<<" "<<__LINE__<<std::endl;
        /*
        static_assert(sizeof(vtkm::Id) == sizeof(int), "blueprint expects connectivity to be ints");
        */
        auto conn = cells.GetConnectivityArray(vtkm::TopologyElementTagCell(),
                                               vtkm::TopologyElementTagPoint());

        if(zero_copy)
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set_external(vtkh::GetVTKMPointer(conn), conn.GetNumberOfValues());
        }
        else
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set(vtkh::GetVTKMPointer(conn), conn.GetNumberOfValues());
        }
      }
      else if(vtkh::VTKMDataSetInfo::IsSingleCellShape(dyn_cells, shape_id))
      {
        // If we are here, the we know that the cell set is explicit,
        // but only a single cell shape
        auto cells = dyn_cells.Cast<vtkm::cont::CellSetExplicit<>>();
        auto shapes = cells.GetShapesArray(vtkm::TopologyElementTagCell(),
                                           vtkm::TopologyElementTagPoint());

        std::string conduit_name = GetBlueprintCellName(shape_id);
        output["topologies/"+topo_name+"/elements/shape"] = conduit_name;

        std::cerr<<"BAD TYPE!!! "<<__FILE__<<" "<<__LINE__<<std::endl;
        /*
        static_assert(sizeof(vtkm::Id) == sizeof(int), "blueprint expects connectivity to be ints");
        */

        auto conn = cells.GetConnectivityArray(vtkm::TopologyElementTagCell(),
                                               vtkm::TopologyElementTagPoint());

        if(zero_copy)
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set_external(vtkh::GetVTKMPointer(conn), conn.GetNumberOfValues());
        }
        else
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set(vtkh::GetVTKMPointer(conn), conn.GetNumberOfValues());
        }

      }
      else
      {
        ASCENT_ERROR("Mixed explicit types not implemented");
        data_set.PrintSummary(std::cout);
        MixedType cells = dyn_cells.Cast<MixedType>();
      }

    }
  }
  return is_empty;
}

template<typename T, int N>
void ConvertVecToNode(conduit::Node &output,
                      std::string path,
                      vtkm::cont::ArrayHandle<vtkm::Vec<T,N>> &handle,
                      bool zero_copy)
{
  static_assert(N > 1 && N < 4, "Vecs must be size 2 or 3");
  output[path + "/type"] = "vector";
  if(zero_copy)
  {
    output[path + "/values/u"].set_external((T*) vtkh::GetVTKMPointer(handle),
                                            handle.GetNumberOfValues(),
                                            sizeof(T)*0,   // starting offset in bytes
                                            sizeof(T)*N);  // stride in bytes
    output[path + "/values/v"].set_external((T*) vtkh::GetVTKMPointer(handle),
                                            handle.GetNumberOfValues(),
                                            sizeof(T)*1,   // starting offset in bytes
                                            sizeof(T)*N);  // stride in bytes
  }
  else
  {
    output[path + "/values/u"].set((T*) vtkh::GetVTKMPointer(handle),
                                   handle.GetNumberOfValues(),
                                   sizeof(T)*0,   // starting offset in bytes
                                   sizeof(T)*N);  // stride in bytes
    output[path + "/values/v"].set((T*) vtkh::GetVTKMPointer(handle),
                                   handle.GetNumberOfValues(),
                                   sizeof(T)*1,   // starting offset in bytes
                                   sizeof(T)*N);  // stride in bytes
  }
  if(N == 3)
  {

    if(zero_copy)
    {
      output[path + "/values/w"].set_external((T*) vtkh::GetVTKMPointer(handle),
                                              handle.GetNumberOfValues(),
                                              sizeof(T)*2,   // starting offset in bytes
                                              sizeof(T)*N);  // stride in bytes
    }
    else
    {
      output[path + "/values/w"].set((T*) vtkh::GetVTKMPointer(handle),
                                     handle.GetNumberOfValues(),
                                     sizeof(T)*2,   // starting offset in bytes
                                     sizeof(T)*N);  // stride in bytes
    }
  }
}

void
VTKHDataAdapter::VTKmFieldToBlueprint(conduit::Node &output,
                                      const vtkm::cont::Field &field,
                                      const std::string topo_name,
                                      bool zero_copy)
{
  std::string name = field.GetName();
  std::string path = "fields/" + name;
  bool assoc_points = vtkm::cont::Field::Association::POINTS == field.GetAssociation();
  bool assoc_cells  = vtkm::cont::Field::Association::CELL_SET == field.GetAssociation();
  //bool assoc_mesh  = vtkm::cont::Field::ASSOC_WHOLE_MESH == field.GetAssociation();
  if(!assoc_points && ! assoc_cells)
  {
    ASCENT_ERROR("Field must be associtated with cells or points\n");
  }
  std::string conduit_name;

  if(assoc_points) conduit_name = "vertex";
  else conduit_name = "element";

  output[path + "/association"] = conduit_name;
  output[path + "/topology"] = topo_name;

  vtkm::cont::VariantArrayHandle dyn_handle = field.GetData();
  //
  // this can be literally anything. Lets do some exhaustive casting
  //
  if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Float32>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Float32>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    if(zero_copy)
    {
      output[path + "/values"].
        set_external(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
    else
    {
      output[path + "/values"].  set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Float64>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Float64>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    if(zero_copy)
    {
      output[path + "/values"].
        set_external(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
    else
    {
      output[path + "/values"].  set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Int8>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Int8>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    if(zero_copy)
    {
      output[path + "/values"].
        set_external(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
    else
    {
      output[path + "/values"].  set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Int32>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Int32>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    if(zero_copy)
    {
      output[path + "/values"].
        set_external(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
    else
    {
      output[path + "/values"].  set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Int64>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Int64>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ASCENT_ERROR("Conduit int64 and vtkm::Int64 are different. Cannot convert vtkm::Int64\n");
    //output[path + "/values"].set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::UInt32>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::UInt32>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    if(zero_copy)
    {
      output[path + "/values"].
        set_external(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
    else
    {
      output[path + "/values"].  set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::UInt8>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::UInt8>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    if(zero_copy)
    {
      output[path + "/values"].
        set_external(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
    else
    {
      output[path + "/values"].  set(vtkh::GetVTKMPointer(handle), handle.GetNumberOfValues());
    }
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle, zero_copy);
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle, zero_copy);
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,3>>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,3>>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle, zero_copy);
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle, zero_copy);
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle, zero_copy);
  }
  else if(dyn_handle.IsType<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,2>>>())
  {
    using HandleType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Int32,2>>;
    HandleType handle = dyn_handle.Cast<HandleType>();
    ConvertVecToNode(output, path, handle, zero_copy);
  }
  else
  {
    field.PrintSummary(std::cerr);
    ASCENT_ERROR("Field type unsupported for conversion");
  }
}

void VTKHDataAdapter::VTKHCollectionToBlueprintDataSet(VTKHCollection *collection,
                                                       conduit::Node &node,
                                                       bool zero_copy)
{
  node.reset();

<<<<<<< HEAD
  const int cycle = collection->cycle();
  const double time = collection->time();
=======
  bool success = true;
>>>>>>> a9cfa757646d576189c06353db935ebb9c25812a
  // we have to re-merge the domains so all domains with the same
  // domain id end up in a single domain
  std::map<int, std::map<std::string,vtkm::cont::DataSet>> domain_map;
  domain_map = collection->by_domain_id();

  for(auto domain_it : domain_map)
  {
    const int domain_id = domain_it.first;

<<<<<<< HEAD
    conduit::Node &dom = node.append();
    dom["state/domain_id"] = (int) domain_id;
    dom["state/cycle"] = cycle;
    dom["state/time"] = time;
=======
      conduit::Node &dom = node.append();
      dom["state/domain_id"] = (int) domain_id;
>>>>>>> a9cfa757646d576189c06353db935ebb9c25812a

    for(auto topo_it : domain_it.second)
    {
      const std::string topo_name = topo_it.first;
      vtkm::cont::DataSet &dataset = topo_it.second;
      VTKHDataAdapter::VTKmToBlueprintDataSet(&dataset, dom, topo_name, zero_copy);
    }
  }

}

void
VTKHDataAdapter::VTKHToBlueprintDataSet(vtkh::DataSet *dset,
                                        conduit::Node &node,
                                        bool zero_copy)
{
  node.reset();
  const int num_doms = dset->GetNumberOfDomains();
  for(int i = 0; i < num_doms; ++i)
  {
    conduit::Node &dom = node.append();
    vtkm::cont::DataSet vtkm_dom;
    vtkm::Id domain_id;
    int cycle = dset->GetCycle();
    dset->GetDomain(i, vtkm_dom, domain_id);
    VTKHDataAdapter::VTKmToBlueprintDataSet(&vtkm_dom,dom, "topo", zero_copy);
    dom["state/domain_id"] = (int) domain_id;
    dom["state/cycle"] = cycle;
  }
}

void
VTKHDataAdapter::VTKmToBlueprintDataSet(const vtkm::cont::DataSet *dset,
                                        conduit::Node &node,
                                        const std::string topo_name,
                                        bool zero_copy)
{
  //
  // with vtkm, we have no idea what the type is of anything inside
  // dataset, so we have to ask all fields, cell sets anc coordinate systems.
  //

  bool is_empty = VTKmTopologyToBlueprint(node, *dset, topo_name, zero_copy);

  if(!is_empty)
  {
    const vtkm::Id num_fields = dset->GetNumberOfFields();
    for(vtkm::Id i = 0; i < num_fields; ++i)
    {
      vtkm::cont::Field field = dset->GetField(i);
      VTKmFieldToBlueprint(node, field, topo_name, zero_copy);
    }
  }
}


};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
