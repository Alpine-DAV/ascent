//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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

#include <ascent_logging.hpp>
#include <ascent_logging_old.hpp>

// VTKm includes
#define VTKM_USE_DOUBLE_PRECISION
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleExtractComponent.h>
#include <vtkm/cont/CoordinateSystem.h>
#include <vtkm/cont/Invoker.h>
#include <vtkh/DataSet.hpp>

// other ascent includes
#include <ascent_logging.hpp>
#include <ascent_block_timer.hpp>
#include <ascent_mpi_utils.hpp>
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
void
BlueprintIndexArrayToVTKmIdArray(const conduit::Node &n,
                                 bool zero_copy,
                                 vtkm::cont::ArrayHandle<T> &vtkm_handle)
{
    int array_size = n.dtype().number_of_elements();

    if( sizeof(T) == 1 ) // uint8 is what vtk-m will use for this case.
    {
        if(n.is_compact() && n.dtype().is_uint8())
        {
            // directly compatible
            const void *idx_ptr = n.data_ptr();
            CopyArray(vtkm_handle, (const T*)idx_ptr, array_size,zero_copy);
        }
        else
        {
            // we need to convert to uint8 to match vtkm::Id
            vtkm_handle.Allocate(array_size);
            void *ptr = (void*) vtkh::GetVTKMPointer(vtkm_handle);
            Node n_tmp;
            n_tmp.set_external(DataType::uint8(array_size),ptr);
            n.to_uint8_array(n_tmp);
        }
    }
    else if( sizeof(T) == 2)
    {
        // unsupported!
        ASCENT_ERROR("BlueprintIndexArrayToVTKmIdArray does not support 2-byte index arrays");
    }
    else if( sizeof(T) == 4) // int32 is what vtk-m will use for this case.
    {
        if(n.is_compact() && n.dtype().is_int32())
        {
            // directly compatible
            const void *idx_ptr = n.data_ptr();
            CopyArray(vtkm_handle, (const T*)idx_ptr, array_size,zero_copy);
        }
        else
        {
            // we need to convert to int32 to match vtkm::Id
            vtkm_handle.Allocate(array_size);
            void *ptr = (void*) vtkh::GetVTKMPointer(vtkm_handle);
            Node n_tmp;
            n_tmp.set_external(DataType::int32(array_size),ptr);
            n.to_int32_array(n_tmp);
        }
    }
    else if( sizeof(T) == 8) // int64 is what vtk-m will use for this case.
    {
        if(n.is_compact() && n.dtype().is_int64())
        {
            // directly compatible
            const void *idx_ptr = n.data_ptr();
            CopyArray(vtkm_handle, (const T*)idx_ptr, array_size, zero_copy);
        }
        else
        {
            // we need to convert to int64 to match vtkm::Id
            vtkm_handle.Allocate(array_size);
            void *ptr = (void*) vtkh::GetVTKMPointer(vtkm_handle);
            Node n_tmp;
            n_tmp.set_external(DataType::int64(array_size),ptr);
            n.to_int64_array(n_tmp);
        }
    }
}


template<typename T>
vtkm::cont::CoordinateSystem
GetExplicitCoordinateSystem(const conduit::Node &n_coords,
                            const std::string &name,
                            int &ndims,
                            index_t &x_element_stride,
                            index_t &y_element_stride,
                            index_t &z_element_stride,
                            bool zero_copy)
{
    vtkm::CopyFlag copy = vtkm::CopyFlag::On;
    if(zero_copy)
    {
      copy = vtkm::CopyFlag::Off;
    }
      
    int nverts = n_coords["values/x"].dtype().number_of_elements();
    //bool is_interleaved = blueprint::mcarray::is_interleaved(n_coords["values"]);

    // some interleaved cases aren't working
    // disabling this path until we find out what is going wrong.
    //is_interleaved = false;

    vtkm::cont::ArrayHandle<T> x_coords_handle;
    vtkm::cont::ArrayHandle<T> y_coords_handle;
    vtkm::cont::ArrayHandle<T> z_coords_handle;

    ndims = 2;

    if(x_element_stride == 1)
    {
      const T *x_verts_ptr = n_coords["values/x"].value();
      detail::CopyArray(x_coords_handle, x_verts_ptr, nverts, zero_copy);
    }
    else
    {
      int x_verts_expanded = nverts * x_element_stride;
      const T *x_verts_ptr = n_coords["values/x"].value();
      vtkm::cont::ArrayHandle<T> x_source_array = vtkm::cont::make_ArrayHandle<T>(x_verts_ptr,
                                                                                  x_verts_expanded,
                                                                                  copy);
      vtkm::cont::ArrayHandleStride<T> x_stride_handle(x_source_array,
                                                       nverts,
                                                       x_element_stride,
                                                       0); // offset

      vtkm::cont::Algorithm::Copy(x_stride_handle, x_coords_handle);
    }

    if(y_element_stride == 1)
    {
      const T *y_verts_ptr = n_coords["values/y"].value();
      detail::CopyArray(y_coords_handle, y_verts_ptr, nverts, zero_copy);
    }
    else
    {
      int y_verts_expanded = nverts * y_element_stride;
      const T *y_verts_ptr = n_coords["values/y"].value();
      vtkm::cont::ArrayHandle<T> y_source_array = vtkm::cont::make_ArrayHandle<T>(y_verts_ptr,
                                                                                  y_verts_expanded,
                                                                                  copy);
      vtkm::cont::ArrayHandleStride<T> y_stride_handle(y_source_array,
                                                       nverts,
                                                       y_element_stride,
                                                       0); // offset

      vtkm::cont::Algorithm::Copy(y_stride_handle, y_coords_handle);
    }

    if(z_element_stride == 0)
    {
      z_coords_handle.Allocate(nverts);
      // TODO: Set on device?
      // This does not get initialized to zero
      T *z = vtkh::GetVTKMPointer(z_coords_handle);
      memset(z, 0, nverts * sizeof(T));
    }
    else if(z_element_stride == 1)
    {
      ndims = 3;
      const T *z_verts_ptr = n_coords["values/z"].value();
      detail::CopyArray(z_coords_handle, z_verts_ptr, nverts, zero_copy);
    }
    else
    {
      ndims = 3;
      int z_verts_expanded = nverts * z_element_stride;
      const T *z_verts_ptr = n_coords["values/z"].value();
      vtkm::cont::ArrayHandle<T> z_source_array = vtkm::cont::make_ArrayHandle<T>(z_verts_ptr,
                                                                                  z_verts_expanded,
                                                                                  copy);
      vtkm::cont::ArrayHandleStride<T> z_stride_handle(z_source_array,
                                                       nverts,
                                                       z_element_stride,
                                                       0); // offset

      vtkm::cont::Algorithm::Copy(z_stride_handle, z_coords_handle);
    }

    return vtkm::cont::CoordinateSystem(name,
                                        make_ArrayHandleSOA(x_coords_handle,
                                                            y_coords_handle,
                                                            z_coords_handle));

}


template<typename T>
vtkm::cont::Field GetField(const conduit::Node &node,
                           const std::string &field_name,
                           const std::string &assoc_str,
                           const std::string &topo_str,
                           index_t element_stride,
                           bool zero_copy)
{
  vtkm::CopyFlag copy = vtkm::CopyFlag::On;
  if(zero_copy)
  {
    copy = vtkm::CopyFlag::Off;
  }
  vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    vtkm_assoc = vtkm::cont::Field::Association::Points;
  }
  else if(assoc_str == "element")
  {
    vtkm_assoc = vtkm::cont::Field::Association::Cells;
  }
  else if(assoc_str == "whole")
  {
    vtkm_assoc = vtkm::cont::Field::Association::WholeDataSet;
  }
  else
  {
    ASCENT_ERROR("Cannot add field association "<<assoc_str<<" from field "<<field_name);
  }

  int num_vals = node.dtype().number_of_elements();

  const T *values_ptr = node.value();

  vtkm::cont::Field field;
  // base case is naturally stride data
  if(element_stride == 1)
  {
      field = vtkm::cont::make_Field(field_name,
                                     vtkm_assoc,
                                     values_ptr,
                                     num_vals,
                                     copy);
  }
  else
  {
      //
      // use ArrayHandleStride to create new field
      //

      // NOTE: In this case, the num_vals, needs to be
      // the full extent of the strided area3

      int num_vals_expanded = num_vals * element_stride;
      vtkm::cont::ArrayHandle<T> source_array = vtkm::cont::make_ArrayHandle(values_ptr,
                                                                             num_vals_expanded,
                                                                             copy);
      vtkm::cont::ArrayHandleStride<T> stride_array(source_array,
                                                    num_vals,
                                                    element_stride,
                                                    0);
      field =  vtkm::cont::Field(field_name,
                                 vtkm_assoc,
                                 stride_array);
  }

  return field;
}


template<typename T>
vtkm::cont::Field GetVectorField(T *values_ptr,
                                 const int num_vals,
                                 const std::string &field_name,
                                 const std::string &assoc_str,
                                 const std::string &topo_str,
                                 bool zero_copy)
{
  vtkm::CopyFlag copy = vtkm::CopyFlag::On;
  if(zero_copy)
  {
    copy = vtkm::CopyFlag::Off;
  }
  vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    vtkm_assoc = vtkm::cont::Field::Association::Points;
  }
  else if(assoc_str == "element")
  {
    vtkm_assoc = vtkm::cont::Field::Association::Cells;
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
                   const std::string &field_name,
                   const std::string &assoc_str,
                   const std::string &topo_name,
                   bool zero_copy)
{
  // TODO: Do we need to fix this for striding?
  // GetField<T> expects compact
  if(dims != 2 && dims != 3)
  {
    ASCENT_ERROR("Extract vector: only 2 and 3 dims supported given "<<dims);
  }

  vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    vtkm_assoc = vtkm::cont::Field::Association::Points;
  }
  else if (assoc_str == "element")
  {
    vtkm_assoc = vtkm::cont::Field::Association::Cells;
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


void VTKmCellShape(const std::string &shape_type,
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
  else if(shape_type == "pyramid")
  {
      shape_id = 14;
      num_indices = 5;
  }
  else if(shape_type == "wedge")
  {
      shape_id = 13;
      num_indices = 6;
  }
  else
  {
    ASCENT_ERROR("Unsupported cell type "<<shape_type);
  }
}

template<typename T>
bool allEqual(std::vector<T> const &v) 
{
  return std::adjacent_find(v.begin(), v.end(), std::not_equal_to<T>()) == v.end();
}



template<typename T, typename S>
void GetMatSetFields(const conduit::Node &node, //materials["matset"]
                           const std::string &length_name,
                           const std::string &offsets_name,
                           const std::string &ids_name,
                           const std::string &vfs_name,
                           const std::string &topo_str,
                           const int neles,
                           vtkm::cont::Field &length,
                           vtkm::cont::Field &offsets,
                           vtkm::cont::Field &ids,
                           vtkm::cont::Field &vfs)
{
  vtkm::CopyFlag copy = vtkm::CopyFlag::On;

  vtkm::cont::Field::Association vtkm_assoc_c = vtkm::cont::Field::Association::Cells;

  std::vector<T> v_length(neles,0);
  std::vector<T> v_offsets(neles,0);
  if(node.has_child("element_ids"))
  {
    NodeConstIterator itr = node["element_ids"].children();
    std::string material_name;
    while(itr.has_next())
    {

      const Node &n_material = itr.next();
      const int nvals = n_material.dtype().number_of_elements();
      const T *data = n_material.value();
      //increase length when a material vf value > 0
      for(int i = 0; i < nvals; ++i)
      {
        v_length[data[i]] += 1;
      }
    }
  }
  else
  {
    NodeConstIterator itr = node["volume_fractions"].children();
    std::string material_name;
    while(itr.has_next())
    {

      const Node &n_material = itr.next();
      const S *data = n_material.value();
      //increase length when a material vf value > 0
      for(index_t i = 0; i < neles; ++i)
      {
        if(data[i] > 0)
          v_length[i] += 1;
      }
    }
  }

  //calc offset of length and total length
  int l_total = 0;
  for(index_t i = 0; i < neles-1; ++i)
  {
    v_offsets[i+1] = v_offsets[i] + v_length[i];
    l_total += v_length[i];
  }
  l_total += v_length[neles-1];

  const T *length_ptr = v_length.data();

  length = vtkm::cont::make_Field(length_name,
                                 vtkm_assoc_c,
                                 length_ptr,
                                 neles,
                                 copy);

  const T *offsets_ptr = v_offsets.data();

  offsets = vtkm::cont::make_Field(offsets_name,
                                 vtkm_assoc_c,
                                 offsets_ptr,
                                 neles,
                                 copy);
  //calc vfs and mat ids
  vtkm::cont::Field::Association vtkm_assoc_w = vtkm::cont::Field::Association::WholeDataSet;
  std::vector<T> v_ids(l_total,0);
  std::vector<S> v_vfs(l_total,0);

  if(node.has_child("element_ids"))
  {

    int num_materials = node["element_ids"].number_of_children();
    const Node &n_vol_fracs = node["volume_fractions"];
    const Node &n_ele_ids = node["element_ids"];

    for(index_t i = 0; i < num_materials; ++i)
    {
      const Node &n_vol_frac = n_vol_fracs.child(i);
      const Node &n_ele_id = n_ele_ids.child(i);
      const S *vf_data = n_vol_frac.value();
      const T *id_data = n_ele_id.value();
      int num_vals = n_ele_id.dtype().number_of_elements(); 

      for(index_t j = 0; j < num_vals; ++j)
      {
        v_length[id_data[j]] -= 1;
        index_t offset = v_offsets[id_data[j]];
        index_t length = v_length[id_data[j]];
        v_vfs[offset + length] = vf_data[j];
        v_ids[offset + length] = i+1; //material ids can't start at 0
      }
    }
  }
  else
  {
    int num_materials = node["volume_fractions"].number_of_children();
    for(index_t i = 0; i < num_materials; ++i)
    {
      const Node &n_materials = node["volume_fractions"];
      const Node &n_material = n_materials.child(i);
      const S *data = n_material.value();

      for(index_t j = 0; j < neles; ++j)
      {
        index_t offset = v_offsets[j];
        if(data[j] > 0)
        {
          v_length[j] -= 1;
          index_t length = v_length[j];
          v_ids[offset + length] = i + 1; //IDs cannot start at 0
          v_vfs[offset + length] = data[j];
        }
      }
    }
  }

  const T *ids_ptr = v_ids.data();

  ids = vtkm::cont::make_Field(ids_name,
                               vtkm_assoc_w,
                               ids_ptr,
                               l_total,
                               copy);

  const S *vfs_ptr = v_vfs.data();

  vfs = vtkm::cont::make_Field(vfs_name,
                               vtkm_assoc_w,
                               vfs_ptr,
                               l_total,
                               copy);
}

//template<typename T, typename S>
//void GetMatSetIDsAndVFs(const conduit::Node &node, //materials["matset"]
//                           const std::string &ids_name,
//                           const std::string &vfs_name,
//                           const std::string &topo_str,
//                           const int total,
//                           const int neles,
//                           vtkm::cont::Field &offsets,
//{
//  vtkm::CopyFlag copy = vtkm::CopyFlag::On;
//
//  vtkm::cont::ArrayHandle<int> ah_offsets;
//  offsets.GetData().AsArrayHandle(ah_offsets);
//  
//  int num_materials = node["volume_fractions"].number_of_children();
//  for(int i = 0; i < num_materials; ++i)
//  {
//    int offset = ah_offsets.ReadPortal().Get(j);
//    const Node &n_materials = node["volume_fractions"];
//    const Node &n_material = n_materials.child(i);
//    const S *data = n_material.value();
//
//    for(int j = 0; j < neles; ++j)
//    {
//      if(data[j] > 0)
//      {
//        v_ids[offset] = j + 1; //IDs cannot start at 0
//        v_vfs[offset] = data[j];
//        offset++;
//      }
//    }
//  }
//
//  const T *ids_ptr = v_ids.data();
//
//  ids = vtkm::cont::make_Field(ids_name,
//                               vtkm_assoc,
//                               ids_ptr,
//                               total,
//                               copy);
//
//  const S *vfs_ptr = v_vfs.data();
//
//  vfs = vtkm::cont::make_Field(vfs_name,
//                               vtkm_assoc,
//                               vfs_ptr,
//                               total,
//                               copy);
//
//}

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
//    if(num_domains == 0)
//      return nullptr;

    VTKHCollection *res = new VTKHCollection();
    std::map<std::string, vtkh::DataSet> datasets;
    vtkm::UInt64 cycle = 0;
    double time = 0;
    std::vector<vtkm::UInt64> allCycles;
    std::vector<double> allTimes;

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
	allCycles.push_back(cycle);
      }

      if(dom.has_path("state/time"))
      {
        time = dom["state/time"].to_float64();
	allTimes.push_back(time);
      }
      for(int t = 0; t < topo_names.size(); ++t)
      {
        const std::string topo_name = topo_names[t];
        vtkm::cont::DataSet *dset = BlueprintToVTKmDataSet(dom, zero_copy, topo_name);
        datasets[topo_name].AddDomain(*dset,domain_id);
        delete dset;
      }
    }

    //check to make sure there is data to grab
    if(num_domains > 0)
    {
      //time and cycle should be the same for all domains
      //if that's the case grab a topo and add the info
      const conduit::Node &dom = n.child(0);
      const std::vector<std::string> topo_names  = dom["topologies"].child_names();
      const std::string topo_name = topo_names[0];

      if(allCycles.size() != 0 && detail::allEqual(allCycles))
        datasets[topo_name].SetCycle(allCycles[0]);
      if(allTimes.size() != 0 && detail::allEqual(allTimes))
        datasets[topo_name].SetTime(allTimes[0]);
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
    else if( mesh_type ==  "points")
    {
        result =  PointsImplicitBlueprintToVTKmDataSet(coords_name,
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
        std::string field_name;
        while(itr.has_next())
        {

            const Node &n_field = itr.next();
            field_name = itr.name();
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

    if(node.has_child("matsets"))
    {
        // add all of the materials:
        NodeConstIterator itr = node["matsets"].children();
        std::string matset_name;
        while(itr.has_next())
        {
            const Node &n_matset = itr.next();
            matset_name = itr.name();
            if(n_matset["topology"].as_string() != topo_name)
            {
              // these are not the materials we are looking for
              continue;
            }
            AddMatSets(matset_name,
                     n_matset,
                     topo_name,
                     neles,
                     result,
                     zero_copy);

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
    else if(shape_type == "wedge")
    {
        shape_id = 13;
        indices = 6;
        dimensionality = 3;
    }
    else if(shape_type == "pyramid")
    {
        shape_id = 14;
        indices = 5;
        dimensionality = 3;
    }
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
#ifdef ASCENT_OPENMP_ENABLED
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

    string coords_type = n_coords["type"].as_string();
    nverts = n_coords["values/x"].dtype().number_of_elements();
    int ndims = 0;

    vtkm::cont::CoordinateSystem coords;
    if(n_coords["values/x"].dtype().is_float64())
    {
      index_t x_stride = n_coords["values/x"].dtype().stride();
      index_t x_element_stride = x_stride / sizeof(float64);
      index_t y_stride = n_coords["values/y"].dtype().stride();
      index_t y_element_stride = y_stride / sizeof(float64);
      index_t z_element_stride = 0;
      if(n_coords.has_path("values/z"))
      {
        index_t z_stride = n_coords["values/z"].dtype().stride();
        z_element_stride = z_stride / sizeof(float64);
      }

      coords = detail::GetExplicitCoordinateSystem<float64>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            x_element_stride,
                                                            y_element_stride,
                                                            z_element_stride,
                                                            zero_copy);
    }
    else if(n_coords["values/x"].dtype().is_float32())
    {
      index_t x_stride = n_coords["values/x"].dtype().stride();
      index_t x_element_stride = x_stride / sizeof(float32);
      index_t y_stride = n_coords["values/y"].dtype().stride();
      index_t y_element_stride = y_stride / sizeof(float32);
      index_t z_element_stride = 0;
      if(n_coords.has_path("values/z"))
      {
        index_t z_stride = n_coords["values/z"].dtype().stride();
        z_element_stride = z_stride / sizeof(float32);
      }

      coords = detail::GetExplicitCoordinateSystem<float32>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            x_element_stride,
                                                            y_element_stride,
                                                            z_element_stride,
                                                            zero_copy);
    }
    else
    {
      ASCENT_ERROR("Coordinate system must be floating point values");
    }

    result->AddCoordinateSystem(coords);

    int32 x_elems = n_topo["elements/dims/i"].to_int();
    int32 y_elems = n_topo["elements/dims/j"].to_int();

    vtkm::Id3 topo_origin = detail::topo_origin(n_topo);

    if(coords_type == "explicit")
    {
      if(ndims == 2)
      {
        vtkm::Id x_verts = x_elems + 1;
        vtkm::Id y_verts = y_elems + 1;
        neles = x_elems * y_elems;
        nverts = (x_verts) * (y_verts);
        
        std::string ele_shape = "quad";
        vtkm::UInt8 shape_id;
        vtkm::IdComponent indices_per;
        detail::VTKmCellShape(ele_shape, shape_id, indices_per);
        vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
        connectivity.Allocate(neles * indices_per);
        auto conn_portal = connectivity.WritePortal();
        int offset = 0;
        // Build Connectivity 
        for (vtkm::Id i = 0; i < x_elems; ++i) 
        {
          for (vtkm::Id j = 0; j < y_elems; ++j) 
          {
            vtkm::Id v0 = j * x_verts + i;
            vtkm::Id v1 = v0 + 1;
            vtkm::Id v2 = v0 + x_verts;
            vtkm::Id v3 = v0 + x_verts + 1;
            
            conn_portal.Set(offset, v0);// bottom left
            conn_portal.Set(offset+1, v1); //bottom right
            conn_portal.Set(offset+2, v3); //top right
            conn_portal.Set(offset+3, v2); //top left 
            offset = offset + 4;
          }
        }
        vtkm::cont::CellSetSingleType<> cell_set;
        cell_set.Fill(nverts, shape_id, indices_per, connectivity);
        neles = cell_set.GetNumberOfCells();
        result->SetCellSet(cell_set);
      }
      else
      {
        int32 z_elems = n_topo["elements/dims/k"].to_int();

        vtkm::Id x_verts = x_elems + 1;
        vtkm::Id y_verts = y_elems + 1;
        vtkm::Id z_verts = z_elems + 1;
        neles = x_elems * y_elems * z_elems;
        nverts = (x_verts) * (y_verts) * (z_verts);
        
        std::string ele_shape = "hex";
        vtkm::UInt8 shape_id;
        vtkm::IdComponent indices_per;
        detail::VTKmCellShape(ele_shape, shape_id, indices_per);
        vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
        connectivity.Allocate(neles * indices_per);
        auto conn_portal = connectivity.WritePortal();
        int offset = 0;
        // Build Connectivity (Polyhedral cells)
        for (vtkm::Id i = 0; i < x_elems; ++i) 
        {
          for (vtkm::Id j = 0; j < y_elems; ++j) 
          {
            for (vtkm::Id k = 0; k < z_elems; ++k) 
            {
              vtkm::Id v0 = k * y_verts * x_verts + j * x_verts + i;
              vtkm::Id v1 = v0 + 1;
              vtkm::Id v2 = v0 + x_verts;
              vtkm::Id v3 = v0 + x_verts + 1;
              vtkm::Id v4 = v0 + y_verts * x_verts;
              vtkm::Id v5 = v4 + 1;
              vtkm::Id v6 = v4 + x_verts;
              vtkm::Id v7 = v4 + x_verts + 1;
              
              conn_portal.Set(offset, v0);
              conn_portal.Set(offset+1, v1);
              conn_portal.Set(offset+2, v3);
              conn_portal.Set(offset+3, v2);
              conn_portal.Set(offset+4, v4);
              conn_portal.Set(offset+5, v5);
              conn_portal.Set(offset+6, v7);
              conn_portal.Set(offset+7, v6);
              offset = offset + 8;
            }
          }
        }

        vtkm::cont::CellSetSingleType<> cell_set;
        cell_set.Fill(nverts, shape_id, indices_per, connectivity);
        neles = cell_set.GetNumberOfCells();
        result->SetCellSet(cell_set);
      }
    }
    else
    {
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
        int32 z_elems = n_topo["elements/dims/k"].to_int();
        vtkm::cont::CellSetStructured<3> cell_set;
        cell_set.SetPointDimensions(vtkm::make_Vec(x_elems+1,
                                                   y_elems+1,
                                                   z_elems+1));
        cell_set.SetGlobalPointIndexStart(topo_origin);
        result->SetCellSet(cell_set);
        neles = x_elems * y_elems * z_elems;
      }
    }    
    return result;
}

//-----------------------------------------------------------------------------

vtkm::cont::DataSet *
VTKHDataAdapter::PointsImplicitBlueprintToVTKmDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed unstructured)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles  (will be the same as nverts)
     int &nverts,                    // output, number of verts (will be the same as neles)
     bool zero_copy)                 // attempt to zero copy
{
    vtkm::cont::DataSet *result = new vtkm::cont::DataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();

    int32 ndims;
    vtkm::cont::CoordinateSystem coords;
    if(n_coords["values/x"].dtype().is_float64())
    {
      index_t x_stride = n_coords["values/x"].dtype().stride();
      index_t x_element_stride = x_stride / sizeof(float64);
      index_t y_stride = n_coords["values/y"].dtype().stride();
      index_t y_element_stride = y_stride / sizeof(float64);
      index_t z_element_stride = 0;
      if(n_coords.has_path("values/z"))
      {
        index_t z_stride = n_coords["values/z"].dtype().stride();
        z_element_stride = z_stride / sizeof(float64);
      }

      coords = detail::GetExplicitCoordinateSystem<float64>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            x_element_stride,
                                                            y_element_stride,
                                                            z_element_stride,
                                                            zero_copy);
    }
    else if(n_coords["values/x"].dtype().is_float32())
    {
      index_t x_stride = n_coords["values/x"].dtype().stride();
      index_t x_element_stride = x_stride / sizeof(float32);
      index_t y_stride = n_coords["values/y"].dtype().stride();
      index_t y_element_stride = y_stride / sizeof(float32);
      index_t z_element_stride = 0;
      if(n_coords.has_path("values/z"))
      {
        index_t z_stride = n_coords["values/z"].dtype().stride();
        z_element_stride = z_stride / sizeof(float32);
      }

      coords = detail::GetExplicitCoordinateSystem<float32>(n_coords,
                                                            coords_name,
                                                            ndims,
                                                            x_element_stride,
                                                            y_element_stride,
                                                            z_element_stride,
                                                            zero_copy);
    }
    else
    {
      ASCENT_ERROR("Coordinate system must be floating point values");
    }

    result->AddCoordinateSystem(coords);

    vtkm::UInt8 shape_id = 1;
    vtkm::IdComponent indices_per = 1;
    vtkm::cont::CellSetSingleType<> cellset;
    // alloc conn to nverts, fill with 0 --> nverts-1)
    vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
    connectivity.Allocate(nverts);
    auto conn_portal = connectivity.WritePortal();
    for(int i = 0; i < nverts; ++i)
    {
        conn_portal.Set(i, i);
    }
    cellset.Fill(nverts, shape_id, indices_per, connectivity);
    neles = cellset.GetNumberOfCells();
    result->SetCellSet(cellset);
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
      index_t x_stride = n_coords["values/x"].dtype().stride();
      index_t x_element_stride = x_stride / sizeof(float64);
      index_t y_stride = n_coords["values/y"].dtype().stride();
      index_t y_element_stride = y_stride / sizeof(float64);
      index_t z_element_stride = 0;
      if(n_coords.has_path("values/z"))
      {
        index_t z_stride = n_coords["values/z"].dtype().stride();
        z_element_stride = z_stride / sizeof(float64);
      }

      //TODO:
      //can we assume all by checking one? 
      //or check ystride & zstride % float64 == 0? 
      if(x_stride % sizeof(float64) == 0)
      {
        coords = detail::GetExplicitCoordinateSystem<float64>(n_coords,
                                                              coords_name,
                                                              ndims,
                                                              x_element_stride,
                                                              y_element_stride,
                                                              z_element_stride,
                                                              zero_copy);
      }
    }
    else if(n_coords["values/x"].dtype().is_float32())
    {
      index_t x_stride = n_coords["values/x"].dtype().stride();
      index_t x_element_stride = x_stride / sizeof(float32);
      index_t y_stride = n_coords["values/y"].dtype().stride();
      index_t y_element_stride = y_stride / sizeof(float32);
      index_t z_element_stride = 0;
      if(n_coords.has_path("values/z"))
      {
        index_t z_stride = n_coords["values/z"].dtype().stride();
        z_element_stride = z_stride / sizeof(float32);
      }

      //TODO:
      //can we assume all by checking one? 
      //or check ystride & zstride % float64 == 0? 
      if(x_stride % sizeof(float32) == 0)
      {
        coords = detail::GetExplicitCoordinateSystem<float32>(n_coords,
                                                              coords_name,
                                                              ndims,
                                                              x_element_stride,
                                                              y_element_stride,
                                                              z_element_stride,
                                                              zero_copy);
      }
    }
    else
    {
      ASCENT_ERROR("Coordinate system must be floating point values");
    }

    result->AddCoordinateSystem(coords);

    // shapes, number of indices, and connectivity.
    // Will have to do something different if this is a "zoo"

    const Node &n_topo_eles = n_topo["elements"];
    std::string ele_shape = n_topo_eles["shape"].as_string();

    if(ele_shape == "mixed")
    {
        // blueprint allows mapping of shape names
        // to arbitrary ids, check if shape ids match the VTK-m ids
        index_t num_of_shapes = n_topo_eles["shape_map"].number_of_children();

        if(!CheckShapeMapVsVTKmShapeIds(n_topo_eles["shape_map"]))
        {
            Node ref_map;
            VTKmBlueprintShapeMap(ref_map);
            // TODO -- (strategy to remap ids)?
            ASCENT_ERROR("Shape Map Entries do not match required VTK-m Shape Ids."
                         << std::endl
                         << "Passed Shape Map:"  << std::endl
                         << n_topo_eles["shape_map"].to_yaml()
                         << std::endl
                         << "Supported Shape Map:"
                         << std::endl 
                         <<ref_map.to_yaml()
                         );
        }

        index_t num_ids  = n_topo_eles["connectivity"].dtype().number_of_elements();
        // number of elements is the number of shapes presented
        neles = (int) n_topo_eles["shapes"].dtype().number_of_elements();

        vtkm::cont::ArrayHandle<vtkm::Id> vtkm_conn;
        detail::BlueprintIndexArrayToVTKmIdArray(n_topo_eles["connectivity"],
                                                 zero_copy,
                                                 vtkm_conn);

        // shapes
        vtkm::cont::ArrayHandle<vtkm::UInt8> vtkm_shapes;
        detail::BlueprintIndexArrayToVTKmIdArray(n_topo_eles["shapes"],
                                                 zero_copy,
                                                 vtkm_shapes);

        // offsets
        vtkm::cont::ArrayHandle<vtkm::Id> vtkm_offsets;
        detail::BlueprintIndexArrayToVTKmIdArray(n_topo_eles["offsets"],
                                                 zero_copy,
                                                 vtkm_offsets);

        // vtk-m offsets needs an extra entry
        // the last entry needs to be the size of the conn array
        vtkm::cont::ArrayHandle<vtkm::Id> vtkm_offsets_full;
        vtkm_offsets_full.Allocate(neles + 1);
        vtkm::cont::ArrayHandle<vtkm::Id>::WritePortalType vtkm_offsets_full_wp = vtkm_offsets_full.WritePortal();
        vtkm::cont::ArrayHandle<vtkm::Id>::ReadPortalType vtkm_offsets_rp = vtkm_offsets.ReadPortal();

        for(int i=0;i<neles;i++)
        {
          vtkm_offsets_full_wp.Set(i,vtkm_offsets_rp.Get(i));
        }
        // set last
        vtkm_offsets_full_wp.Set(neles,num_ids);

        vtkm::cont::CellSetExplicit<> cell_set;
        cell_set.Fill(nverts, vtkm_shapes, vtkm_conn, vtkm_offsets_full);
        result->SetCellSet(cell_set);
        // for debugging help
        //result->PrintSummary(std::cout);
    }
    else
    {
        vtkm::cont::ArrayHandle<vtkm::Id> vtkm_conn;
        detail::BlueprintIndexArrayToVTKmIdArray(n_topo_eles["connectivity"],zero_copy,vtkm_conn);
        vtkm::UInt8 shape_id;
        vtkm::IdComponent indices_per;
        detail::VTKmCellShape(ele_shape, shape_id, indices_per);
        vtkm::cont::CellSetSingleType<> cell_set;
        cell_set.Fill(nverts, shape_id, indices_per, vtkm_conn);
        neles = cell_set.GetNumberOfCells();
        result->SetCellSet(cell_set);
    }
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

    vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::Any;
    if(assoc_str == "vertex")
    {
      vtkm_assoc = vtkm::cont::Field::Association::Points;
    }
    else if(assoc_str == "element")
    {
      vtkm_assoc = vtkm::cont::Field::Association::Cells;
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

        // vtk-m can stride as long as the strides are a multiple of the native stride

        // we compile vtk-h with fp types
        if(n_vals.dtype().is_float32())
        {
            // check that the byte stride is a multiple of native stride
            index_t stride = n_vals.dtype().stride();
            index_t element_stride = stride / sizeof(float32);

            //std::cout << "field name: " << field_name << " <float32>"
            //          << " byte stride: " << stride
            //          << " element_stride: " << element_stride << std::endl;
            // if element_stride is evenly divided by native, we are good to
            // use vtk m array handles
            if( stride % sizeof(float32) == 0 )
            {
                // in this case we can use a strided array handle
                dset->AddField(detail::GetField<float32>(n_vals,
                                                         field_name,
                                                         assoc_str,
                                                         topo_name,
                                                         element_stride,
                                                         zero_copy));
                supported_type = true;
            }
        }
        else if(n_vals.dtype().is_float64())
        {
            // check that the byte stride is a multiple of native stride
            index_t stride = n_vals.dtype().stride();
            index_t element_stride = stride / sizeof(float64);
            //std::cout << "field name: " << field_name << " <float64>"
            //          << " byte stride: " << stride
            //          << " element_stride: " << element_stride << std::endl;
            // if element_stride is evenly divided by native, we are good to
            // use vtk m array handles
            if( stride % sizeof(float64) == 0 )
            {
                // in this case we can use a strided array handle
                dset->AddField(detail::GetField<float64>(n_vals,
                                                         field_name,
                                                         assoc_str,
                                                         topo_name,
                                                         element_stride,
                                                         zero_copy));
                supported_type = true;
            }
        }
        else if(n_vals.dtype().is_uint8())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::uint8 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::uint8> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_uint16())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::uint16 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::uint16> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_uint32())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::uint32 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::uint32> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_uint64())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::uint64 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::uint64> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int8())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::int8 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::int8> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int16())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::int16 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::int16> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int32())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::int32 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::int32> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int64())
        {

          vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
          vtkm_arr.Allocate(num_vals);

          const conduit::int64 *input = n_vals.value();
          vtkm::cont::ArrayHandle<conduit::int64> input_arr = vtkm::cont::make_ArrayHandle(input, num_vals, vtkm::CopyFlag::Off);

          vtkm::cont::Invoker invoker;
          vtkh::VTKmTypeCast worklet;

          invoker(worklet,input_arr,vtkm_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Points,
                                               vtkm_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                               vtkm::cont::Field::Association::Cells,
                                               vtkm_arr));
              supported_type = true;
          }
        }
        // ***********************************************************************
        // ***********************************************************************
        // ***********************************************************************
        // NOTE: TODO OUR VTK-M is not compiled with int32 and int64 support ...
        // ***********************************************************************
        // These cases fail and provide this error message:
        //   Execution failed with vtkm: Could not find appropriate cast for array in CastAndCall.
        //   Array: valueType=x storageType=N4vtkm4cont15StorageTagBasicE 27 values occupying 216 bytes [0 1 2 ... 24 25 26]
        //   TypeList: N4vtkm4ListIJfdEEE
        // ***********************************************************************
        // ***********************************************************************
        // NOTE: int32 should work as of sept 10 2024 
        // NOTE: int32 & 64 are back to not working as of March 28 2025
        // ***********************************************************************
        //
        //else if(n_vals.dtype().is_int32())
        //{
        //    // check that the byte stride is a multiple of native stride
        //    index_t stride = n_vals.dtype().stride();
        //    index_t element_stride = stride / sizeof(int32);
        //    //std::cout << "field name: " << field_name << " <int32>"
        //    //          << " byte stride: " << stride
        //    //          << " element_stride: " << element_stride << std::endl;
        //    // if element_stride is evenly divided by native, we are good to
        //    // use vtk m array handles
        //    if( stride % sizeof(int32) == 0 )
        //    {
        //        // in this case we can use a strided array handle
        //        dset->AddField(detail::GetField<int32>(n_vals,
        //                                                 field_name,
        //                                                 assoc_str,
        //                                                 topo_name,
        //                                                 element_stride,
        //                                                 zero_copy));
        //        supported_type = true;
        //    }
        //}
        //else if(n_vals.dtype().is_int64())
        //{
        //    // check that the byte stride is a multiple of native stride
        //    index_t stride = n_vals.dtype().stride();
        //    index_t element_stride = stride / sizeof(int64);
        //    //std::cout << "field name: " << field_name << " <int64>"
        //    //          << " byte stride: " << stride
        //    //          << " element_stride: " << element_stride << std::endl;
        //    // if element_stride is evenly divided by native, we are good to
        //    // use vtk m array handles
        //    if( stride % sizeof(int64) == 0 )
        //    {
        //        // in this case we can use a strided array handle
        //        dset->AddField(detail::GetField<int64>(n_vals,
        //                                                 field_name,
        //                                                 assoc_str,
        //                                                 topo_name,
        //                                                 element_stride,
        //                                                 zero_copy));
        //        supported_type = true;
        //    }
        //}
        //else if(n_vals.dtype().is_uint64())
        //{
        //    // check that the byte stride is a multiple of native stride
        //    index_t stride = n_vals.dtype().stride();
        //    index_t element_stride = stride / sizeof(uint64);
        //    //std::cout << "field name: " << field_name << " <int64>"
        //    //          << " byte stride: " << stride
        //    //          << " element_stride: " << element_stride << std::endl;
        //    // if element_stride is evenly divided by native, we are good to
        //    // use vtk m array handles
        //    if( stride % sizeof(uint64) == 0 )
        //    {
        //        // in this case we can use a strided array handle
        //        dset->AddField(detail::GetField<uint64>(n_vals,
        //                                                 field_name,
        //                                                 assoc_str,
        //                                                 topo_name,
        //                                                 element_stride,
        //                                                 zero_copy));
        //        supported_type = true;
        //    }
        //}

        // vtk-m cant support zero copy for this layout or was not compiled to expose this datatype
        // use float64 by default
        if(!supported_type)
        {

            //std::cout << "WE ARE IN UNSUPPORTED DATA TYPE: "
            //          << n_vals.dtype().name() << std::endl;
            // convert to float64, we use this as a compromise to cover the widest range
            vtkm::cont::ArrayHandle<vtkm::Float64> vtkm_arr;
            vtkm_arr.Allocate(num_vals);

            // TODO -- FUTURE: Do this conversion w/ device if on device
            void *ptr = (void*) vtkh::GetVTKMPointer(vtkm_arr);
            Node n_tmp;
            n_tmp.set_external(DataType::float64(num_vals),ptr);
            n_vals.to_float64_array(n_tmp);

            // add field to dataset
            if(assoc_str == "vertex")
            {
                dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                                 vtkm::cont::Field::Association::Points,
                                                 vtkm_arr));
            }
            else if( assoc_str == "element")
            {
                dset->AddField(vtkm::cont::Field(field_name.c_str(),
                                                 vtkm::cont::Field::Association::Cells,
                                                 vtkm_arr));
            }
        // else
        // {
        //     std::cout << "SUPPORTED DATA TYPE: "
        //               << n_vals.dtype().name() << std::endl;
        // }
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

    vtkm::cont::Field::Association vtkm_assoc = vtkm::cont::Field::Association::Any;
    if(assoc_str == "vertex")
    {
      vtkm_assoc = vtkm::cont::Field::Association::Points;
    }
    else if(assoc_str == "element")
    {
      vtkm_assoc = vtkm::cont::Field::Association::Cells;
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

void
VTKHDataAdapter::AddMatSets(const std::string &matset_name,
                            const Node &n_matset,
                            const std::string &topo_name,
                            int neles,
                            vtkm::cont::DataSet *dset,
                            bool zero_copy)                 // attempt to zero copy
{

    if(!n_matset.has_child("volume_fractions"))
        ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);
    //TODO: zero_copy = true segfaulting in vtkm mir filter
    //zero_copy = false;
    
    
    std::string assoc_str = "element";
    //fields required from VTK-m MIR filter
    //std::string length_name, offsets_name, ids_name, vfs_name;
    std::string length_name = "sizes";//matset_name + "_lengths";
    std::string offsets_name = "offsets";//matset_name + "_offsets";
    std::string ids_name = "material_ids";//matset_name + "_ids";
    std::string vfs_name = "volume_fractions";//matset_name + "_vfs";
    //matset is "sparse_by_element"
    if(n_matset.has_child("material_map"))
    {
        try
        {
            bool supported_type = false;

            // we compile vtk-h with fp types
            if(n_matset["volume_fractions"].dtype().is_float32())
            {
                //add materials directly
                const conduit::Node &n_length = n_matset["sizes"];
                dset->AddField(detail::GetField<int>(n_length,
                                                     length_name,
                                                     assoc_str,
                                                     topo_name,
                                                     index_t(1),
                                                     zero_copy));
                const conduit::Node &n_offsets = n_matset["offsets"];
                dset->AddField(detail::GetField<int>(n_offsets,
                                                     offsets_name,
                                                     assoc_str,
                                                     topo_name,
                                                     index_t(1),
                                                     zero_copy));
                const conduit::Node &n_material_ids = n_matset["material_ids"];
                int num_vals = n_material_ids.dtype().number_of_elements();
                if(n_material_ids.dtype().is_int32())
                {
                    const conduit::int32 *n_ids = n_material_ids.value();
                    const vector<conduit::int32> vec_ids(n_ids, n_ids + num_vals);
                    bool zeroes = std::any_of(vec_ids.begin(), vec_ids.end(), [](int value) { return value<=0; });
                    if(zeroes) //need to make a copy and increment all material ids
                    {
                        conduit::Node n_mat_ids = n_matset["material_ids"];
                        conduit::int32 *tmp_vec_ids = n_mat_ids.value();
                        for(index_t i = 0; i < num_vals; ++i)
                        {
                            tmp_vec_ids[i] += 1.0; 
                        }
                        vtkm::cont::Field field_copy = detail::GetField<int32>(n_mat_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               false);
                        dset->AddField(field_copy);
                    }
                    else //can zero copy the material ids
                    {
                        vtkm::cont::Field field_copy = detail::GetField<int32>(n_material_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               zero_copy);

                        dset->AddField(field_copy);
                    }
                }
                else if(n_material_ids.dtype().is_int64())
                {
                    const conduit::int64 *n_ids = n_material_ids.value();
                    const vector<conduit::int64> vec_ids(n_ids, n_ids + num_vals);
                    bool zeroes = std::any_of(vec_ids.begin(), vec_ids.end(), [](int value) { return value<=0; });
                    if(zeroes) //need to make a copy and increment all material ids
                    {
                        conduit::Node n_mat_ids = n_matset["material_ids"];
                        conduit::int64 *tmp_vec_ids = n_mat_ids.value();
                        for(index_t i = 0; i < num_vals; ++i)
                        {
                            tmp_vec_ids[i] += 1.0; 
                        }
                        vtkm::cont::Field field_copy = detail::GetField<int64>(n_mat_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               false);
                        dset->AddField(field_copy);
                    }
                    else //can zero copy the material ids
                    {
                        vtkm::cont::Field field_copy = detail::GetField<int64>(n_material_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               zero_copy);

                        dset->AddField(field_copy);
                    }
                }
                else
                {
                    ASCENT_ERROR("Unsupported integer type for material IDs");
                }
                const conduit::Node &n_volume_fractions = n_matset["volume_fractions"];
                dset->AddField(detail::GetField<float32>(n_volume_fractions,
                                                         vfs_name,
                                                         "whole",
                                                         topo_name,
                                                         index_t(1),
                                                         zero_copy));
                supported_type = true;
            }
            else if(n_matset["volume_fractions"].dtype().is_float64())
            {
                //add materials directly
                const Node &n_length = n_matset["sizes"];
                dset->AddField(detail::GetField<int>(n_length,
                                                     length_name,
                                                     assoc_str,
                                                     topo_name,
                                                     index_t(1),
                                                     zero_copy));
                const conduit::Node &n_offsets = n_matset["offsets"];
                dset->AddField(detail::GetField<int>(n_offsets,
                                                     offsets_name,
                                                     assoc_str,
                                                     topo_name,
                                                     index_t(1),
                                                     zero_copy));
                const conduit::Node &n_material_ids = n_matset["material_ids"];
                int num_vals = n_material_ids.dtype().number_of_elements(); 
                if(n_material_ids.dtype().is_int32())
                {
                    const conduit::int32 *n_ids = n_material_ids.value();
                    const vector<conduit::int32> vec_ids(n_ids, n_ids + num_vals);
                    bool zeroes = std::any_of(vec_ids.begin(), vec_ids.end(), [](int value) { return value<=0; });
                    if(zeroes) //need to make a copy and increment all material ids
                    {
                        conduit::Node n_mat_ids = n_matset["material_ids"];
                        conduit::int32 *tmp_vec_ids = n_mat_ids.value();
                        for(index_t i = 0; i < num_vals; ++i)
                        {
                            tmp_vec_ids[i] += 1.0; 
                        }
                        vtkm::cont::Field field_copy = detail::GetField<int32>(n_mat_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               false);
                        dset->AddField(field_copy);
                    }
                    else //can zero copy the material ids
                    {
                        vtkm::cont::Field field_copy = detail::GetField<int32>(n_material_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               zero_copy);

                        dset->AddField(field_copy);
                    }
                }
                else if(n_material_ids.dtype().is_int64())
                {
                    const conduit::int64 *n_ids = n_material_ids.value();
                    const vector<conduit::int64> vec_ids(n_ids, n_ids + num_vals);
                    bool zeroes = std::any_of(vec_ids.begin(), vec_ids.end(), [](int value) { return value<=0; });
                    if(zeroes) //need to make a copy and increment all material ids
                    {
                      conduit::Node n_mat_ids = n_matset["material_ids"];
                      conduit::int64 *tmp_vec_ids = n_mat_ids.value();
                      for(index_t i = 0; i < num_vals; ++i)
                      {
                        tmp_vec_ids[i] += 1.0; 
                      }
                      vtkm::cont::Field field_copy = detail::GetField<int64>(n_mat_ids,
                                                                             ids_name,
                                                                             "whole",
                                                                             topo_name,
                                                                             index_t(1),
                                                                             false);
                      dset->AddField(field_copy);
                    }
                    else //can zero copy the material ids
                    {
                      vtkm::cont::Field field_copy = detail::GetField<int64>(n_material_ids,
                                                                             ids_name,
                                                                             "whole",
                                                                             topo_name,
                                                                             index_t(1),
                                                                             zero_copy);

                      dset->AddField(field_copy);
                    }
                }
                else
                {
                    ASCENT_ERROR("Unsupported integer type for material IDs");
                }
                const conduit::Node &n_volume_fractions = n_matset["volume_fractions"];
                dset->AddField(detail::GetField<float64>(n_volume_fractions,
                                                         vfs_name,
                                                         "whole",
                                                         topo_name,
                                                         index_t(1),
                                                         zero_copy));
                supported_type = true;
            }
        }
        catch (vtkm::cont::Error error)
        {
            ASCENT_ERROR("VTKm exception:" << error.GetMessage());
        }

    }
    else if(n_matset.has_child("element_ids"))//matset is "sparse_by_material"
    {
        int num_ids = n_matset["element_ids"].number_of_children();
        if(num_ids == 0)
        {
            ASCENT_ERROR("No element ids were defined for matset: " << matset_name);
        }

        int num_materials = n_matset["volume_fractions"].number_of_children();
        if(num_materials == 0)
        {
            ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);
        }
        
        if(num_materials != num_ids)
        {
            ASCENT_ERROR("Number of materials (" << num_materials << 
                         ") does not match number of elment IDs(" << num_ids << 
                         " defined for matset: " << matset_name);
        }

        try
        {
            bool supported_type = false;

            const conduit::Node n_vfs = n_matset["volume_fractions"].child(0);
            // we compile vtk-h with fp types
            if(n_vfs.dtype().is_float32())
            {
                supported_type = true;
                //add calculated material fields for vtkm
                vtkm::cont::Field length, offsets, ids, vfs;
                detail::GetMatSetFields<int,float32>(n_matset, 
                                                     length_name, 
                                                     offsets_name, 
                                                     ids_name,
                                                     vfs_name,
                                                     topo_name, 
                                                     neles, 
                                                     length, 
                                                     offsets,
                                                     ids,
                                                     vfs);
                dset->AddField(length);
                dset->AddField(offsets);
                dset->AddField(ids);
                dset->AddField(vfs);
            }
            else if(n_vfs.dtype().is_float64())
            {
                supported_type = true;
                //add calculated material fields for vtkm
                vtkm::cont::Field length, offsets, ids, vfs;
                detail::GetMatSetFields<int,float64>(n_matset, 
                                                     length_name, 
                                                     offsets_name, 
                                                     ids_name,
                                                     vfs_name,
                                                     topo_name, 
                                                     neles, 
                                                     length, 
                                                     offsets,
                                                     ids,
                                                     vfs);
                dset->AddField(length);
                dset->AddField(offsets);
                dset->AddField(ids);
                dset->AddField(vfs);
            }
        }
        catch (vtkm::cont::Error error)
        {
            ASCENT_ERROR("VTKm exception:" << error.GetMessage());
        }
    }
    else //matset is "full"
    {
        int num_materials = n_matset["volume_fractions"].number_of_children();
        if(num_materials == 0)
            ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);

        const Node n_material = n_matset["volume_fractions"].child(0);
        std::string material_name = n_material.name();

        int num_vals = n_material.dtype().number_of_elements();

        if(num_vals != neles )
        {
            ASCENT_ERROR("Number of vf values " 
                          << num_vals 
                          << " for material " 
                          << material_name 
                          << " does not equal number of cells "
                          << neles);
        }
        try
        {
            bool supported_type = false;

            // we compile vtk-h with fp types
            if(n_material.dtype().is_float32())
            {
                supported_type = true;
                //add calculated material fields for vtkm
                int total;
                vtkm::cont::Field length, offsets, ids, vfs;
                detail::GetMatSetFields<int,float32>(n_matset, 
                                                     length_name, 
                                                     offsets_name, 
                                                     ids_name,
                                                     vfs_name,
                                                     topo_name, 
                                                     neles, 
                                                     length, 
                                                     offsets,
                                                     ids,
                                                     vfs);
                dset->AddField(length);
                dset->AddField(offsets);
                dset->AddField(ids);
                dset->AddField(vfs);
            }
            else if(n_material.dtype().is_float64())
            {
                supported_type = true;
                //add calculated material fields for vtkm
                int total;
                vtkm::cont::Field length, offsets, ids, vfs;
                detail::GetMatSetFields<int,float64>(n_matset, 
                                                     length_name, 
                                                     offsets_name, 
                                                     ids_name,
                                                     vfs_name,
                                                     topo_name, 
                                                     neles, 
                                                     length, 
                                                     offsets,
                                                     ids,
                                                     vfs);
                dset->AddField(length);
                dset->AddField(offsets);
                dset->AddField(ids);
                dset->AddField(vfs);
            }
        }
        catch (vtkm::cont::Error error)
        {
            ASCENT_ERROR("VTKm exception:" << error.GetMessage());
        }
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
  else if(shape_id == vtkm::CELL_SHAPE_PYRAMID)
  {
    name = "pyramid";
  }
  else if(shape_id == vtkm::CELL_SHAPE_WEDGE)
  {
    name = "wedge";
  }
  return name;
}


inline index_t
vtkm_shape_size(vtkm::Id shape_id)
{
    switch(shape_id)
    {
        // point
        case vtkm::CELL_SHAPE_VERTEX:  return 1; break;
        // line
        case vtkm::CELL_SHAPE_LINE:  return 2; break;
        // tri
        case vtkm::CELL_SHAPE_TRIANGLE:  return 3; break;
        // quad
        case vtkm::CELL_SHAPE_QUAD:  return 4; break;
        // tet
        case vtkm::CELL_SHAPE_TETRA: return 4; break;
        // hex
        case vtkm::CELL_SHAPE_HEXAHEDRON: return 8; break;
        // pyramid
        case vtkm::CELL_SHAPE_PYRAMID: return 5; break;
        // wedge
        case vtkm::CELL_SHAPE_WEDGE: return 6; break;
        //
        default: return 0;
    }
}


void
generate_sizes_from_shapes(const conduit::Node &shapes,conduit::Node &sizes)
{
    index_t num_eles = shapes.dtype().number_of_elements();
    uint8_array   shapes_arr = shapes.value();
    index_t_array sizes_arr = sizes.value();

    for(index_t i=0; i < num_eles; i++)
    {
        sizes_arr[i] = vtkm_shape_size(shapes_arr[i]);
    }
    
}

bool
VTKHDataAdapter::VTKmTopologyToBlueprint(conduit::Node &output,
                                         const vtkm::cont::DataSet &data_set,
                                         const std::string &topo_name,
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

    vtkm::cont::UnknownArrayHandle coordsHandle(coords.GetData());

    if(coordsHandle.CanConvert<Coords32>())
    {
      Coords32 points;
      coordsHandle.AsArrayHandle(points);

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
      CoordsVec32 points;
      coordsHandle.AsArrayHandle(points);

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
    else if(coordsHandle.CanConvert<Coords64>())
    {
      Coords64 points;
      coordsHandle.AsArrayHandle(points);

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
    else if(coordsHandle.IsType<CoordsVec64>())
    {
      CoordsVec64 points;
      coordsHandle.AsArrayHandle(points);

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

      vtkm::cont::UnknownCellSet dyn_cells = data_set.GetCellSet();
      using Structured2D = vtkm::cont::CellSetStructured<2>;
      using Structured3D = vtkm::cont::CellSetStructured<3>;
      if(dyn_cells.CanConvert<Structured2D>())
      {
        Structured2D cells = dyn_cells.AsCellSet<Structured2D>();
        vtkm::Id2 cell_dims = cells.GetCellDimensions();
        output["topologies/"+topo_name+"/elements/dims/i"] = (int) cell_dims[0];
        output["topologies/"+topo_name+"/elements/dims/j"] = (int) cell_dims[1];
      }
      else if(dyn_cells.CanConvert<Structured3D>())
      {
        Structured3D cells = dyn_cells.AsCellSet<Structured3D>();
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
      vtkm::cont::UnknownCellSet dyn_cells = data_set.GetCellSet();

      using SingleType = vtkm::cont::CellSetSingleType<>;
      using MixedType = vtkm::cont::CellSetExplicit<>;

      if(dyn_cells.CanConvert<SingleType>())
      {
        SingleType cells = dyn_cells.AsCellSet<SingleType>();
        vtkm::UInt8 shape_id = cells.GetCellShape(0);
        std::string conduit_name = GetBlueprintCellName(shape_id);
        output["topologies/"+topo_name+"/elements/shape"] = conduit_name;

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
        auto cells = dyn_cells.AsCellSet<vtkm::cont::CellSetExplicit<>>();
        auto shapes = cells.GetShapesArray(vtkm::TopologyElementTagCell(),
                                           vtkm::TopologyElementTagPoint());

        std::string conduit_name = GetBlueprintCellName(shape_id);
        output["topologies/"+topo_name+"/elements/shape"] = conduit_name;

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
        //data_set.PrintSummary(std::cout);
        //ASCENT_ERROR("Mixed explicit types not implemented");
        MixedType cells = dyn_cells.AsCellSet<MixedType>();
        Node &topo_ele = output
            ["topologies/" + topo_name + "/elements"];
        topo_ele["shape"] = "mixed";

        VTKmBlueprintShapeMap(topo_ele["shape_map"]);

        size_t num_cells = static_cast<size_t>(cells.GetNumberOfCells());
        auto vtkm_shapes  = cells.GetShapesArray(vtkm::TopologyElementTagCell{}, vtkm::TopologyElementTagPoint{});
        auto vtkm_conn    = cells.GetConnectivityArray(vtkm::TopologyElementTagCell{}, vtkm::TopologyElementTagPoint{});
        auto vtkm_offsets = cells.GetOffsetsArray(vtkm::TopologyElementTagCell{}, vtkm::TopologyElementTagPoint{});


        std::size_t conn_size = static_cast<std::size_t>(vtkm_conn.GetNumberOfValues());

        if(zero_copy)
        {
            topo_ele["shapes"].set_external(vtkh::GetVTKMPointer(vtkm_shapes), num_cells);
            topo_ele["connectivity"].set_external(vtkh::GetVTKMPointer(vtkm_conn), conn_size);
            topo_ele["offsets"].set_external(vtkh::GetVTKMPointer(vtkm_offsets), num_cells);
        }
        else
        {
            topo_ele["shapes"].set(vtkh::GetVTKMPointer(vtkm_shapes), num_cells);
            topo_ele["connectivity"].set(vtkh::GetVTKMPointer(vtkm_conn), conn_size);
            topo_ele["offsets"].set(vtkh::GetVTKMPointer(vtkm_offsets), num_cells);
        }

        // bp requires sizes, so we have to compute them
        topo_ele["sizes"].set(DataType::index_t(num_cells));
        generate_sizes_from_shapes(topo_ele["shapes"],topo_ele["sizes"]);
      }

    }
  }
  return is_empty;
}

//---------------------------------------------------------------------------//
// helper to set conduit field values for from a vector style vtkm array
//---------------------------------------------------------------------------//
template<typename T, int N>
void SetFieldValuesFromVTKmUnknownArrayHandleVec(vtkm::cont::UnknownArrayHandle &dyn_handle,
                                                 bool zero_copy,
                                                 Node &output_values)
{
    static_assert(N > 1 && N < 4, "Vecs must be size 2 or 3");

    static const std::vector<std::string> comp_names = { "u", "v", "w"};

    bool try_zero_copy = zero_copy;


    for(index_t comp = 0; comp < N; comp++)
    {
      zero_copy = try_zero_copy;
      vtkm::cont::ArrayHandleStride<T> stride_handle;

      Node &output_values_component = output_values[comp_names[comp]];

      if(zero_copy)
      {
        try
        {
          stride_handle = dyn_handle.ExtractComponent<T>(comp,vtkm::CopyFlag::Off);
        }
        catch(...)
        {
          stride_handle = dyn_handle.ExtractComponent<T>(comp,vtkm::CopyFlag::On);
          zero_copy = false;
        }
      }
      else
      {
        stride_handle = dyn_handle.ExtractComponent<T>(comp,vtkm::CopyFlag::On);
      }

      vtkm::cont::ArrayHandleBasic<T> basic_array = stride_handle.GetBasicArray();

      if(zero_copy)
      {
        output_values_component.set_external((T*) vtkh::GetVTKMPointer(basic_array),
                                             stride_handle.GetNumberOfValues(),
                                             sizeof(T)*stride_handle.GetOffset(),   // starting offset in bytes
                                             sizeof(T)*stride_handle.GetStride());  // stride in bytes
      }
      else
      {
        output_values_component.set((T*) vtkh::GetVTKMPointer(basic_array),
                                    stride_handle.GetNumberOfValues(),
                                    sizeof(T)*stride_handle.GetOffset(),   // starting offset in bytes
                                    sizeof(T)*stride_handle.GetStride());  // stride in bytes
      }
      
    }
}


//---------------------------------------------------------------------------//
// helper to set conduit field values for from a vtkm array
//---------------------------------------------------------------------------//
template<typename T>
void SetFieldValuesFromVTKmUnknownArrayHandle(vtkm::cont::UnknownArrayHandle &dyn_handle,
                                              bool zero_copy,
                                              Node &output_values)
{
    vtkm::cont::ArrayHandleStride<T> stride_handle;
    if(zero_copy)
    {
      // if we cannot zero copy, extract component will throw an exception
      // and we can fall back to copying
      try
      {
        stride_handle = dyn_handle.ExtractComponent<T>(0,vtkm::CopyFlag::Off);
      }catch(vtkm::cont::Error &e)  // fall back to copy
      {
        stride_handle = dyn_handle.ExtractComponent<T>(0,vtkm::CopyFlag::On);
        zero_copy = false;
      }
    }

    vtkm::cont::ArrayHandleBasic<T> basic_array = stride_handle.GetBasicArray();
    if(zero_copy)
    {
      output_values.set_external(vtkh::GetVTKMPointer(basic_array),
                                 stride_handle.GetNumberOfValues());
    }
    else // copy case
    {
      output_values.set(vtkh::GetVTKMPointer(basic_array),
                        stride_handle.GetNumberOfValues());
    }
}



void
VTKHDataAdapter::VTKmFieldToBlueprint(conduit::Node &output,
                                      const vtkm::cont::Field &field,
                                      const std::string &topo_name,
                                      bool zero_copy)
{
  std::string name = field.GetName();
  std::string path = "fields/" + name;
  bool assoc_points = vtkm::cont::Field::Association::Points == field.GetAssociation();
  bool assoc_cells  = vtkm::cont::Field::Association::Cells == field.GetAssociation();
  //bool assoc_mesh  = vtkm::cont::Field::ASSOC_WHOLE_MESH == field.GetAssociation();
  if(!assoc_points && ! assoc_cells)
  {
    ASCENT_ERROR("Field must be associated with cells or points\n");
  }
  std::string conduit_name;

  if(assoc_points) conduit_name = "vertex";
  else conduit_name = "element";

  output[path + "/association"] = conduit_name;
  output[path + "/topology"] = topo_name;
  Node &output_values =   output[path + "/values"];
  vtkm::cont::UnknownArrayHandle dyn_handle = field.GetData();

  //
  // this can be literally anything. Lets do some exhaustive casting
  //
  if (dyn_handle.IsValueType<vtkm::Vec<vtkm::Float32, 3>>())
  {
      SetFieldValuesFromVTKmUnknownArrayHandleVec<vtkm::Float32, 3>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<vtkm::Vec<vtkm::Float64, 3>>())
  {
      SetFieldValuesFromVTKmUnknownArrayHandleVec<vtkm::Float64, 3>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<vtkm::Vec<vtkm::Int32, 3>>())
  {
      SetFieldValuesFromVTKmUnknownArrayHandleVec<vtkm::Int32, 3>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<vtkm::Vec<vtkm::Float32, 2>>())
  {
      SetFieldValuesFromVTKmUnknownArrayHandleVec<vtkm::Float32, 2>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<vtkm::Vec<vtkm::Float64, 2>>())
  {
      SetFieldValuesFromVTKmUnknownArrayHandleVec<vtkm::Float64, 2>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<vtkm::Vec<vtkm::Int32, 2>>())
  {
      SetFieldValuesFromVTKmUnknownArrayHandleVec<vtkm::Int32, 2>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if(dyn_handle.IsValueType<vtkm::Float32>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::Float32>(dyn_handle,
                                  zero_copy,
                                  output_values);
  }
  else if(dyn_handle.IsValueType<vtkm::Float64>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::Float64>(dyn_handle,
                                  zero_copy,
                                  output_values);
  }
  else if(dyn_handle.IsValueType<vtkm::Int8>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::Int8>(dyn_handle,
                              zero_copy,
                              output_values);

  }
  else if(dyn_handle.IsValueType<vtkm::Int32>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::Int32>(dyn_handle,
                              zero_copy,
                              output_values);
  }
  else if(dyn_handle.IsValueType<vtkm::Int64>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::Int64>(dyn_handle,
                              zero_copy,
                              output_values);

  }
  else if(dyn_handle.IsValueType<vtkm::UInt32>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::UInt32>(dyn_handle,
                              zero_copy,
                              output_values);
  }
  else if(dyn_handle.IsValueType<vtkm::UInt8>())
  {
    SetFieldValuesFromVTKmUnknownArrayHandle<vtkm::UInt8>(dyn_handle,
                              zero_copy,
                              output_values);
  }

  else
  {
    std::stringstream msg;
    msg<<"Field type unsupported for conversion to blueprint.\n";
    field.PrintSummary(msg);
    msg<<" Skipping.";
    ASCENT_INFO(msg.str());
  }
}



//-----------------------------------------------------------------------------
bool
VTKHDataAdapter::CheckShapeMapVsVTKmShapeIds(const Node &shape_map)
{
    bool res = true;
    Node ref_map;

    VTKHDataAdapter::VTKmBlueprintShapeMap(ref_map);
    NodeConstIterator itr = shape_map.children();
    while(itr.has_next() && res)
    {
        const Node &curr = itr.next();
        std::string name = itr.name();
        if(curr.dtype().is_number() && ref_map.has_child(name))
        {
            // check vs ref map
            res = ( ref_map[name].to_index_t() == curr.to_index_t() );
        }
        else // unknown/unsupported shape type
        {
            res = false;
        }
    }
    return res;
}



void
VTKHDataAdapter::VTKmBlueprintShapeMap(conduit::Node &output)
{
    output.reset();
    output["tri"]     = 5;
    output["quad"]    = 9;
    output["tet"]     = 10;
    output["hex"]     = 12;
    output["point"]   = 1;
    output["line"]    = 3;
    output["wedge"]   = 13;
    output["pyramid"] = 14;
}

void VTKHDataAdapter::VTKHCollectionToBlueprintDataSet(VTKHCollection *collection,
                                                       conduit::Node &node,
                                                       bool zero_copy)
{
  node.reset();

  bool success = true;
  // we have to re-merge the domains so all domains with the same
  // domain id end up in a single domain
  std::map<int, std::map<std::string,vtkm::cont::DataSet>> domain_map;
  domain_map = collection->by_domain_id();
  std::string err_msg;
  try
  {
    for(auto domain_it : domain_map)
    {
      const int domain_id = domain_it.first;

      conduit::Node &dom = node.append();
      dom["state/domain_id"] = (int) domain_id;

      for(auto topo_it : domain_it.second)
      {
        const std::string topo_name = topo_it.first;
        vtkm::cont::DataSet &dataset = topo_it.second;
        VTKHDataAdapter::VTKmToBlueprintDataSet(&dataset, dom, topo_name, zero_copy);
      }
    }
  }
  catch (conduit::Error error)
  {
     err_msg = error.message();
     success = false;
  }
  catch (vtkm::cont::Error error)
  {
    err_msg =  error.GetMessage();
    success = false;
  }
  catch (...)
  {
      err_msg = "[Unknown exception]";
  }

  success = global_agreement(success);
  if(!success)
  { 
    //  TODO: broadcast error messages to root?
    ASCENT_ERROR("Failed to convert VTK-m data set to blueprint: " << err_msg);
  }
}

void
VTKHDataAdapter::VTKHToBlueprintDataSet(vtkh::DataSet *dset,
                                        conduit::Node &node,
                                        bool zero_copy)
{
  node.reset();
  bool success = true;
  std::string err_msg;
  try
  {
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
  catch (conduit::Error error)
  {
      err_msg = error.message();
      success = false;
  }
  catch (vtkm::cont::Error error)
  {
      err_msg = error.GetMessage();
      success = false;
  }
  catch (...)
  {
      err_msg = "[Unknown exception]";
      success = false;
  }

  success = global_agreement(success);
  if(!success)
  {
    //  TODO: broadcast error messages to root?
    ASCENT_ERROR("Failed to convert VTK-m data set to blueprint: " << err_msg);
  }
}

void
VTKHDataAdapter::VTKmToBlueprintDataSet(const vtkm::cont::DataSet *dset,
                                        conduit::Node &node,
                                        const std::string &topo_name,
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
      // as of VTK-m 2.0, coordinates are also stored as VTK-m fields
      // skip wrapping coords as a field, since they are 
      // already captured in the blueprint coordset
      if (!dset->HasCoordinateSystem(field.GetName()))
      {
          VTKmFieldToBlueprint(node, field, topo_name, zero_copy);
      }
    }
  }
}


};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
