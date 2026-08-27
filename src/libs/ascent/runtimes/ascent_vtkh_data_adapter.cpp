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
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

// third party includes

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#include <ascent_logging.hpp>
#include <ascent_logging_old.hpp>

// Viskores includes
#define VISKORES_USE_DOUBLE_PRECISION
#include <viskores/cont/DataSet.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandleCompositeVector.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleExtractComponent.h>
#include <viskores/cont/CoordinateSystem.h>
#include <viskores/cont/Invoker.h>
#include <vtkh/DataSet.hpp>

// other ascent includes
#include <ascent_logging.hpp>
#include <ascent_block_timer.hpp>
#include <ascent_mpi_utils.hpp>
#include <vtkh/utils/viskores_array_utils.hpp>
#include <vtkh/utils/viskores_dataset_info.hpp>

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

viskores::Id3 topo_origin(const conduit::Node &n_topo)
{
  viskores::Id3 topo_origin(0,0,0);
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
void CopyArray(viskores::cont::ArrayHandle<T> &viskores_handle, const T* vals_ptr, const int size, bool zero_copy)
{
  viskores::CopyFlag copy = viskores::CopyFlag::On;
  if(zero_copy)
  {
    copy = viskores::CopyFlag::Off;
  }

  viskores_handle = viskores::cont::make_ArrayHandle(vals_ptr, size, copy);
}

// Convert a Conduit integer array into a standard vector.
std::vector<index_t>
IndexVector(const conduit::Node &node)
{
  conduit::Node idx_node;
  node.to_index_t_array(idx_node);
  conduit::index_t_array idx_array = idx_node.as_index_t_array();

  std::vector<index_t> res(idx_array.number_of_elements());
  for(index_t i = 0; i < idx_array.number_of_elements(); ++i)
  {
    res[i] = idx_array[i];
  }
  return res;
}

// Check for Blueprint strided layout metadata.
bool
HasStridedLayout(const conduit::Node &node)
{
  return node.has_child("offsets") && node.has_child("strides");
}

// Get logical field or coordinate dimensions from topology dimensions.
std::vector<index_t>
LogicalDims(const conduit::Node &n_topo, const std::string &assoc)
{
  const conduit::Node &n_dims = n_topo["elements/dims"];
  std::vector<index_t> dims;
  dims.push_back(n_dims["i"].to_index_t());
  dims.push_back(n_dims["j"].to_index_t());
  if(n_dims.has_child("k"))
  {
    dims.push_back(n_dims["k"].to_index_t());
  }

  if(assoc == "vertex")
  {
    for(size_t i = 0; i < dims.size(); ++i)
    {
      dims[i] += 1;
    }
  }
  return dims;
}

// Build logical-to-source indices for a strided Blueprint window.
viskores::cont::ArrayHandle<viskores::Id>
LogicalIndexArray(const std::vector<index_t> &logical_dims,
                  const std::vector<index_t> &offsets,
                  const std::vector<index_t> &strides,
                  const index_t element_stride,
                  viskores::Id &source_size)
{
  const index_t ni = logical_dims[0];
  const index_t nj = logical_dims.size() > 1 ? logical_dims[1] : 1;
  const index_t nk = logical_dims.size() > 2 ? logical_dims[2] : 1;

  const index_t oi = offsets.size() > 0 ? offsets[0] : 0;
  const index_t oj = offsets.size() > 1 ? offsets[1] : 0;
  const index_t ok = offsets.size() > 2 ? offsets[2] : 0;

  const index_t si = strides.size() > 0 ? strides[0] : 1;
  const index_t sj = strides.size() > 1 ? strides[1] : ni;
  const index_t sk = strides.size() > 2 ? strides[2] : ni * nj;

  viskores::cont::ArrayHandle<viskores::Id> index_array;
  const viskores::Id logical_size = static_cast<viskores::Id>(ni * nj * nk);
  index_array.Allocate(logical_size);
  auto portal = index_array.WritePortal();

  viskores::Id out_idx = 0;
  viskores::Id max_idx = 0;
  for(index_t k = 0; k < nk; ++k)
  {
    for(index_t j = 0; j < nj; ++j)
    {
      for(index_t i = 0; i < ni; ++i)
      {
        const viskores::Id src_idx = static_cast<viskores::Id>((ok + k) * sk +
                                                               (oj + j) * sj +
                                                               (oi + i) * si) *
                                     static_cast<viskores::Id>(element_stride);
        portal.Set(out_idx++, src_idx);
        max_idx = std::max(max_idx, src_idx);
      }
    }
  }

  source_size = logical_size == 0 ? 0 : max_idx + 1;
  return index_array;
}

// Wrap strided storage as a logical Viskores permutation array.
template<typename T>
auto
GetPermutedArray(const conduit::Node &node,
                 const std::vector<index_t> &logical_dims,
                 const std::vector<index_t> &offsets,
                 const std::vector<index_t> &strides,
                 const index_t element_stride,
                 const bool zero_copy)
  -> decltype(viskores::cont::make_ArrayHandlePermutation(
      std::declval<viskores::cont::ArrayHandle<viskores::Id> >(),
      std::declval<viskores::cont::ArrayHandle<T> >()))
{
  viskores::CopyFlag copy = zero_copy ? viskores::CopyFlag::Off
                                      : viskores::CopyFlag::On;
  viskores::Id source_size = 0;
  viskores::cont::ArrayHandle<viskores::Id> index_array = LogicalIndexArray(logical_dims,
                                                                            offsets,
                                                                            strides,
                                                                            element_stride,
                                                                            source_size);

  const T *values_ptr = node.value();
  viskores::cont::ArrayHandle<T> source_array = viskores::cont::make_ArrayHandle(values_ptr,
                                                                                 source_size,
                                                                                 copy);

  return viskores::cont::make_ArrayHandlePermutation(index_array, source_array);
}


template<typename T>
void
BlueprintIndexArrayToViskoresIdArray(const conduit::Node &n,
                                 bool zero_copy,
                                 viskores::cont::ArrayHandle<T> &viskores_handle)
{
    int array_size = n.dtype().number_of_elements();

    if( sizeof(T) == 1 ) // uint8 is what viskores will use for this case.
    {
        if(n.is_compact() && n.dtype().is_uint8())
        {
            // directly compatible
            const void *idx_ptr = n.data_ptr();
            CopyArray(viskores_handle, (const T*)idx_ptr, array_size,zero_copy);
        }
        else
        {
            // we need to convert to uint8 to match viskores::Id
            viskores_handle.Allocate(array_size);
            void *ptr = (void*) vtkh::GetVISKORESPointer(viskores_handle);
            Node n_tmp;
            n_tmp.set_external(DataType::uint8(array_size),ptr);
            n.to_uint8_array(n_tmp);
        }
    }
    else if( sizeof(T) == 2)
    {
        // unsupported!
        ASCENT_ERROR("BlueprintIndexArrayToViskoresIdArray does not support 2-byte index arrays");
    }
    else if( sizeof(T) == 4) // int32 is what viskores will use for this case.
    {
        if(n.is_compact() && n.dtype().is_int32())
        {
            // directly compatible
            const void *idx_ptr = n.data_ptr();
            CopyArray(viskores_handle, (const T*)idx_ptr, array_size,zero_copy);
        }
        else
        {
            // we need to convert to int32 to match viskores::Id
            viskores_handle.Allocate(array_size);
            void *ptr = (void*) vtkh::GetVISKORESPointer(viskores_handle);
            Node n_tmp;
            n_tmp.set_external(DataType::int32(array_size),ptr);
            n.to_int32_array(n_tmp);
        }
    }
    else if( sizeof(T) == 8) // int64 is what viskores will use for this case.
    {
        if(n.is_compact() && n.dtype().is_int64())
        {
            // directly compatible
            const void *idx_ptr = n.data_ptr();
            CopyArray(viskores_handle, (const T*)idx_ptr, array_size, zero_copy);
        }
        else
        {
            // we need to convert to int64 to match viskores::Id
            viskores_handle.Allocate(array_size);
            void *ptr = (void*) vtkh::GetVISKORESPointer(viskores_handle);
            Node n_tmp;
            n_tmp.set_external(DataType::int64(array_size),ptr);
            n.to_int64_array(n_tmp);
        }
    }
}


template<typename T>
viskores::cont::CoordinateSystem
GetExplicitCoordinateSystem(const conduit::Node &n_coords,
                            const std::string &name,
                            int &ndims,
                            index_t &x_element_stride,
                            index_t &y_element_stride,
                            index_t &z_element_stride,
                            bool zero_copy)
{
    viskores::CopyFlag copy = viskores::CopyFlag::On;
    if(zero_copy)
    {
      copy = viskores::CopyFlag::Off;
    }
      
    int nverts = n_coords["values/x"].dtype().number_of_elements();
    //bool is_interleaved = blueprint::mcarray::is_interleaved(n_coords["values"]);

    // some interleaved cases aren't working
    // disabling this path until we find out what is going wrong.
    //is_interleaved = false;

    viskores::cont::ArrayHandle<T> x_coords_handle;
    viskores::cont::ArrayHandle<T> y_coords_handle;
    viskores::cont::ArrayHandle<T> z_coords_handle;

    ndims = 2;

    if(x_element_stride == 1)
    {
      const T *x_verts_ptr = n_coords["values/x"].value();
      detail::CopyArray(x_coords_handle, x_verts_ptr, nverts, zero_copy);
    }
    else
    {
      int x_verts_expanded = (nverts - 1) * x_element_stride + 1;
      const T *x_verts_ptr = n_coords["values/x"].value();
      viskores::cont::ArrayHandle<T> x_source_array = viskores::cont::make_ArrayHandle<T>(x_verts_ptr,
                                                                                  x_verts_expanded,
                                                                                  copy);
      viskores::cont::ArrayHandleStride<T> x_stride_handle(x_source_array,
                                                       nverts,
                                                       x_element_stride,
                                                       0); // offset

      viskores::cont::Algorithm::Copy(x_stride_handle, x_coords_handle);
    }

    if(y_element_stride == 1)
    {
      const T *y_verts_ptr = n_coords["values/y"].value();
      detail::CopyArray(y_coords_handle, y_verts_ptr, nverts, zero_copy);
    }
    else
    {
      int y_verts_expanded = (nverts - 1) * y_element_stride + 1;
      const T *y_verts_ptr = n_coords["values/y"].value();
      viskores::cont::ArrayHandle<T> y_source_array = viskores::cont::make_ArrayHandle<T>(y_verts_ptr,
                                                                                  y_verts_expanded,
                                                                                  copy);
      viskores::cont::ArrayHandleStride<T> y_stride_handle(y_source_array,
                                                       nverts,
                                                       y_element_stride,
                                                       0); // offset

      viskores::cont::Algorithm::Copy(y_stride_handle, y_coords_handle);
    }

    if(z_element_stride == 0)
    {
      z_coords_handle.AllocateAndFill(nverts,0.0);
      T *z = vtkh::GetVISKORESPointer(z_coords_handle);
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
      int z_verts_expanded = (nverts - 1) * z_element_stride + 1;
      const T *z_verts_ptr = n_coords["values/z"].value();
      viskores::cont::ArrayHandle<T> z_source_array = viskores::cont::make_ArrayHandle<T>(z_verts_ptr,
                                                                                  z_verts_expanded,
                                                                                  copy);
      viskores::cont::ArrayHandleStride<T> z_stride_handle(z_source_array,
                                                       nverts,
                                                       z_element_stride,
                                                       0); // offset

      viskores::cont::Algorithm::Copy(z_stride_handle, z_coords_handle);
    }

    return viskores::cont::CoordinateSystem(name,
                                        make_ArrayHandleSOA(x_coords_handle,
                                                            y_coords_handle,
                                                            z_coords_handle));

}

// Build explicit structured coordinates from strided Blueprint topology metadata.
template<typename T>
viskores::cont::CoordinateSystem
GetStructuredExplicitCoordinateSystem(const conduit::Node &n_coords,
                                      const conduit::Node &n_topo,
                                      const std::string &name,
                                      int &ndims,
                                      int &nverts,
                                      bool zero_copy)
{
  const std::vector<index_t> point_dims = LogicalDims(n_topo, "vertex");
  const std::vector<index_t> offsets = n_topo.has_path("elements/dims/offsets")
                                       ? IndexVector(n_topo["elements/dims/offsets"])
                                       : std::vector<index_t>(point_dims.size(), 0);
  const std::vector<index_t> strides = n_topo.has_path("elements/dims/strides")
                                       ? IndexVector(n_topo["elements/dims/strides"])
                                       : std::vector<index_t>();

  nverts = 1;
  for(size_t i = 0; i < point_dims.size(); ++i)
  {
    nverts *= static_cast<int>(point_dims[i]);
  }
  ndims = point_dims.size() == 3 ? 3 : 2;

  index_t x_element_stride = n_coords["values/x"].dtype().stride() / sizeof(T);
  index_t y_element_stride = n_coords["values/y"].dtype().stride() / sizeof(T);
  auto x_coords_handle = GetPermutedArray<T>(n_coords["values/x"],
                                             point_dims,
                                             offsets,
                                             strides,
                                             x_element_stride,
                                             zero_copy);
  auto y_coords_handle = GetPermutedArray<T>(n_coords["values/y"],
                                             point_dims,
                                             offsets,
                                             strides,
                                             y_element_stride,
                                             zero_copy);

  if(n_coords.has_path("values/z"))
  {
    index_t z_element_stride = n_coords["values/z"].dtype().stride() / sizeof(T);
    auto z_coords_handle = GetPermutedArray<T>(n_coords["values/z"],
                                               point_dims,
                                               offsets,
                                               strides,
                                               z_element_stride,
                                               zero_copy);
    return viskores::cont::CoordinateSystem(name,viskores::cont::make_ArrayHandleCompositeVector(x_coords_handle,y_coords_handle,z_coords_handle));
  }

  viskores::cont::ArrayHandle<T> z_coords_handle;
  z_coords_handle.AllocateAndFill(nverts, 0.0);
  return viskores::cont::CoordinateSystem(name,viskores::cont::make_ArrayHandleCompositeVector(x_coords_handle,y_coords_handle,z_coords_handle));
}

template<typename T>
viskores::cont::CoordinateSystem
GetRZCoordinateSystem(const conduit::Node &n_coords,
                            const std::string &name,
                            int &ndims,
                            index_t &r_element_stride,
                            index_t &z_element_stride,
                            bool zero_copy)
{
    viskores::CopyFlag copy = viskores::CopyFlag::On;
    if(zero_copy)
    {
      copy = viskores::CopyFlag::Off;
    }
      
    int nverts = n_coords["values/r"].dtype().number_of_elements();

    viskores::cont::ArrayHandle<T> r_coords_handle;
    viskores::cont::ArrayHandle<T> z_coords_handle;
    viskores::cont::ArrayHandle<T> theta_coords_handle;

    ndims = 2;

    if(r_element_stride == 1)
    {
      const T *r_verts_ptr = n_coords["values/r"].value();
      detail::CopyArray(r_coords_handle, r_verts_ptr, nverts, zero_copy);
    }
    else
    {
      int r_verts_expanded = (nverts - 1) * r_element_stride + 1;
      const T *r_verts_ptr = n_coords["values/r"].value();
      viskores::cont::ArrayHandle<T> r_source_array = viskores::cont::make_ArrayHandle<T>(r_verts_ptr,
                                                                                  r_verts_expanded,
                                                                                  copy);
      viskores::cont::ArrayHandleStride<T> r_stride_handle(r_source_array,
                                                       nverts,
                                                       r_element_stride,
                                                       0); // offset

      viskores::cont::Algorithm::Copy(r_stride_handle, r_coords_handle);
    }

    if(z_element_stride == 1)
    {
      const T *z_verts_ptr = n_coords["values/z"].value();
      detail::CopyArray(z_coords_handle, z_verts_ptr, nverts, zero_copy);
    }
    else
    {
      int z_verts_expanded = (nverts - 1) * z_element_stride + 1;
      const T *z_verts_ptr = n_coords["values/z"].value();
      viskores::cont::ArrayHandle<T> z_source_array = viskores::cont::make_ArrayHandle<T>(z_verts_ptr,
                                                                                  z_verts_expanded,
                                                                                  copy);
      viskores::cont::ArrayHandleStride<T> z_stride_handle(z_source_array,
                                                       nverts,
                                                       z_element_stride,
                                                       0); // offset

      viskores::cont::Algorithm::Copy(z_stride_handle, z_coords_handle);
    }

    theta_coords_handle.AllocateAndFill(nverts,0.0);

    return viskores::cont::CoordinateSystem(name,
                                    make_ArrayHandleSOA(z_coords_handle,
                                                        r_coords_handle,
                                                        theta_coords_handle));
}


template<typename T>
viskores::cont::Field GetField(const conduit::Node &node,
                           const std::string &field_name,
                           const std::string &assoc_str,
                           const std::string &topo_str,
                           index_t element_stride,
                           bool zero_copy)
{
  viskores::CopyFlag copy = viskores::CopyFlag::On;
  if(zero_copy)
  {
    copy = viskores::CopyFlag::Off;
  }
  viskores::cont::Field::Association viskores_assoc = viskores::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    viskores_assoc = viskores::cont::Field::Association::Points;
  }
  else if(assoc_str == "element")
  {
    viskores_assoc = viskores::cont::Field::Association::Cells;
  }
  else if(assoc_str == "whole")
  {
    viskores_assoc = viskores::cont::Field::Association::WholeDataSet;
  }
  else
  {
    ASCENT_ERROR("Cannot add field association "<<assoc_str<<" from field "<<field_name);
  }

  int num_vals = node.dtype().number_of_elements();

  const T *values_ptr = node.value();

  viskores::cont::Field field;
  // base case is naturally stride data
  if(element_stride == 1)
  {
      field = viskores::cont::make_Field(field_name,
                                     viskores_assoc,
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

      int num_vals_expanded = (num_vals - 1) * element_stride + 1;
      viskores::cont::ArrayHandle<T> source_array = viskores::cont::make_ArrayHandle(values_ptr,
                                                                             num_vals_expanded,
                                                                             copy);
      viskores::cont::ArrayHandleStride<T> stride_array(source_array,
                                                    num_vals,
                                                    element_stride,
                                                    0);
      field =  viskores::cont::Field(field_name,
                                 viskores_assoc,
                                 stride_array);
  }

  return field;
}

// Build a field from strided Blueprint values.
template<typename T>
viskores::cont::Field GetStridedField(const conduit::Node &node,
                                      const std::string &field_name,
                                      const std::string &assoc_str,
                                      const std::vector<index_t> &logical_dims,
                                      const std::vector<index_t> &offsets,
                                      const std::vector<index_t> &strides,
                                      const index_t element_stride,
                                      const bool zero_copy)
{
  viskores::cont::Field::Association viskores_assoc = viskores::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    viskores_assoc = viskores::cont::Field::Association::Points;
  }
  else if(assoc_str == "element")
  {
    viskores_assoc = viskores::cont::Field::Association::Cells;
  }
  else
  {
    ASCENT_ERROR("Cannot add field association "<<assoc_str<<" from field "<<field_name);
  }

  auto field_array = GetPermutedArray<T>(node,
                                         logical_dims,
                                         offsets,
                                         strides,
                                         element_stride,
                                         zero_copy);
  return viskores::cont::Field(field_name, viskores_assoc, field_array);
}


template<typename T>
viskores::cont::Field GetVectorField(T *values_ptr,
                                 const int num_vals,
                                 const std::string &field_name,
                                 const std::string &assoc_str,
                                 const std::string &topo_str,
                                 bool zero_copy)
{
  viskores::CopyFlag copy = viskores::CopyFlag::On;
  if(zero_copy)
  {
    copy = viskores::CopyFlag::Off;
  }
  viskores::cont::Field::Association viskores_assoc = viskores::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    viskores_assoc = viskores::cont::Field::Association::Points;
  }
  else if(assoc_str == "element")
  {
    viskores_assoc = viskores::cont::Field::Association::Cells;
  }
  else
  {
    ASCENT_ERROR("Cannot add vector field with association "
                 <<assoc_str<<" field_name "<<field_name);
  }

  viskores::cont::Field field;
  field = viskores::cont::make_Field(field_name,
                                 viskores_assoc,
                                 values_ptr,
                                 num_vals,
                                 copy);

  return field;
}

//
// extract a vector from 3 separate arrays
//
template<typename T>
void ExtractVector(viskores::cont::DataSet *dset,
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

  viskores::cont::Field::Association viskores_assoc = viskores::cont::Field::Association::Any;
  if(assoc_str == "vertex")
  {
    viskores_assoc = viskores::cont::Field::Association::Points;
  }
  else if (assoc_str == "element")
  {
    viskores_assoc = viskores::cont::Field::Association::Cells;
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

    viskores::cont::ArrayHandle<T> x_handle;
    viskores::cont::ArrayHandle<T> y_handle;
    detail::CopyArray(x_handle, x_ptr, num_vals, zero_copy);
    detail::CopyArray(y_handle, y_ptr, num_vals, zero_copy);

    auto composite = make_ArrayHandleSOA(x_handle,
                                         y_handle);

    viskores::cont::Field field(field_name, viskores_assoc, composite);
    dset->AddField(field);
  }

  if(dims == 3)
  {
    const T *x_ptr = GetNodePointer<T>(u);
    const T *y_ptr = GetNodePointer<T>(v);
    const T *z_ptr = GetNodePointer<T>(w);

    viskores::cont::ArrayHandle<T> x_handle;
    viskores::cont::ArrayHandle<T> y_handle;
    viskores::cont::ArrayHandle<T> z_handle;
    detail::CopyArray(x_handle, x_ptr, num_vals, zero_copy);
    detail::CopyArray(y_handle, y_ptr, num_vals, zero_copy);
    detail::CopyArray(z_handle, z_ptr, num_vals, zero_copy);

    auto composite = make_ArrayHandleSOA(x_handle,
                                         y_handle,
                                         z_handle);

    viskores::cont::Field field(field_name, viskores_assoc, composite);
    dset->AddField(field);
  }
}


void ViskoresCellShape(const std::string &shape_type,
                   viskores::UInt8 &shape_id,
                   viskores::IdComponent &num_indices)
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
                           viskores::cont::Field &length,
                           viskores::cont::Field &offsets,
                           viskores::cont::Field &ids,
                           viskores::cont::Field &vfs)
{
  viskores::CopyFlag copy = viskores::CopyFlag::On;

  viskores::cont::Field::Association viskores_assoc_c = viskores::cont::Field::Association::Cells;

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
    // Check whether the provided volume fractions leave implicit background material.
    const S vf_tolerance = static_cast<S>(1e-6);
    const S vf_dust = static_cast<S>(1e-4);
    const S one = static_cast<S>(1);
    std::vector<S> v_sums(neles, static_cast<S>(0));
    std::vector<S> v_kept_sums(neles, static_cast<S>(0));

    NodeConstIterator itr = node["volume_fractions"].children();
    std::string material_name;
    while(itr.has_next())
    {

      const conduit::Node * n_material;
      const conduit::Node &n_next = itr.next();
      // n_next is not leaf i.e. has values: [v0,v1,...,vn]
      if(n_next.number_of_children() != 0)
      {
        n_material = &n_next.child(0);
      }
      else
      {
        n_material = &n_next;
      }
      const S *data = n_material->value();
      for(index_t i = 0; i < neles; ++i)
      {
        if(data[i] > static_cast<S>(0))
        {
          v_sums[i] += data[i];
        }
      }
    }

    // If fractions do not sum to one, add a background material for the remainder.
    bool add_implicit_background = false;
    for(index_t i = 0; i < neles; ++i)
    {
      if(v_sums[i] < one - vf_tolerance)
      {
        add_implicit_background = true;
        break;
      }
    }

    itr = node["volume_fractions"].children();
    while(itr.has_next())
    {

      const conduit::Node * n_material;
      const conduit::Node &n_next = itr.next();
      // n_next is not leaf i.e. has values: [v0,v1,...,vn]
      if(n_next.number_of_children() != 0)
      {
        n_material = &n_next.child(0);
      }
      else
      {
        n_material = &n_next;
      }
      const S *data = n_material->value();
      // increase length when a material vf value > 0
      for(index_t i = 0; i < neles; ++i)
      {
        if(data[i] > (add_implicit_background ? vf_dust : static_cast<S>(0)))
        {
          v_length[i] += 1;
          v_kept_sums[i] += data[i];
        }
      }
    }

    if(add_implicit_background)
    {
      for(index_t i = 0; i < neles; ++i)
      {
        if(v_kept_sums[i] < one - vf_dust)
        {
          v_length[i] += 1;
        }
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

  length = viskores::cont::make_Field(length_name,
                                 viskores_assoc_c,
                                 length_ptr,
                                 neles,
                                 copy);

  const T *offsets_ptr = v_offsets.data();

  offsets = viskores::cont::make_Field(offsets_name,
                                 viskores_assoc_c,
                                 offsets_ptr,
                                 neles,
                                 copy);
  //calc vfs and mat ids
  viskores::cont::Field::Association viskores_assoc_w = viskores::cont::Field::Association::WholeDataSet;
  std::vector<T> v_ids(l_total,0);
  std::vector<S> v_vfs(l_total,0);

  if(node.has_child("element_ids"))
  {

    int num_materials = node["element_ids"].number_of_children();
    const Node &n_vol_fracs = node["volume_fractions"];
    const Node &n_ele_ids = node["element_ids"];

    for(index_t i = 0; i < num_materials; ++i)
    {
      const conduit::Node * n_vol_frac;	    
      const conduit::Node &n_child = n_vol_fracs.child(i);
      // n_child is not leaf i.e. has values: [v0,v1,...,vn]
      if(n_child.number_of_children() != 0)
      {
        n_vol_frac = &n_child.child(0);
      }
      else
        n_vol_frac = &n_child;	      
      const Node &n_ele_id = n_ele_ids.child(i);
      const S *vf_data = n_vol_frac->value();
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
    // Check whether the provided volume fractions leave implicit background material.
    const S vf_tolerance = static_cast<S>(1e-6);
    const S vf_dust = static_cast<S>(1e-4);
    const S one = static_cast<S>(1);
    std::vector<S> v_sums(neles, static_cast<S>(0));
    std::vector<S> v_kept_sums(neles, static_cast<S>(0));
    bool add_implicit_background = false;

    for(index_t i = 0; i < num_materials; ++i)
    {
      const Node &n_materials = node["volume_fractions"];
      const Node &n_child = n_materials.child(i);

      const Node * n_material;
      // n_child is not leaf i.e. has values: [v0,v1,...,vn]
      if(n_child.number_of_children() != 0)
      {
        n_material = &n_child.child(0);
      }
      else
      {
        n_material = &n_child;
      }

      const S *data = n_material->value();

      for(index_t j = 0; j < neles; ++j)
      {
        if(data[j] > static_cast<S>(0))
        {
          v_sums[j] += data[j];
        }
      }
    }

    // If fractions do not sum to one, add a background material for the remainder.
    for(index_t j = 0; j < neles; ++j)
    {
      if(v_sums[j] < one - vf_tolerance)
      {
        add_implicit_background = true;
        break;
      }
    }

    for(index_t i = 0; i < num_materials; ++i)
    {
      const Node &n_materials = node["volume_fractions"];
      const Node &n_child = n_materials.child(i);

      const Node * n_material;
      // n_child is not leaf i.e. has values: [v0,v1,...,vn]
      if(n_child.number_of_children() != 0)
      {
        n_material = &n_child.child(0);
      }
      else
      {
        n_material = &n_child;
      }

      const S *data = n_material->value();

      for(index_t j = 0; j < neles; ++j)
      {
        index_t offset = v_offsets[j];
        if(data[j] > (add_implicit_background ? vf_dust : static_cast<S>(0)))
        {
          v_length[j] -= 1;
          index_t length = v_length[j];
          v_ids[offset + length] = i + 1; //IDs cannot start at 0
          v_vfs[offset + length] = data[j];
          v_kept_sums[j] += data[j];
        }
      }
    }

    if(add_implicit_background)
    {
      for(index_t j = 0; j < neles; ++j)
      {
        if(v_kept_sums[j] < one - vf_dust)
        {
          index_t offset = v_offsets[j];
          v_length[j] -= 1;
          index_t length = v_length[j];
          v_ids[offset + length] = num_materials + 1; //implicit background material
          v_vfs[offset + length] = one - v_kept_sums[j];
        }
      }
    }
  }

  const T *ids_ptr = v_ids.data();

  ids = viskores::cont::make_Field(ids_name,
                               viskores_assoc_w,
                               ids_ptr,
                               l_total,
                               copy);

  const S *vfs_ptr = v_vfs.data();

  vfs = viskores::cont::make_Field(vfs_name,
                               viskores_assoc_w,
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
//                           viskores::cont::Field &offsets,
//{
//  viskores::CopyFlag copy = viskores::CopyFlag::On;
//
//  viskores::cont::ArrayHandle<int> ah_offsets;
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
//  ids = viskores::cont::make_Field(ids_name,
//                               viskores_assoc,
//                               ids_ptr,
//                               total,
//                               copy);
//
//  const S *vfs_ptr = v_vfs.data();
//
//  vfs = viskores::cont::make_Field(vfs_name,
//                               viskores_assoc,
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

namespace
{

bool
material_volume_fraction_name(const std::string &field_name,
                              std::string &material_name)
{
  // Convert supported VisIt volume fraction field names into material names.
  const std::string visit_prefix = "volume_fraction_";
  if(field_name.rfind(visit_prefix, 0) == 0 &&
     field_name.size() > visit_prefix.size())
  {
    const std::string suffix = field_name.substr(visit_prefix.size());
    for(size_t i = 0; i < suffix.size(); ++i)
    {
      if(!std::isdigit(static_cast<unsigned char>(suffix[i])))
      {
        return false;
      }
    }

    material_name = "material_" + suffix;
    return true;
  }

  const std::string axom_prefix = "vol_frac_";
  if(field_name.rfind(axom_prefix, 0) == 0 &&
     field_name.size() > axom_prefix.size())
  {
    material_name = field_name.substr(axom_prefix.size());
    return true;
  }

  return false;
}

const conduit::Node *
scalar_field_values(const conduit::Node &field)
{
  // Return scalar values, allowing either direct values or one named component.
  if(!field.has_child("values"))
  {
    return NULL;
  }

  const conduit::Node &values = field["values"];
  if(values.number_of_children() == 0)
  {
    return &values;
  }

  if(values.number_of_children() == 1)
  {
    return &values.child(0);
  }

  return NULL;
}

bool
build_visit_style_matset(const conduit::Node &node,
                         const std::string &topo_name,
                         int neles,
                         conduit::Node &matsets)
{
  // Build a temporary matset from VisIt volume fraction fields.
  if(!node.has_child("fields"))
  {
    return false;
  }

  std::vector<std::pair<std::string, const conduit::Node *> > materials;
  conduit::NodeConstIterator itr = node["fields"].children();
  while(itr.has_next())
  {
    const conduit::Node &field = itr.next();
    std::string material_name;
    if(!material_volume_fraction_name(itr.name(), material_name))
    {
      continue;
    }

    if(!field.has_child("topology") ||
       field["topology"].as_string() != topo_name)
    {
      continue;
    }

    if(field.has_child("association") &&
       field["association"].as_string() != "element")
    {
      continue;
    }

    const conduit::Node *values = scalar_field_values(field);
    if(values == NULL)
    {
      continue;
    }

    if(!values->dtype().is_float32() && !values->dtype().is_float64())
    {
      continue;
    }

    if(values->dtype().number_of_elements() != neles)
    {
      continue;
    }

    materials.push_back(std::make_pair(material_name, values));
  }

  if(materials.empty())
  {
    return false;
  }

  std::sort(materials.begin(), materials.end());

  conduit::Node &matset = matsets["materials"];
  matset["topology"] = topo_name;
  for(size_t i = 0; i < materials.size(); ++i)
  {
    matset["volume_fractions"][materials[i].first].set_external(*materials[i].second);
  }

  return true;
}

} // namespace

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
    viskores::UInt64 cycle = 0;
    double time = 0;
    std::vector<viskores::UInt64> allCycles;
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
        viskores::cont::DataSet *dset = BlueprintToViskoresDataSet(dom, zero_copy, topo_name);
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
      viskores::cont::DataSet *dset = VTKHDataAdapter::BlueprintToViskoresDataSet(dom,
                                                                          zero_copy,
                                                                          topo_name);
      int domain_id = dom["state/domain_id"].to_int();

      if(dom.has_path("state/cycle"))
      {
        viskores::UInt64 cycle = dom["state/cycle"].to_uint64();
        res->SetCycle(cycle);
      }

      res->AddDomain(*dset,domain_id);
      // viskores will shallow copy the data assoced with dset
      // clean up our copy
      delete dset;

    }
    return res;
}

//-----------------------------------------------------------------------------
vtkh::DataSet *
VTKHDataAdapter::ViskoresDataSetToVTKHDataSet(viskores::cont::DataSet *dset)
{
    // wrap a single Viskores data set into a VTKH dataset
    vtkh::DataSet   *res = new  vtkh::DataSet;
    int domain_id = 0; // TODO, MPI_TASK_ID ?
    res->AddDomain(*dset,domain_id);
    return res;
}

//-----------------------------------------------------------------------------
viskores::cont::DataSet *
VTKHDataAdapter::BlueprintToViskoresDataSet(const Node &node,
                                        bool zero_copy,
                                        const std::string &topo_name_str)
{
    viskores::cont::DataSet * result = NULL;

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
        result = UniformBlueprintToViskoresDataSet(coords_name,
                                               n_coords,
                                               topo_name,
                                               n_topo,
                                               neles,
                                               nverts);
    }
    else if(mesh_type == "rectilinear")
    {
        result = RectilinearBlueprintToViskoresDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts,
                                                   zero_copy);

    }
    else if(mesh_type == "structured")
    {
        result =  StructuredBlueprintToViskoresDataSet(coords_name,
                                                   n_coords,
                                                   topo_name,
                                                   n_topo,
                                                   neles,
                                                   nverts,
                                                   zero_copy);
    }
    else if( mesh_type ==  "points")
    {
        result =  PointsImplicitBlueprintToViskoresDataSet(coords_name,
                                                       n_coords,
                                                       topo_name,
                                                       n_topo,
                                                       neles,
                                                       nverts,
                                                       zero_copy);
    }
    else if( mesh_type ==  "unstructured")
    {
        result =  UnstructuredBlueprintToViskoresDataSet(coords_name,
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
                         n_topo,
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

    conduit::Node visit_style_matsets;
    const conduit::Node *matsets = NULL;
    // Use explicit matsets when present; otherwise try to synthesize one from fields.
    if(node.has_child("matsets"))
    {
        matsets = &node["matsets"];
    }
    else if(build_visit_style_matset(node, topo_name, neles, visit_style_matsets))
    {
        matsets = &visit_style_matsets;
    }

    if(matsets != NULL)
    {
        // add all of the materials:
        NodeConstIterator itr = matsets->children();
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
// Helper function to create explicit coordinate arrays for viskores data sets
void CreateExplicitArrays(viskores::cont::ArrayHandle<viskores::UInt8> &shapes,
                          viskores::cont::ArrayHandle<viskores::IdComponent> &num_indices,
                          const std::string &shape_type,
                          const viskores::Id &conn_size,
                          viskores::IdComponent &dimensionality,
                          int &neles)
{
    viskores::UInt8 shape_id = 0;
    viskores::IdComponent indices= 0;
    if(shape_type == "tri")
    {
        shape_id = 3;
        indices = 3;
        // note: viskores cell dimensions are topological
        dimensionality = 2;
    }
    else if(shape_type == "quad")
    {
        shape_id = 9;
        indices = 4;
        // note: viskores cell dimensions are topological
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

    const viskores::Id num_shapes = conn_size / indices;

    neles = num_shapes;

    shapes.Allocate(num_shapes);
    num_indices.Allocate(num_shapes);

    // We could memset these and then zero copy them but that
    // would make us responsible for the data. If we just create
    // them, smart pointers will automatically delete them.
    // Hopefull the compiler turns this into a memset.

    const viskores::UInt8 shape_value = shape_id;
    const viskores::IdComponent indices_value = indices;
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

viskores::cont::DataSet *
VTKHDataAdapter::UniformBlueprintToViskoresDataSet
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

    //Create implicit viskores coordinate system
    viskores::cont::DataSet *result = new viskores::cont::DataSet();

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

    const bool is_rz = (n_coords.has_child("origin") && (n_coords["origin"].has_child("r") || (n_coords["origin"].has_child("z") && is_2d))) ||
                 (n_coords.has_child("spacing") && (n_coords["spacing"].has_child("dr") || (n_coords["spacing"].has_child("dz") && is_2d)));
    const bool is_cartesian = (n_coords.has_child("origin") && (n_coords["origin"].has_child("x") || n_coords["origin"].has_child("y"))) ||
                          (n_coords.has_child("spacing") && (n_coords["spacing"].has_child("dx") || n_coords["spacing"].has_child("dy"))) ||
                          !is_rz;

    if (is_rz && is_cartesian) {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z} but got parameters for both.")
    }
    
    if (!is_rz && !is_cartesian)
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z} but got neither.")
    }

    if(n_coords.has_child("origin"))
    {
        const Node &n_origin = n_coords["origin"];

        if (is_cartesian)
        {
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
        else if (is_rz && is_2d)
        {
            if(n_origin.has_child("z"))
            {
                origin_x = n_origin["z"].to_float64();
            }

            if(n_origin.has_child("r"))
            {
                origin_y = n_origin["r"].to_float64();
            }
        }
        else if (is_rz && !is_2d)
        {
            ASCENT_ERROR("Unsupported coordset: cylindrical {r,z} coordinates only supported in 2d.")
        }
    }

    if(n_coords.has_path("spacing"))
    {
        const Node &n_spacing = n_coords["spacing"];

        if (is_cartesian)
        {
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
        else if (is_rz && is_2d)
        {
            if(n_spacing.has_path("dz"))
            {
                spacing_x = n_spacing["dz"].to_float64();
            }

            if(n_spacing.has_path("dr"))
            {
                spacing_y = n_spacing["dr"].to_float64();
            }
        }
        else if (is_rz && !is_2d)
        {
            ASCENT_ERROR("Unsupported coordset: cylindrical {r,z} coordinates only supported in 2d.")
        }
    }

    // todo, should this be float64 -- or should we read float32 above?

    viskores::Vec<viskores::Float32,3> origin(origin_x,
                                      origin_y,
                                      origin_z);

    viskores::Vec<viskores::Float32,3> spacing(spacing_x,
                                       spacing_y,
                                       spacing_z);

    viskores::Id3 dims;
    if(is_rz)
    {
        dims = viskores::Id3(dims_j,
                    dims_i,
                    dims_k);
    }
    else
    {
        dims = viskores::Id3(dims_i,
                    dims_j,
                    dims_k);
    }

    // todo, use actually coordset and topo names?
    result->AddCoordinateSystem( viskores::cont::CoordinateSystem(coords_name.c_str(),
                                                              dims,
                                                              origin,
                                                              spacing));

    viskores::Id3 topo_origin = detail::topo_origin(n_topo);
    if(is_2d)
    {
      viskores::Id2 dims2(dims[0], dims[1]);
      viskores::cont::CellSetStructured<2> cell_set;
      cell_set.SetPointDimensions(dims2);
      viskores::Id2 origin2(topo_origin[0], topo_origin[1]);
      cell_set.SetGlobalPointIndexStart(origin2);
      result->SetCellSet(cell_set);
    }
    else
    {
      viskores::cont::CellSetStructured<3> cell_set;
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

viskores::cont::DataSet *
VTKHDataAdapter::RectilinearBlueprintToViskoresDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts,                    // output, number of verts
     bool zero_copy)                 // attempt to zero copy
{
    viskores::cont::DataSet *result = new viskores::cont::DataSet();

    const bool is_rz = n_coords["values"].has_child("r") && n_coords["values"].has_child("z");
    const bool is_cartesian = n_coords["values"].has_child("x") && n_coords["values"].has_child("y");

    if (is_rz && is_cartesian)
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z} but got parameters for both.")
    }
    
    if (!is_rz && !is_cartesian)
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z} but got neither.")
    }

    if (is_cartesian)
    {
        if (zero_copy &&
            (!n_coords["values/x"].dtype().is_float64() || 
             !n_coords["values/y"].dtype().is_float64()))
        {
            ASCENT_INFO("Zero-copy requested, but either x or y coordinate data is not float64." <<
                        "x type: " << n_coords["values/x"].dtype().name() << 
                        ", y type: " << n_coords["values/y"].dtype().name() << 
                        ". Turning zero-copy off.");
            zero_copy = false;
        }

        int x_npts = n_coords["values/x"].dtype().number_of_elements();
        int y_npts = n_coords["values/y"].dtype().number_of_elements();
        int z_npts = 0;

        const float64 *x_coords_ptr;
        Node temp_x;
        if (n_coords["values/x"].dtype().is_float64())
        {
            x_coords_ptr = n_coords["values/x"].as_float64_ptr();
        }
        else
        {
            n_coords["values/x"].to_float64_array(temp_x);
            x_coords_ptr = temp_x.value();
        }
        
        const float64 *y_coords_ptr;
        Node temp_y;
        if (n_coords["values/y"].dtype().is_float64())
        {
            y_coords_ptr = n_coords["values/y"].as_float64_ptr();
        }
        else
        {
            n_coords["values/y"].to_float64_array(temp_y);
            y_coords_ptr = temp_y.value();
        }

        int32 ndims = 2;
        const float64 *z_coords_ptr = NULL;
        Node temp_z;
        if(n_coords.has_path("values/z"))
        {
            if (zero_copy && !n_coords["values/z"].dtype().is_float64())
            {
                ASCENT_INFO("Zero-copy requested, but z coordinate data is " <<
                            n_coords["values/z"].dtype().name() <<
                            " not float64. Turning zero-copy off.");
                zero_copy = false;
            }
            ndims = 3;
            z_npts = n_coords["values/z"].dtype().number_of_elements();
            if (n_coords["values/z"].dtype().is_float64())
            {
                z_coords_ptr = n_coords["values/z"].as_float64_ptr();
            }
            else
            {
                n_coords["values/z"].to_float64_array(temp_z);
                z_coords_ptr = temp_z.value();
            }
        }

        viskores::cont::ArrayHandle<viskores::Float64> x_coords_handle;
        viskores::cont::ArrayHandle<viskores::Float64> y_coords_handle;
        viskores::cont::ArrayHandle<viskores::Float64> z_coords_handle;

        if(zero_copy)
        {
            x_coords_handle = viskores::cont::make_ArrayHandle(x_coords_ptr, x_npts, viskores::CopyFlag::Off);
            y_coords_handle = viskores::cont::make_ArrayHandle(y_coords_ptr, y_npts, viskores::CopyFlag::Off);
        }
        else
        {
            x_coords_handle.Allocate(x_npts);
            y_coords_handle.Allocate(y_npts);

            viskores::Float64 *x = vtkh::GetVISKORESPointer(x_coords_handle);
            memcpy(x, x_coords_ptr, sizeof(float64) * x_npts);
            viskores::Float64 *y = vtkh::GetVISKORESPointer(y_coords_handle);
            memcpy(y, y_coords_ptr, sizeof(float64) * y_npts);
        }

        if(ndims == 3)
        {
            if(zero_copy)
            {
                z_coords_handle = viskores::cont::make_ArrayHandle(z_coords_ptr, z_npts, viskores::CopyFlag::Off);
            }
            else
            {
                z_coords_handle.Allocate(z_npts);
                viskores::Float64 *z = vtkh::GetVISKORESPointer(z_coords_handle);
                memcpy(z, z_coords_ptr, sizeof(float64) * z_npts);
            }
        }
        else
        {
            z_coords_handle.Allocate(1);
            z_coords_handle.WritePortal().Set(0, 0.0);
        }

        static_assert(std::is_same<viskores::FloatDefault, double>::value,
                    "Viskores needs to be configured with 'Viskores_USE_DOUBLE_PRECISION=ON'");
        viskores::cont::ArrayHandleCartesianProduct<
            viskores::cont::ArrayHandle<viskores::FloatDefault>,
            viskores::cont::ArrayHandle<viskores::FloatDefault>,
            viskores::cont::ArrayHandle<viskores::FloatDefault> > coords;

        coords = viskores::cont::make_ArrayHandleCartesianProduct(x_coords_handle,
                                                            y_coords_handle,
                                                            z_coords_handle);

        viskores::cont::CoordinateSystem coordinate_system(coords_name.c_str(), coords);
        result->AddCoordinateSystem(coordinate_system);

        viskores::Id3 topo_origin = detail::topo_origin(n_topo);

        if (ndims == 2)
        {
            viskores::cont::CellSetStructured<2> cell_set;
            cell_set.SetPointDimensions(viskores::make_Vec(x_npts,
                                                        y_npts));
            viskores::Id2 origin2(topo_origin[0], topo_origin[1]);
            cell_set.SetGlobalPointIndexStart(origin2);
            result->SetCellSet(cell_set);
        }
        else
        {
            viskores::cont::CellSetStructured<3> cell_set;
            cell_set.SetPointDimensions(viskores::make_Vec(x_npts,
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
    else
    {
        if (zero_copy &&
            (!n_coords["values/r"].dtype().is_float64() || 
            !n_coords["values/z"].dtype().is_float64()))
        {
            ASCENT_INFO("Zero-copy requested, but either r or z coordinate data is not float64." <<
                        "r type: " << n_coords["values/r"].dtype().name() << 
                        ", z type: " << n_coords["values/z"].dtype().name() << 
                        ". Turning zero-copy off.");
            zero_copy = false;
        }

        int r_npts = n_coords["values/r"].dtype().number_of_elements();
        int z_npts = n_coords["values/z"].dtype().number_of_elements();

        const float64 *r_coords_ptr;
        Node temp_r;
        if (n_coords["values/r"].dtype().is_float64())
        {
            r_coords_ptr = n_coords["values/r"].as_float64_ptr();
        }
        else
        {
            n_coords["values/r"].to_float64_array(temp_r);
            r_coords_ptr = temp_r.value();
        }
        
        const float64 *z_coords_ptr;
        Node temp_z;
        if (n_coords["values/z"].dtype().is_float64())
        {
            z_coords_ptr = n_coords["values/z"].as_float64_ptr();
        }
        else
        {
            n_coords["values/z"].to_float64_array(temp_z);
            z_coords_ptr = temp_z.value();
        }

        viskores::cont::ArrayHandle<viskores::Float64> r_coords_handle;
        viskores::cont::ArrayHandle<viskores::Float64> z_coords_handle;
        viskores::cont::ArrayHandle<viskores::Float64> theta_coords_handle;

        if(zero_copy)
        {
            r_coords_handle = viskores::cont::make_ArrayHandle(r_coords_ptr, r_npts, viskores::CopyFlag::Off);
            z_coords_handle = viskores::cont::make_ArrayHandle(z_coords_ptr, z_npts, viskores::CopyFlag::Off);
        }
        else
        {
            r_coords_handle.Allocate(r_npts);
            z_coords_handle.Allocate(z_npts);

            viskores::Float64 *r = vtkh::GetVISKORESPointer(r_coords_handle);
            memcpy(r, r_coords_ptr, sizeof(float64) * r_npts);
            viskores::Float64 *z = vtkh::GetVISKORESPointer(z_coords_handle);
            memcpy(z, z_coords_ptr, sizeof(float64) * z_npts);
        }

        theta_coords_handle.Allocate(1);
        theta_coords_handle.WritePortal().Set(0, 0.0);

        static_assert(std::is_same<viskores::FloatDefault, double>::value,
                    "Viskores needs to be configured with 'Viskores_USE_DOUBLE_PRECISION=ON'");

        viskores::cont::ArrayHandleCartesianProduct<
            viskores::cont::ArrayHandle<viskores::FloatDefault>,
            viskores::cont::ArrayHandle<viskores::FloatDefault>,
            viskores::cont::ArrayHandle<viskores::FloatDefault> > coords;

        coords = viskores::cont::make_ArrayHandleCartesianProduct(z_coords_handle,
                                                                    r_coords_handle,
                                                                    theta_coords_handle);

        viskores::cont::CoordinateSystem coordinate_system(coords_name.c_str(), coords);

        result->AddCoordinateSystem(coordinate_system);

        viskores::Id3 topo_origin = detail::topo_origin(n_topo);

        viskores::cont::CellSetStructured<2> cell_set;
        cell_set.SetPointDimensions(viskores::make_Vec(z_npts,
                                                    r_npts));
        viskores::Id2 origin2(topo_origin[0], topo_origin[1]);
        cell_set.SetGlobalPointIndexStart(origin2);
        result->SetCellSet(cell_set);

        nverts = r_npts * z_npts;
        neles = (r_npts - 1) * (z_npts - 1);

        return result;
    }
}

//-----------------------------------------------------------------------------

viskores::cont::DataSet *
VTKHDataAdapter::StructuredBlueprintToViskoresDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed rectilinear)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts,                    // output, number of verts
     bool zero_copy)                 // attempt to zero copy
{
    viskores::cont::DataSet *result = new viskores::cont::DataSet();

    string coords_type = n_coords["type"].as_string();
    viskores::cont::CoordinateSystem coords;
    int ndims = 0;
    const bool has_strided_topology =
      n_topo.has_path("elements/dims/offsets") &&
      n_topo.has_path("elements/dims/strides");

    const bool is_rz = n_coords["values"].has_child("r") && n_coords["values"].has_child("z");
    const bool is_cartesian = n_coords["values"].has_child("x") && n_coords["values"].has_child("y");

    if (is_rz && is_cartesian)
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z} but got parameters for both.")
    }
    
    if (!is_rz && !is_cartesian)
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z} but got neither.")
    }

    if (is_cartesian)
    {
        nverts = n_coords["values/x"].dtype().number_of_elements();
        if(n_coords["values/x"].dtype().is_float64())
        {
            // Use topology offsets and strides for structured explicit coordinates.
            if(coords_type == "explicit" && has_strided_topology)
            {
                coords = detail::GetStructuredExplicitCoordinateSystem<float64>(n_coords,
                                                                                n_topo,
                                                                                coords_name,
                                                                                ndims,
                                                                                nverts,
                                                                                zero_copy);
            }
            else
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
        }
        else if(n_coords["values/x"].dtype().is_float32())
        {
            // Use topology offsets and strides for structured explicit coordinates.
            if(coords_type == "explicit" && has_strided_topology)
            {
                coords = detail::GetStructuredExplicitCoordinateSystem<float32>(n_coords,
                                                                                n_topo,
                                                                                coords_name,
                                                                                ndims,
                                                                                nverts,
                                                                                zero_copy);
            }
            else
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
        }
        else
        {
            ASCENT_ERROR("Coordinate system must be floating point values");
        }
    }
    else if (n_coords["values"].has_child("r") && n_coords["values"].has_child("z"))
    {
        nverts = n_coords["values/r"].dtype().number_of_elements();
        if(n_coords["values/r"].dtype().is_float64())
        {
            index_t r_stride = n_coords["values/r"].dtype().stride();
            index_t r_element_stride = r_stride / sizeof(float64);
            index_t z_stride = n_coords["values/z"].dtype().stride();
            index_t z_element_stride = z_stride / sizeof(float64);
            
            coords = detail::GetRZCoordinateSystem<float64>(n_coords,
                                                                coords_name,
                                                                ndims,
                                                                r_element_stride,
                                                                z_element_stride,
                                                                zero_copy);
        }
        else if(n_coords["values/r"].dtype().is_float32())
        {
            index_t r_stride = n_coords["values/r"].dtype().stride();
            index_t r_element_stride = r_stride / sizeof(float32);
            index_t z_stride = n_coords["values/z"].dtype().stride();
            index_t z_element_stride = z_stride / sizeof(float32);

            coords = detail::GetRZCoordinateSystem<float32>(n_coords,
                                                                coords_name,
                                                                ndims,
                                                                r_element_stride,
                                                                z_element_stride,
                                                                zero_copy);
        }
        else
        {
            ASCENT_ERROR("Coordinate system must be floating point values");
        }
    }
    else
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z}");
    }

    result->AddCoordinateSystem(coords);

    int32 i_elems = n_topo["elements/dims/i"].to_int();
    int32 j_elems = n_topo["elements/dims/j"].to_int();

    viskores::Id3 topo_origin = detail::topo_origin(n_topo);

    if(coords_type == "explicit")
    {
      if(ndims == 2)
      {
        viskores::Id x_verts = i_elems + 1;
        viskores::Id y_verts = j_elems + 1;
        neles = i_elems * j_elems;
        nverts = (x_verts) * (y_verts);
        
        std::string ele_shape = "quad";
        viskores::UInt8 shape_id;
        viskores::IdComponent indices_per;
        detail::ViskoresCellShape(ele_shape, shape_id, indices_per);
        viskores::cont::ArrayHandle<viskores::Id> connectivity;
        connectivity.Allocate(neles * indices_per);
        auto conn_portal = connectivity.WritePortal();
        int offset = 0;
        // Build Connectivity 

        if (is_rz)
        {
            for (viskores::Id j = 0; j < j_elems; ++j) 
            {
                for (viskores::Id i = 0; i < i_elems; ++i) 
                {
                    viskores::Id v0 = j * x_verts + i;
                    viskores::Id v1 = v0 + 1;
                    viskores::Id v2 = v0 + x_verts;
                    viskores::Id v3 = v0 + x_verts + 1;
                    
                    conn_portal.Set(offset, v0);// bottom left
                    conn_portal.Set(offset+1, v1); //bottom right
                    conn_portal.Set(offset+2, v3); //top right
                    conn_portal.Set(offset+3, v2); //top left 
                    offset = offset + 4;
                }
            }
        }
        else
        {
            for (viskores::Id i = 0; i < i_elems; ++i) 
            {
                for (viskores::Id j = 0; j < j_elems; ++j) 
                {
                    viskores::Id v0 = j * x_verts + i;
                    viskores::Id v1 = v0 + 1;
                    viskores::Id v2 = v0 + x_verts;
                    viskores::Id v3 = v0 + x_verts + 1;
                    
                    conn_portal.Set(offset, v0);// bottom left
                    conn_portal.Set(offset+1, v1); //bottom right
                    conn_portal.Set(offset+2, v3); //top right
                    conn_portal.Set(offset+3, v2); //top left 
                    offset = offset + 4;
                }
            }
        }
        viskores::cont::CellSetSingleType<> cell_set;
        cell_set.Fill(nverts, shape_id, indices_per, connectivity);
        neles = cell_set.GetNumberOfCells();
        result->SetCellSet(cell_set);
      }
      else
      {
        int32 k_elems = n_topo["elements/dims/k"].to_int();

        viskores::Id x_verts = i_elems + 1;
        viskores::Id y_verts = j_elems + 1;
        viskores::Id z_verts = k_elems + 1;
        neles = i_elems * j_elems * k_elems;
        nverts = (x_verts) * (y_verts) * (z_verts);
        
        std::string ele_shape = "hex";
        viskores::UInt8 shape_id;
        viskores::IdComponent indices_per;
        detail::ViskoresCellShape(ele_shape, shape_id, indices_per);
        viskores::cont::ArrayHandle<viskores::Id> connectivity;
        connectivity.Allocate(neles * indices_per);
        auto conn_portal = connectivity.WritePortal();
        int offset = 0;
        // Build Connectivity (Polyhedral cells)
        for (viskores::Id i = 0; i < i_elems; ++i) 
        {
          for (viskores::Id j = 0; j < j_elems; ++j) 
          {
            for (viskores::Id k = 0; k < k_elems; ++k) 
            {
              viskores::Id v0 = k * y_verts * x_verts + j * x_verts + i;
              viskores::Id v1 = v0 + 1;
              viskores::Id v2 = v0 + x_verts;
              viskores::Id v3 = v0 + x_verts + 1;
              viskores::Id v4 = v0 + y_verts * x_verts;
              viskores::Id v5 = v4 + 1;
              viskores::Id v6 = v4 + x_verts;
              viskores::Id v7 = v4 + x_verts + 1;
              
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

        viskores::cont::CellSetSingleType<> cell_set;
        cell_set.Fill(nverts, shape_id, indices_per, connectivity);
        neles = cell_set.GetNumberOfCells();
        result->SetCellSet(cell_set);
      }
    }
    else
    {
      if (ndims == 2)
      {
        viskores::cont::CellSetStructured<2> cell_set;
        cell_set.SetPointDimensions(viskores::make_Vec(i_elems+1,
                                                   j_elems+1));
        viskores::Id2 origin2(topo_origin[0], topo_origin[1]);
        cell_set.SetGlobalPointIndexStart(origin2);
        result->SetCellSet(cell_set);
        neles = i_elems * j_elems;
        nverts = (i_elems + 1) * (j_elems + 1);
      }
      else
      {
        int32 k_elems = n_topo["elements/dims/k"].to_int();
        viskores::cont::CellSetStructured<3> cell_set;
        cell_set.SetPointDimensions(viskores::make_Vec(i_elems+1,
                                                   j_elems+1,
                                                   k_elems+1));
        cell_set.SetGlobalPointIndexStart(topo_origin);
        result->SetCellSet(cell_set);
        neles = i_elems * j_elems * k_elems;
        nverts = (i_elems + 1) * (j_elems + 1) * (k_elems + 1);
      }
    }    
    return result;
}

//-----------------------------------------------------------------------------

viskores::cont::DataSet *
VTKHDataAdapter::PointsImplicitBlueprintToViskoresDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed unstructured)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles  (will be the same as nverts)
     int &nverts,                    // output, number of verts (will be the same as neles)
     bool zero_copy)                 // attempt to zero copy
{
    viskores::cont::DataSet *result = new viskores::cont::DataSet();

    nverts = n_coords["values/x"].dtype().number_of_elements();

    int32 ndims;
    viskores::cont::CoordinateSystem coords;
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

    viskores::UInt8 shape_id = 1;
    viskores::IdComponent indices_per = 1;
    viskores::cont::CellSetSingleType<> cellset;
    // alloc conn to nverts, fill with 0 --> nverts-1)
    viskores::cont::ArrayHandle<viskores::Id> connectivity;
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
viskores::cont::DataSet *
VTKHDataAdapter::UnstructuredBlueprintToViskoresDataSet
    (const std::string &coords_name, // input string with coordset name
     const Node &n_coords,           // input mesh bp coordset (assumed unstructured)
     const std::string &topo_name,   // input string with topo name
     const Node &n_topo,             // input mesh bp topo
     int &neles,                     // output, number of eles
     int &nverts,                    // output, number of verts
     bool zero_copy)                 // attempt to zero copy
{
    viskores::cont::DataSet *result = new viskores::cont::DataSet();

    viskores::cont::CoordinateSystem coords;
    int32 ndims;

    if (n_coords["values"].has_child("x") && n_coords["values"].has_child("y"))
    {
        nverts = n_coords["values/x"].dtype().number_of_elements();
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
    }
    else if (n_coords["values"].has_child("r") && n_coords["values"].has_child("z"))
    {
        nverts = n_coords["values/r"].dtype().number_of_elements();
        if(n_coords["values/r"].dtype().is_float64())
        {
            index_t r_stride = n_coords["values/r"].dtype().stride();
            index_t r_element_stride = r_stride / sizeof(float64);
            index_t z_stride = n_coords["values/z"].dtype().stride();
            index_t z_element_stride = z_stride / sizeof(float64);

            if(r_stride % sizeof(float64) == 0)
            {
                coords = detail::GetRZCoordinateSystem<float64>(n_coords,
                                                                coords_name,
                                                                ndims,
                                                                r_element_stride,
                                                                z_element_stride,
                                                                zero_copy);
            }
        }
        else if(n_coords["values/r"].dtype().is_float32())
        {
            index_t r_stride = n_coords["values/r"].dtype().stride();
            index_t r_element_stride = r_stride / sizeof(float32);
            index_t z_stride = n_coords["values/z"].dtype().stride();
            index_t z_element_stride = z_stride / sizeof(float32);

            if(r_stride % sizeof(float64) == 0)
            {
                coords = detail::GetRZCoordinateSystem<float32>(n_coords,
                                                                coords_name,
                                                                ndims,
                                                                r_element_stride,
                                                                z_element_stride,
                                                                zero_copy);
            }
        }
        else
        {
            ASCENT_ERROR("Coordinate system must be floating point values");
        }
    }
    else
    {
        ASCENT_ERROR("Unsupported coordset: expected cartesian {x,y,(z)} or cylindrical {r,z}");
    }

    result->AddCoordinateSystem(coords);

    // shapes, number of indices, and connectivity.
    // Will have to do something different if this is a "zoo"

    const Node &n_topo_eles = n_topo["elements"];
    std::string ele_shape = n_topo_eles["shape"].as_string();

    if(ele_shape == "mixed")
    {
        // blueprint allows mapping of shape names
        // to arbitrary ids, check if shape ids match the Viskores ids
        index_t num_of_shapes = n_topo_eles["shape_map"].number_of_children();

        if(!CheckShapeMapVsViskoresShapeIds(n_topo_eles["shape_map"]))
        {
            Node ref_map;
            ViskoresBlueprintShapeMap(ref_map);
            // TODO -- (strategy to remap ids)?
            ASCENT_ERROR("Shape Map Entries do not match required Viskores Shape Ids."
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

        viskores::cont::ArrayHandle<viskores::Id> viskores_conn;
        detail::BlueprintIndexArrayToViskoresIdArray(n_topo_eles["connectivity"],
                                                 zero_copy,
                                                 viskores_conn);

        // shapes
        viskores::cont::ArrayHandle<viskores::UInt8> viskores_shapes;
        detail::BlueprintIndexArrayToViskoresIdArray(n_topo_eles["shapes"],
                                                 zero_copy,
                                                 viskores_shapes);

        // offsets
        viskores::cont::ArrayHandle<viskores::Id> viskores_offsets;
        detail::BlueprintIndexArrayToViskoresIdArray(n_topo_eles["offsets"],
                                                 zero_copy,
                                                 viskores_offsets);

        // viskores offsets needs an extra entry
        // the last entry needs to be the size of the conn array
        viskores::cont::ArrayHandle<viskores::Id> viskores_offsets_full;
        viskores_offsets_full.Allocate(neles + 1);
        viskores::cont::ArrayHandle<viskores::Id>::WritePortalType viskores_offsets_full_wp = viskores_offsets_full.WritePortal();
        viskores::cont::ArrayHandle<viskores::Id>::ReadPortalType viskores_offsets_rp = viskores_offsets.ReadPortal();

        for(int i=0;i<neles;i++)
        {
          viskores_offsets_full_wp.Set(i,viskores_offsets_rp.Get(i));
        }
        // set last
        viskores_offsets_full_wp.Set(neles,num_ids);

        viskores::cont::CellSetExplicit<> cell_set;
        cell_set.Fill(nverts, viskores_shapes, viskores_conn, viskores_offsets_full);
        result->SetCellSet(cell_set);
        // for debugging help
        //result->PrintSummary(std::cout);
    }
    else
    {
        viskores::cont::ArrayHandle<viskores::Id> viskores_conn;
        detail::BlueprintIndexArrayToViskoresIdArray(n_topo_eles["connectivity"],zero_copy,viskores_conn);
        viskores::UInt8 shape_id;
        viskores::IdComponent indices_per;
        detail::ViskoresCellShape(ele_shape, shape_id, indices_per);
        viskores::cont::CellSetSingleType<> cell_set;
        cell_set.Fill(nverts, shape_id, indices_per, viskores_conn);
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
                          const Node &n_topo,
                          int neles,
                          int nverts,
                          viskores::cont::DataSet *dset,
                          bool zero_copy)                 // attempt to zero copy
{
    // TODO: how do we deal with vector valued fields?, these will be mcarrays

    string assoc_str = n_field["association"].as_string();

    viskores::cont::Field::Association viskores_assoc = viskores::cont::Field::Association::Any;
    if(assoc_str == "vertex")
    {
      viskores_assoc = viskores::cont::Field::Association::Points;
    }
    else if(assoc_str == "element")
    {
      viskores_assoc = viskores::cont::Field::Association::Cells;
    }
    else
    {
      ASCENT_INFO("Viskores conversion does not support field assoc "<<assoc_str<<". Skipping");
      return;
    }
    if(n_field["values"].number_of_children() > 1)
    {
      ASCENT_ERROR("Add field can only use zero or one component");
    }

    bool is_values = n_field["values"].number_of_children() == 0;
    const Node &n_vals = is_values ? n_field["values"] : n_field["values"].child(0);
    int num_vals = n_vals.dtype().number_of_elements();
    // Strided fields can have padded storage larger than the logical mesh.
    const bool has_strided_layout = detail::HasStridedLayout(n_field);

    if(!has_strided_layout && assoc_str == "vertex" && nverts != num_vals)
    {
      ASCENT_INFO("Field '"<<field_name<<"' (topology: '" << topo_name <<
                  "') number of values "<<num_vals<<
                  " does not match the number of points "<<nverts<<". Skipping");
      return;
    }

    if(!has_strided_layout && assoc_str == "element" && neles != num_vals)
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
        std::vector<index_t> logical_dims;
        std::vector<index_t> offsets;
        std::vector<index_t> strides;
        // Read field offsets and strides for logical value mapping.
        if(has_strided_layout)
        {
            logical_dims = detail::LogicalDims(n_topo, assoc_str);
            offsets = detail::IndexVector(n_field["offsets"]);
            strides = detail::IndexVector(n_field["strides"]);
        }

        // viskores can stride as long as the strides are a multiple of the native stride

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
                // Build a logical field view when Blueprint stride metadata is present.
                if(has_strided_layout)
                {
                    dset->AddField(detail::GetStridedField<float32>(n_vals,
                                                                    field_name,
                                                                    assoc_str,
                                                                    logical_dims,
                                                                    offsets,
                                                                    strides,
                                                                    element_stride,
                                                                    zero_copy));
                }
                else
                {
                    dset->AddField(detail::GetField<float32>(n_vals,
                                                             field_name,
                                                             assoc_str,
                                                             topo_name,
                                                             element_stride,
                                                             zero_copy));
                }
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
                // Build a logical field view when Blueprint stride metadata is present.
                if(has_strided_layout)
                {
                    dset->AddField(detail::GetStridedField<float64>(n_vals,
                                                                    field_name,
                                                                    assoc_str,
                                                                    logical_dims,
                                                                    offsets,
                                                                    strides,
                                                                    element_stride,
                                                                    zero_copy));
                }
                else
                {
                    dset->AddField(detail::GetField<float64>(n_vals,
                                                             field_name,
                                                             assoc_str,
                                                             topo_name,
                                                             element_stride,
                                                             zero_copy));
                }
                supported_type = true;
            }
        }
        else if(n_vals.dtype().is_uint8())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::uint8 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::uint8> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_uint16())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::uint16 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::uint16> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_uint32())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::uint32 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::uint32> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_uint64())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::uint64 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::uint64> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int8())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::int8 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::int8> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int16())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::int16 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::int16> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int32())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::int32 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::int32> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        else if(n_vals.dtype().is_int64())
        {

          viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
          viskores_arr.Allocate(num_vals);

          const conduit::int64 *input = n_vals.value();
          viskores::cont::ArrayHandle<conduit::int64> input_arr = viskores::cont::make_ArrayHandle(input, num_vals, viskores::CopyFlag::Off);

          viskores::cont::Invoker invoker;
          vtkh::ViskoresTypeCast worklet;

          invoker(worklet,input_arr,viskores_arr);

          // add field to dataset
          if(assoc_str == "vertex")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Points,
                                               viskores_arr));
              supported_type = true;
          }
          else if( assoc_str == "element")
          {
              dset->AddField(viskores::cont::Field(field_name.c_str(),
                                               viskores::cont::Field::Association::Cells,
                                               viskores_arr));
              supported_type = true;
          }
        }
        // ***********************************************************************
        // ***********************************************************************
        // ***********************************************************************
        // NOTE: TODO OUR VISKORES is not compiled with int32 and int64 support ...
        // ***********************************************************************
        // These cases fail and provide this error message:
        //   Execution failed with viskores: Could not find appropriate cast for array in CastAndCall.
        //   Array: valueType=x storageType=N4viskores4cont15StorageTagBasicE 27 values occupying 216 bytes [0 1 2 ... 24 25 26]
        //   TypeList: N4viskores4ListIJfdEEE
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

        // viskores cant support zero copy for this layout or was not compiled to expose this datatype
        // use float64 by default
        if(!supported_type)
        {

            //std::cout << "WE ARE IN UNSUPPORTED DATA TYPE: "
            //          << n_vals.dtype().name() << std::endl;
            // convert to float64, we use this as a compromise to cover the widest range
            viskores::cont::ArrayHandle<viskores::Float64> viskores_arr;
            viskores_arr.Allocate(num_vals);

            // TODO -- FUTURE: Do this conversion w/ device if on device
            void *ptr = (void*) vtkh::GetVISKORESPointer(viskores_arr);
            Node n_tmp;
            n_tmp.set_external(DataType::float64(num_vals),ptr);
            n_vals.to_float64_array(n_tmp);

            // add field to dataset
            if(assoc_str == "vertex")
            {
                dset->AddField(viskores::cont::Field(field_name.c_str(),
                                                 viskores::cont::Field::Association::Points,
                                                 viskores_arr));
            }
            else if( assoc_str == "element")
            {
                dset->AddField(viskores::cont::Field(field_name.c_str(),
                                                 viskores::cont::Field::Association::Cells,
                                                 viskores_arr));
            }
        // else
        // {
        //     std::cout << "SUPPORTED DATA TYPE: "
        //               << n_vals.dtype().name() << std::endl;
        // }
      }
    }
    catch (viskores::cont::Error error)
    {
        ASCENT_ERROR("Viskores exception:" << error.GetMessage());
    }

}

void
VTKHDataAdapter::AddVectorField(const std::string &field_name,
                                const Node &n_field,
                                const std::string &topo_name,
                                int neles,
                                int nverts,
                                viskores::cont::DataSet *dset,
                                const int dims,
                                bool zero_copy)                 // attempt to zero copy
{
    string assoc_str = n_field["association"].as_string();

    viskores::cont::Field::Association viskores_assoc = viskores::cont::Field::Association::Any;
    if(assoc_str == "vertex")
    {
      viskores_assoc = viskores::cont::Field::Association::Points;
    }
    else if(assoc_str == "element")
    {
      viskores_assoc = viskores::cont::Field::Association::Cells;
    }
    else
    {
      ASCENT_INFO("Viskores conversion does not support field assoc "<<assoc_str<<". Skipping");
      return;
    }


    const Node &n_vals = n_field["values"];
    const int num_vals = (assoc_str == "vertex") ? nverts : neles;
    int num_components = n_field["values"].number_of_children();

    if(n_vals.child(0).dtype().number_of_elements() < num_vals)
    {
      ASCENT_INFO("Field '"<<field_name<<"' (topology: '" << topo_name <<
                  "') number of values "<<n_vals.child(0).dtype().number_of_elements()<<
                  " does not match the number of "
                  <<(assoc_str == "vertex" ? "points " : "elements ")
                  <<num_vals<<". Skipping");
      return;
    }

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

                using Vec3f32 = viskores::Vec<viskores::Float32,3>;
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

                using Vec3f64 = viskores::Vec<viskores::Float64,3>;
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

                using Vec2f32 = viskores::Vec<viskores::Float32,2>;
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

                using Vec2f64 = viskores::Vec<viskores::Float64,2>;
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
          // While viskores supports ArrayHandleCompositeVectors for
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
    catch (viskores::cont::Error error)
    {
        ASCENT_ERROR("Viskores exception:" << error.GetMessage());
    }

}

template <typename Id_T, typename Float_T>
void AddMatSetFieldsCommon(const conduit::Node &matset,
                           const std::string &length_name,
                           const std::string &offsets_name,
                           const std::string &ids_name,
                           const std::string &vfs_name,
                           const std::string &topo_name,
                           int neles,
                           viskores::cont::DataSet *dset)
{
    viskores::cont::Field length, offsets, ids, vfs;

    detail::GetMatSetFields<Id_T, Float_T>(
        matset,
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

const conduit::Node &
GetSparseByMaterialVfsSample(const conduit::Node &matset,
                             const std::string   &matset_name)
{
    const conduit::Node &elem_ids = matset["element_ids"];
    const conduit::Node &vf_group = matset["volume_fractions"];

    const int num_ids       = elem_ids.number_of_children();
    const int num_materials = vf_group.number_of_children();

    if (num_ids == 0)
    {
        ASCENT_ERROR("No element ids were defined for matset: " << matset_name);
    }

    if (num_materials == 0)
    {
        ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);
    }

    if (num_materials != num_ids)
    {
        ASCENT_ERROR("Number of materials (" << num_materials
                     << ") does not match number of element IDs ("
                     << num_ids << ") defined for matset: " << matset_name);
    }

    const conduit::Node &first_child = vf_group.child(0);
    const int            child_count = first_child.number_of_children();

    return (child_count != 0)
           ? *first_child.child_ptr(0)
           : *vf_group.child_ptr(0);
}

void
VTKHDataAdapter::AddMatSets(const std::string &matset_name,
                            const conduit::Node &n_matset,
                            const std::string &topo_name,
                            int neles,
                            viskores::cont::DataSet *dset,
                            bool zero_copy)
{
    // Common precondition: all matsets must have volume fractions.
    if (!n_matset.has_child("volume_fractions"))
    {
        ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);
    }

    const bool use64BitIds = (sizeof(viskores::Id) == 8);

    const std::string assoc_str = "element";
    const std::string length_name = "sizes";
    const std::string offsets_name = "offsets";
    const std::string ids_name = "material_ids";
    const std::string vfs_name = "volume_fractions";

    // Helper: add an integer Node as a viskores::Id field, converting width if needed.
    auto add_index_field_as_Id = [&](const conduit::Node &src, const std::string &name, const std::string &assoc)
    {
        const index_t n = static_cast<index_t>(src.dtype().number_of_elements());
        const bool type_ok = ( use64BitIds && src.dtype().is_int64() ) || (!use64BitIds && src.dtype().is_int32());

        if (type_ok)
        {
            dset->AddField(
                detail::GetField<viskores::Id>(src,
                                               name,
                                               assoc,
                                               topo_name,
                                               index_t(1),
                                               zero_copy));
            return;
        }

        conduit::Node tmp;

        if (use64BitIds && src.dtype().is_int32())
        {
            // 32 -> 64
            tmp.set(conduit::DataType::int64(n));

            const conduit::int32 *p32 = src.as_int32_ptr();
            conduit::int64 *p64 = tmp.as_int64_ptr();

            for (index_t i = 0; i < n; ++i)
            {
                p64[i] = static_cast<conduit::int64>(p32[i]);
            }
        }
        else if (!use64BitIds && src.dtype().is_int64())
        {
            // 64 -> 32
            tmp.set(conduit::DataType::int32(n));

            const conduit::int64 *p64 = src.as_int64_ptr();
            conduit::int32 *p32 = tmp.as_int32_ptr();

            for (index_t i = 0; i < n; ++i)
            {
                p32[i] = static_cast<conduit::int32>(p64[i]);
            }
        }
        else
        {
            ASCENT_ERROR("Unsupported integer type for index field '" << name << "'");
        }

        dset->AddField(detail::GetField<viskores::Id>(tmp,
                                                      name,
                                                      assoc,
                                                      topo_name,
                                                      index_t(1),
                                                      false));
    };

    // ------------------------------------------------------------------------
    // 64-bit ID path
    // ------------------------------------------------------------------------
    if (use64BitIds)
    {
        // --------------------------------------------------------------------
        // Case 1: "sparse_by_element" / material_map
        // --------------------------------------------------------------------
        if (n_matset.has_child("material_map"))
        {
            try
            {
                // sizes and offsets as Id-type fields
                const conduit::Node &n_sizes = n_matset["sizes"];
                const conduit::Node &n_offsets = n_matset["offsets"];

                add_index_field_as_Id(n_sizes, length_name, assoc_str);
                add_index_field_as_Id(n_offsets, offsets_name, assoc_str);

                // Material IDs: allow int32 or int64 in input, ensure > 0, then adapt to Id type
                const conduit::Node &n_material_ids = n_matset["material_ids"];
                const auto &id_dtype = n_material_ids.dtype();
                const index_t num_vals = static_cast<index_t>(id_dtype.number_of_elements());

                conduit::Node tmp_ids;
                const conduit::Node *ids_src = &n_material_ids;

                if (id_dtype.is_int32())
                {
                    const conduit::int32 *ids = n_material_ids.as_int32_ptr();
                    const bool has_non_positive = std::any_of(ids, ids + num_vals, [](conduit::int32 v) { return v <= 0; });

                    if (has_non_positive)
                    {
                        tmp_ids.set(n_material_ids);
                        conduit::int32 *mutable_ids = tmp_ids.as_int32_ptr();
                        for (index_t i = 0; i < num_vals; ++i)
                        {
                            mutable_ids[i] += 1;
                        }
                        ids_src = &tmp_ids;
                    }
                }
                else if (id_dtype.is_int64())
                {
                    const conduit::int64 *ids = n_material_ids.as_int64_ptr();
                    const bool has_non_positive = std::any_of(ids, ids + num_vals, [](conduit::int64 v) { return v <= 0; });

                    if (has_non_positive)
                    {
                        tmp_ids.set(n_material_ids);
                        conduit::int64 *mutable_ids = tmp_ids.as_int64_ptr();
                        for (index_t i = 0; i < num_vals; ++i)
                        {
                            mutable_ids[i] += 1;
                        }
                        ids_src = &tmp_ids;
                    }
                }
                else
                {
                    ASCENT_ERROR("Unsupported integer type for material IDs in matset: "
                                 << matset_name);
                }

                // Now adapt material_ids (possibly shifted) to viskores::Id
                add_index_field_as_Id(*ids_src, ids_name, "whole");

                // Volume fractions: must be float32 or float64
                const conduit::Node &n_vfs = n_matset["volume_fractions"];

                if (n_vfs.dtype().is_float32())
                {
                    dset->AddField(detail::GetField<float32>(n_vfs,
                                                            vfs_name,
                                                            "whole",
                                                            topo_name,
                                                            index_t(1),
                                                            zero_copy));
                }
                else if (n_vfs.dtype().is_float64())
                {
                    dset->AddField(detail::GetField<float64>(n_vfs,
                                                            vfs_name,
                                                            "whole",
                                                            topo_name,
                                                            index_t(1),
                                                            zero_copy));
                }
                else
                {
                    ASCENT_ERROR("Unsupported floating-point type for volume_fractions in matset: "
                                 << matset_name);
                }
            }
            catch (const viskores::cont::Error &error)
            {
                ASCENT_ERROR("Viskores exception: " << error.GetMessage());
            }
        }
        
        // --------------------------------------------------------------------
        // Case 2: "sparse_by_material" (element_ids)
        // --------------------------------------------------------------------
        else if (n_matset.has_child("element_ids"))
        {
            const conduit::Node &sample_vfs = GetSparseByMaterialVfsSample(n_matset, matset_name);

            try
            {
                // Prepare matset with element_ids widened to int64 if needed
                const conduit::Node *matset_for_fields = &n_matset;
                conduit::Node matset_converted;

                const conduit::Node &elem_ids_src = n_matset["element_ids"];

                bool need_conversion = false;
                const int num_elem_children = elem_ids_src.number_of_children();
                for (int i = 0; i < num_elem_children; ++i)
                {
                    const conduit::Node &child = elem_ids_src.child(i);
                    if (child.dtype().is_int32())
                    {
                        need_conversion = true;
                        break;
                    }
                }

                if (need_conversion)
                {
                    matset_converted.set(n_matset);
                    conduit::Node &elem_ids_dst = matset_converted["element_ids"];

                    for (int i = 0; i < elem_ids_dst.number_of_children(); ++i)
                    {
                        conduit::Node &child = elem_ids_dst.child(i);
                        if (child.dtype().is_int32())
                        {
                            const index_t n = static_cast<index_t>(child.dtype().number_of_elements());

                            conduit::Node tmp64;
                            tmp64.set(conduit::DataType::int64(n));

                            const conduit::int32 *src_ptr = child.as_int32_ptr();
                            conduit::int64 *dst_ptr = tmp64.as_int64_ptr();

                            for (index_t j = 0; j < n; ++j)
                            {
                                dst_ptr[j] = static_cast<conduit::int64>(src_ptr[j]);
                            }

                            // Replace the child array with the 64-bit version
                            child.set(tmp64);
                        }
                    }

                    matset_for_fields = &matset_converted;
                }

                if (sample_vfs.dtype().is_float32())
                {
                    AddMatSetFieldsCommon<viskores::Id, float32>(
                        *matset_for_fields,
                        length_name,
                        offsets_name,
                        ids_name,
                        vfs_name,
                        topo_name,
                        neles,
                        dset);
                }
                else if (sample_vfs.dtype().is_float64())
                {
                    AddMatSetFieldsCommon<viskores::Id, float64>(
                        *matset_for_fields,
                        length_name,
                        offsets_name,
                        ids_name,
                        vfs_name,
                        topo_name,
                        neles,
                        dset);
                }
                else
                {
                    ASCENT_ERROR("Unsupported floating-point type for sparse_by_material "
                                 "volume_fractions in matset: " << matset_name);
                }
            }
            catch (const viskores::cont::Error &error)
            {
                ASCENT_ERROR("Viskores exception: " << error.GetMessage());
            }
        }

        // --------------------------------------------------------------------
        // Case 3: "full" matset
        // --------------------------------------------------------------------
        else
        {
            const conduit::Node &vf_group = n_matset["volume_fractions"];
            const int num_materials = vf_group.number_of_children();

            if (num_materials == 0)
            {
                ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);
            }

            const conduit::Node &first_material = vf_group.child(0);
            const std::string material_name = first_material.name();
            const index_t num_vals = static_cast<index_t>(first_material.dtype().number_of_elements());

            if (num_vals != static_cast<index_t>(neles))
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
                if (first_material.dtype().is_float32())
                {
                    AddMatSetFieldsCommon<viskores::Id, float32>(
                        n_matset,
                        length_name,
                        offsets_name,
                        ids_name,
                        vfs_name,
                        topo_name,
                        neles,
                        dset);
                }
                else if (first_material.dtype().is_float64())
                {
                    AddMatSetFieldsCommon<viskores::Id, float64>(
                        n_matset,
                        length_name,
                        offsets_name,
                        ids_name,
                        vfs_name,
                        topo_name,
                        neles,
                        dset);
                }
                else
                {
                    ASCENT_ERROR("Unsupported floating-point type for full matset "
                                 "volume_fractions in matset: " << matset_name);
                }
            }
            catch (const viskores::cont::Error &error)
            {
                ASCENT_ERROR("Viskores exception: " << error.GetMessage());
            }
        }

        return;
    }

    // ------------------------------------------------------------------------
    // 32-bit ID path
    // ------------------------------------------------------------------------

    //TODO: zero_copy = true segfaulting in viskores mir filter
    //zero_copy = false;

    // --------------------------------------------------------------------
    // Case 1: "sparse_by_element" (material_map)
    // --------------------------------------------------------------------
    if (n_matset.has_child("material_map"))
    {
        try
        {
            // Add materials directly
            const conduit::Node &n_length = n_matset["sizes"];
            const conduit::Node &n_offsets = n_matset["offsets"];

            add_index_field_as_Id(n_length, length_name, assoc_str);
            add_index_field_as_Id(n_offsets, offsets_name, assoc_str);

            const conduit::Node &n_material_ids = n_matset["material_ids"];
            const int num_vals = n_material_ids.dtype().number_of_elements();

            if (n_material_ids.dtype().is_int32())
            {
                const conduit::int32 *ids = n_material_ids.value();
                const bool has_non_positive = std::any_of(ids, ids + num_vals, [](conduit::int32 v) { return v <= 0; });

                if (has_non_positive) // need to make a copy and increment all material ids
                {
                    conduit::Node n_mat_ids = n_matset["material_ids"];
                    conduit::int32 *tmp_vec_ids = n_mat_ids.value();

                    for (index_t i = 0; i < num_vals; ++i)
                    {
                        tmp_vec_ids[i] += 1;
                    }

                    viskores::cont::Field field_copy = detail::GetField<int32>(n_mat_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               false);
                    dset->AddField(field_copy);
                }
                else // can zero copy the material ids
                {
                    viskores::cont::Field field_copy = detail::GetField<int32>(n_material_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               zero_copy);

                    dset->AddField(field_copy);
                }
            }
            else if (n_material_ids.dtype().is_int64())
            {
                const conduit::int64 *ids = n_material_ids.value();
                const bool has_non_positive = std::any_of(ids, ids + num_vals, [](conduit::int64 v) { return v <= 0; });

                if (has_non_positive) // need to make a copy and increment all material ids
                {
                    conduit::Node n_mat_ids = n_matset["material_ids"];
                    conduit::int64 *tmp_vec_ids = n_mat_ids.value();

                    for (index_t i = 0; i < num_vals; ++i)
                    {
                        tmp_vec_ids[i] += 1;
                    }

                    viskores::cont::Field field_copy = detail::GetField<int64>(n_mat_ids,
                                                                               ids_name,
                                                                               "whole",
                                                                               topo_name,
                                                                               index_t(1),
                                                                               false);
                    dset->AddField(field_copy);
                }
                else // can zero copy the material ids
                {
                    dset->AddField(detail::GetField<int64>(n_material_ids,
                                                           ids_name,
                                                           "whole",
                                                           topo_name,
                                                           index_t(1),
                                                           zero_copy));
                }
            }
            else
            {
                ASCENT_ERROR("Unsupported integer type for material IDs");
            }

            if (n_matset["volume_fractions"].dtype().is_float32())
            {
                const conduit::Node &n_volume_fractions = n_matset["volume_fractions"];
                dset->AddField(detail::GetField<float32>(n_volume_fractions,
                                                         vfs_name,
                                                         "whole",
                                                         topo_name,
                                                         index_t(1),
                                                         zero_copy));
            }
            else if (n_matset["volume_fractions"].dtype().is_float64())
            {
                const conduit::Node &n_volume_fractions = n_matset["volume_fractions"];
                dset->AddField(detail::GetField<float64>(n_volume_fractions,
                                                         vfs_name,
                                                         "whole",
                                                         topo_name,
                                                         index_t(1),
                                                         zero_copy));
            }
        }
        catch (const viskores::cont::Error &error)
        {
            ASCENT_ERROR("Viskores exception:" << error.GetMessage());
        }
    }

    // --------------------------------------------------------------------
    // Case 2: "sparse_by_material" (element_ids)
    // --------------------------------------------------------------------
    else if (n_matset.has_child("element_ids"))
    {
        const conduit::Node &sample_vfs = GetSparseByMaterialVfsSample(n_matset, matset_name);

        try
        {
            if (sample_vfs.dtype().is_float32())
            {
                AddMatSetFieldsCommon<int, float32>(
                    n_matset,
                    length_name,
                    offsets_name,
                    ids_name,
                    vfs_name,
                    topo_name,
                    neles,
                    dset);
            }
            else if (sample_vfs.dtype().is_float64())
            {
                AddMatSetFieldsCommon<int, float64>(
                    n_matset,
                    length_name,
                    offsets_name,
                    ids_name,
                    vfs_name,
                    topo_name,
                    neles,
                    dset);
            }
        }
        catch (const viskores::cont::Error &error)
        {
            ASCENT_ERROR("Viskores exception:" << error.GetMessage());
        }
    }

    // --------------------------------------------------------------------
    // Case 3: "full" matset
    // --------------------------------------------------------------------
    else
    {
        int num_materials = n_matset["volume_fractions"].number_of_children();
        if (num_materials == 0)
        {
            ASCENT_ERROR("No volume fractions were defined for matset: " << matset_name);
        }

        const conduit::Node &n_material = n_matset["volume_fractions"].child(0);
        std::string material_name = n_material.name();
        int num_vals = n_material.dtype().number_of_elements();

        if (num_vals != neles)
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
            if (n_material.dtype().is_float32())
            {
                AddMatSetFieldsCommon<int, float32>(
                    n_matset,
                    length_name,
                    offsets_name,
                    ids_name,
                    vfs_name,
                    topo_name,
                    neles,
                    dset);
            }
            else if (n_material.dtype().is_float64())
            {
                AddMatSetFieldsCommon<int, float64>(
                    n_matset,
                    length_name,
                    offsets_name,
                    ids_name,
                    vfs_name,
                    topo_name,
                    neles,
                    dset);
            }
        }
        catch (const viskores::cont::Error &error)
        {
            ASCENT_ERROR("Viskores exception:" << error.GetMessage());
        }
    }
}

std::string
GetBlueprintCellName(viskores::UInt8 shape_id)
{
  std::string name;
  if(shape_id == viskores::CELL_SHAPE_TRIANGLE)
  {
    name = "tri";
  }
  else if(shape_id == viskores::CELL_SHAPE_VERTEX)
  {
    name = "point";
  }
  else if(shape_id == viskores::CELL_SHAPE_LINE)
  {
    name = "line";
  }
  else if(shape_id == viskores::CELL_SHAPE_POLYGON)
  {
    ASCENT_ERROR("Polygon is not supported in blueprint");
  }
  else if(shape_id == viskores::CELL_SHAPE_QUAD)
  {
    name = "quad";
  }
  else if(shape_id == viskores::CELL_SHAPE_TETRA)
  {
    name = "tet";
  }
  else if(shape_id == viskores::CELL_SHAPE_HEXAHEDRON)
  {
    name = "hex";
  }
  else if(shape_id == viskores::CELL_SHAPE_PYRAMID)
  {
    name = "pyramid";
  }
  else if(shape_id == viskores::CELL_SHAPE_WEDGE)
  {
    name = "wedge";
  }
  return name;
}


inline index_t
viskores_shape_size(viskores::Id shape_id)
{
    switch(shape_id)
    {
        // point
        case viskores::CELL_SHAPE_VERTEX:  return 1; break;
        // line
        case viskores::CELL_SHAPE_LINE:  return 2; break;
        // tri
        case viskores::CELL_SHAPE_TRIANGLE:  return 3; break;
        // quad
        case viskores::CELL_SHAPE_QUAD:  return 4; break;
        // tet
        case viskores::CELL_SHAPE_TETRA: return 4; break;
        // hex
        case viskores::CELL_SHAPE_HEXAHEDRON: return 8; break;
        // pyramid
        case viskores::CELL_SHAPE_PYRAMID: return 5; break;
        // wedge
        case viskores::CELL_SHAPE_WEDGE: return 6; break;
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
        sizes_arr[i] = viskores_shape_size(shapes_arr[i]);
    }
    
}

bool
VTKHDataAdapter::ViskoresTopologyToBlueprint(conduit::Node &output,
                                         const viskores::cont::DataSet &data_set,
                                         const std::string &topo_name,
                                         bool zero_copy)
{

  int topo_dims;
  bool is_structured = vtkh::VISKORESDataSetInfo::IsStructured(data_set, topo_dims);
  bool is_uniform = vtkh::VISKORESDataSetInfo::IsUniform(data_set);
  bool is_rectilinear = vtkh::VISKORESDataSetInfo::IsRectilinear(data_set);
  viskores::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
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
    auto points = coords.GetData().AsArrayHandle<viskores::cont::ArrayHandleUniformPointCoordinates>();
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
    typedef viskores::cont::ArrayHandleCartesianProduct<viskores::cont::ArrayHandle<viskores::FloatDefault>,
                                                    viskores::cont::ArrayHandle<viskores::FloatDefault>,
                                                    viskores::cont::ArrayHandle<viskores::FloatDefault>> Cartesian;

    const auto points = coords.GetData().AsArrayHandle<Cartesian>();
    auto portal = points.ReadPortal();
    auto x_portal = portal.GetFirstPortal();
    auto y_portal = portal.GetSecondPortal();
    auto z_portal = portal.GetThirdPortal();

    // work around for conduit not accepting const pointers
    viskores::FloatDefault *x_ptr = const_cast<viskores::FloatDefault*>(x_portal.GetArray());
    viskores::FloatDefault *y_ptr = const_cast<viskores::FloatDefault*>(y_portal.GetArray());
    viskores::FloatDefault *z_ptr = const_cast<viskores::FloatDefault*>(z_portal.GetArray());

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
    using Coords32 = viskores::cont::ArrayHandleSOA<viskores::Vec<viskores::Float32, 3>>;
    using Coords64 = viskores::cont::ArrayHandleSOA<viskores::Vec<viskores::Float64, 3>>;

    using CoordsVec32 = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32,3>>;
    using CoordsVec64 = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,3>>;

    viskores::cont::UnknownArrayHandle coordsHandle(coords.GetData());

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
          set_external(vtkh::GetVISKORESPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set_external(vtkh::GetVISKORESPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set_external(vtkh::GetVISKORESPointer(z_handle), point_dims[2]);
      }
      else
      {
        output["coordsets/"+coords_name+"/values/x"].
          set(vtkh::GetVISKORESPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set(vtkh::GetVISKORESPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set(vtkh::GetVISKORESPointer(z_handle), point_dims[2]);

      }

    }
    else if(coordsHandle.IsType<CoordsVec32>())
    {
      CoordsVec32 points;
      coordsHandle.AsArrayHandle(points);

      const int num_vals = points.GetNumberOfValues();
      viskores::Float32 *points_ptr = (viskores::Float32*)vtkh::GetVISKORESPointer(points);
      const int byte_size = sizeof(viskores::Float32);

      if(zero_copy)
      {
        output["coordsets/"+coords_name+"/values/x"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*0,  // byte offset
                                                                  byte_size*3); // stride
        output["coordsets/"+coords_name+"/values/y"].set_external(points_ptr,
                                                                  num_vals,
                                                                  byte_size*1,  // byte offset
                                                                  sizeof(viskores::Float32)*3); // stride
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
                                                         sizeof(viskores::Float32)*3); // stride
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
          set_external(vtkh::GetVISKORESPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set_external(vtkh::GetVISKORESPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set_external(vtkh::GetVISKORESPointer(z_handle), point_dims[2]);
      }
      else
      {
        output["coordsets/"+coords_name+"/values/x"].
          set(vtkh::GetVISKORESPointer(x_handle), point_dims[0]);
        output["coordsets/"+coords_name+"/values/y"].
          set(vtkh::GetVISKORESPointer(y_handle), point_dims[1]);
        output["coordsets/"+coords_name+"/values/z"].
          set(vtkh::GetVISKORESPointer(z_handle), point_dims[2]);

      }
    }
    else if(coordsHandle.IsType<CoordsVec64>())
    {
      CoordsVec64 points;
      coordsHandle.AsArrayHandle(points);

      const int num_vals = points.GetNumberOfValues();
      viskores::Float64 *points_ptr = (viskores::Float64*)vtkh::GetVISKORESPointer(points);
      const int byte_size = sizeof(viskores::Float64);

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
      // Ok viskores has handed us something we don't know about, and its really
      // hard to ask viskores to tell us what it is. Before we give up, we will
      // attempt to copy the data to a known type and copy that copy.
      // We can't avoid the double copy since conduit can't take ownership
      // and we can't seem to write to a zero copied array

      viskores::cont::ArrayHandle<viskores::Vec<double,3>> coords_copy;
      viskores::cont::ArrayCopy(coordsHandle, coords_copy);
      const int num_vals = coords_copy.GetNumberOfValues();
      viskores::Float64 *points_ptr = (viskores::Float64*)vtkh::GetVISKORESPointer(coords_copy);
      const int byte_size = sizeof(viskores::Float64);


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

    viskores::UInt8 shape_id = 0;
    if(is_structured)
    {
      output["topologies/"+topo_name+"/coordset"] = coords_name;
      output["topologies/"+topo_name+"/type"] = "structured";

      viskores::cont::UnknownCellSet dyn_cells = data_set.GetCellSet();
      using Structured2D = viskores::cont::CellSetStructured<2>;
      using Structured3D = viskores::cont::CellSetStructured<3>;
      if(dyn_cells.CanConvert<Structured2D>())
      {
        Structured2D cells = dyn_cells.AsCellSet<Structured2D>();
        viskores::Id2 cell_dims = cells.GetCellDimensions();
        output["topologies/"+topo_name+"/elements/dims/i"] = (int) cell_dims[0];
        output["topologies/"+topo_name+"/elements/dims/j"] = (int) cell_dims[1];
      }
      else if(dyn_cells.CanConvert<Structured3D>())
      {
        Structured3D cells = dyn_cells.AsCellSet<Structured3D>();
        viskores::Id3 cell_dims = cells.GetCellDimensions();
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
      viskores::cont::UnknownCellSet dyn_cells = data_set.GetCellSet();

      using SingleType = viskores::cont::CellSetSingleType<>;
      using MixedType = viskores::cont::CellSetExplicit<>;

      if(dyn_cells.CanConvert<SingleType>())
      {
        SingleType cells = dyn_cells.AsCellSet<SingleType>();
        viskores::UInt8 shape_id = cells.GetCellShape(0);
        std::string conduit_name = GetBlueprintCellName(shape_id);
        output["topologies/"+topo_name+"/elements/shape"] = conduit_name;

        auto conn = cells.GetConnectivityArray(viskores::TopologyElementTagCell(),
                                               viskores::TopologyElementTagPoint());

        if(zero_copy)
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set_external(vtkh::GetVISKORESPointer(conn), conn.GetNumberOfValues());
        }
        else
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set(vtkh::GetVISKORESPointer(conn), conn.GetNumberOfValues());
        }
      }
      else if(vtkh::VISKORESDataSetInfo::IsSingleCellShape(dyn_cells, shape_id))
      {
        // If we are here, the we know that the cell set is explicit,
        // but only a single cell shape
        auto cells = dyn_cells.AsCellSet<viskores::cont::CellSetExplicit<>>();
        auto shapes = cells.GetShapesArray(viskores::TopologyElementTagCell(),
                                           viskores::TopologyElementTagPoint());

        std::string conduit_name = GetBlueprintCellName(shape_id);
        output["topologies/"+topo_name+"/elements/shape"] = conduit_name;

        auto conn = cells.GetConnectivityArray(viskores::TopologyElementTagCell(),
                                               viskores::TopologyElementTagPoint());

        if(zero_copy)
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set_external(vtkh::GetVISKORESPointer(conn), conn.GetNumberOfValues());
        }
        else
        {
          output["topologies/"+topo_name+"/elements/connectivity"].
            set(vtkh::GetVISKORESPointer(conn), conn.GetNumberOfValues());
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

        ViskoresBlueprintShapeMap(topo_ele["shape_map"]);

        size_t num_cells = static_cast<size_t>(cells.GetNumberOfCells());
        auto viskores_shapes  = cells.GetShapesArray(viskores::TopologyElementTagCell{}, viskores::TopologyElementTagPoint{});
        auto viskores_conn    = cells.GetConnectivityArray(viskores::TopologyElementTagCell{}, viskores::TopologyElementTagPoint{});
        auto viskores_offsets = cells.GetOffsetsArray(viskores::TopologyElementTagCell{}, viskores::TopologyElementTagPoint{});


        std::size_t conn_size = static_cast<std::size_t>(viskores_conn.GetNumberOfValues());

        if(zero_copy)
        {
            topo_ele["shapes"].set_external(vtkh::GetVISKORESPointer(viskores_shapes), num_cells);
            topo_ele["connectivity"].set_external(vtkh::GetVISKORESPointer(viskores_conn), conn_size);
            topo_ele["offsets"].set_external(vtkh::GetVISKORESPointer(viskores_offsets), num_cells);
        }
        else
        {
            topo_ele["shapes"].set(vtkh::GetVISKORESPointer(viskores_shapes), num_cells);
            topo_ele["connectivity"].set(vtkh::GetVISKORESPointer(viskores_conn), conn_size);
            topo_ele["offsets"].set(vtkh::GetVISKORESPointer(viskores_offsets), num_cells);
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
// helper to set conduit field values for from a vector style viskores array
//---------------------------------------------------------------------------//
template<typename T, int N>
void SetFieldValuesFromViskoresUnknownArrayHandleVec(viskores::cont::UnknownArrayHandle &dyn_handle,
                                                 bool zero_copy,
                                                 Node &output_values)
{
    static_assert(N > 1 && N < 4, "Vecs must be size 2 or 3");

    static const std::vector<std::string> comp_names = { "u", "v", "w"};

    bool try_zero_copy = zero_copy;


    for(index_t comp = 0; comp < N; comp++)
    {
      zero_copy = try_zero_copy;
      viskores::cont::ArrayHandleStride<T> stride_handle;

      Node &output_values_component = output_values[comp_names[comp]];

      if(zero_copy)
      {
        try
        {
          stride_handle = dyn_handle.ExtractComponent<T>(comp,viskores::CopyFlag::Off);
        }
        catch(...)
        {
          stride_handle = dyn_handle.ExtractComponent<T>(comp,viskores::CopyFlag::On);
          zero_copy = false;
        }
      }
      else
      {
        stride_handle = dyn_handle.ExtractComponent<T>(comp,viskores::CopyFlag::On);
      }

      viskores::cont::ArrayHandleBasic<T> basic_array = stride_handle.GetBasicArray();

      if(zero_copy)
      {
        output_values_component.set_external((T*) vtkh::GetVISKORESPointer(basic_array),
                                             stride_handle.GetNumberOfValues(),
                                             sizeof(T)*stride_handle.GetOffset(),   // starting offset in bytes
                                             sizeof(T)*stride_handle.GetStride());  // stride in bytes
      }
      else
      {
        output_values_component.set((T*) vtkh::GetVISKORESPointer(basic_array),
                                    stride_handle.GetNumberOfValues(),
                                    sizeof(T)*stride_handle.GetOffset(),   // starting offset in bytes
                                    sizeof(T)*stride_handle.GetStride());  // stride in bytes
      }
      
    }
}


//---------------------------------------------------------------------------//
// helper to set conduit field values for from a viskores array
//---------------------------------------------------------------------------//
template<typename T>
void SetFieldValuesFromViskoresUnknownArrayHandle(viskores::cont::UnknownArrayHandle &dyn_handle,
                                              bool zero_copy,
                                              Node &output_values)
{
    viskores::cont::ArrayHandleStride<T> stride_handle;
    if(zero_copy)
    {
      // if we cannot zero copy, extract component will throw an exception
      // and we can fall back to copying
      try
      {
        stride_handle = dyn_handle.ExtractComponent<T>(0,viskores::CopyFlag::Off);
      }catch(viskores::cont::Error &e)  // fall back to copy
      {
        stride_handle = dyn_handle.ExtractComponent<T>(0,viskores::CopyFlag::On);
        zero_copy = false;
      }
    }

    viskores::cont::ArrayHandleBasic<T> basic_array = stride_handle.GetBasicArray();
    if(zero_copy)
    {
      output_values.set_external(vtkh::GetVISKORESPointer(basic_array),
                                 stride_handle.GetNumberOfValues());
    }
    else // copy case
    {
      output_values.set(vtkh::GetVISKORESPointer(basic_array),
                        stride_handle.GetNumberOfValues());
    }
}



void
VTKHDataAdapter::ViskoresFieldToBlueprint(conduit::Node &output,
                                      const viskores::cont::Field &field,
                                      const std::string &topo_name,
                                      bool zero_copy)
{
  std::string name = field.GetName();
  std::string path = "fields/" + name;
  bool assoc_points = viskores::cont::Field::Association::Points == field.GetAssociation();
  bool assoc_cells  = viskores::cont::Field::Association::Cells == field.GetAssociation();
  //bool assoc_mesh  = viskores::cont::Field::ASSOC_WHOLE_MESH == field.GetAssociation();
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
  viskores::cont::UnknownArrayHandle dyn_handle = field.GetData();

  //
  // this can be literally anything. Lets do some exhaustive casting
  //
  if (dyn_handle.IsValueType<viskores::Vec<viskores::Float32, 3>>())
  {
      SetFieldValuesFromViskoresUnknownArrayHandleVec<viskores::Float32, 3>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<viskores::Vec<viskores::Float64, 3>>())
  {
      SetFieldValuesFromViskoresUnknownArrayHandleVec<viskores::Float64, 3>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<viskores::Vec<viskores::Int32, 3>>())
  {
      SetFieldValuesFromViskoresUnknownArrayHandleVec<viskores::Int32, 3>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<viskores::Vec<viskores::Float32, 2>>())
  {
      SetFieldValuesFromViskoresUnknownArrayHandleVec<viskores::Float32, 2>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<viskores::Vec<viskores::Float64, 2>>())
  {
      SetFieldValuesFromViskoresUnknownArrayHandleVec<viskores::Float64, 2>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if (dyn_handle.IsValueType<viskores::Vec<viskores::Int32, 2>>())
  {
      SetFieldValuesFromViskoresUnknownArrayHandleVec<viskores::Int32, 2>(dyn_handle,
          zero_copy,
          output_values);
  }
  else if(dyn_handle.IsValueType<viskores::Float32>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::Float32>(dyn_handle,
                                  zero_copy,
                                  output_values);
  }
  else if(dyn_handle.IsValueType<viskores::Float64>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::Float64>(dyn_handle,
                                  zero_copy,
                                  output_values);
  }
  else if(dyn_handle.IsValueType<viskores::Int8>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::Int8>(dyn_handle,
                              zero_copy,
                              output_values);

  }
  else if(dyn_handle.IsValueType<viskores::Int32>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::Int32>(dyn_handle,
                              zero_copy,
                              output_values);
  }
  else if(dyn_handle.IsValueType<viskores::Int64>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::Int64>(dyn_handle,
                              zero_copy,
                              output_values);

  }
  else if(dyn_handle.IsValueType<viskores::UInt32>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::UInt32>(dyn_handle,
                              zero_copy,
                              output_values);
  }
  else if(dyn_handle.IsValueType<viskores::UInt8>())
  {
    SetFieldValuesFromViskoresUnknownArrayHandle<viskores::UInt8>(dyn_handle,
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
VTKHDataAdapter::CheckShapeMapVsViskoresShapeIds(const Node &shape_map)
{
    bool res = true;
    Node ref_map;

    VTKHDataAdapter::ViskoresBlueprintShapeMap(ref_map);
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
VTKHDataAdapter::ViskoresBlueprintShapeMap(conduit::Node &output)
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
  std::map<int, std::map<std::string,viskores::cont::DataSet>> domain_map;
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
        viskores::cont::DataSet &dataset = topo_it.second;
        VTKHDataAdapter::ViskoresToBlueprintDataSet(&dataset, dom, topo_name, zero_copy);
      }
    }
  }
  catch (conduit::Error error)
  {
     err_msg = error.message();
     success = false;
  }
  catch (viskores::cont::Error error)
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
    ASCENT_ERROR("Failed to convert Viskores data set to blueprint: " << err_msg);
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
      viskores::cont::DataSet viskores_dom;
      viskores::Id domain_id;
      int cycle = dset->GetCycle();
      dset->GetDomain(i, viskores_dom, domain_id);
      VTKHDataAdapter::ViskoresToBlueprintDataSet(&viskores_dom,dom, "topo", zero_copy);
      dom["state/domain_id"] = (int) domain_id;
      dom["state/cycle"] = cycle;
    }
  }
  catch (conduit::Error error)
  {
      err_msg = error.message();
      success = false;
  }
  catch (viskores::cont::Error error)
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
    ASCENT_ERROR("Failed to convert Viskores data set to blueprint: " << err_msg);
  }
}

void
VTKHDataAdapter::ViskoresToBlueprintDataSet(const viskores::cont::DataSet *dset,
                                        conduit::Node &node,
                                        const std::string &topo_name,
                                        bool zero_copy)
{
  //
  // with viskores, we have no idea what the type is of anything inside
  // dataset, so we have to ask all fields, cell sets anc coordinate systems.
  //

  bool is_empty = ViskoresTopologyToBlueprint(node, *dset, topo_name, zero_copy);

  if(!is_empty)
  {
    const viskores::Id num_fields = dset->GetNumberOfFields();
    for(viskores::Id i = 0; i < num_fields; ++i)
    {
      viskores::cont::Field field = dset->GetField(i);
      // as of Viskores 2.0, coordinates are also stored as Viskores fields
      // skip wrapping coords as a field, since they are 
      // already captured in the blueprint coordset
      if (!dset->HasCoordinateSystem(field.GetName()))
      {
          ViskoresFieldToBlueprint(node, field, topo_name, zero_copy);
      }
    }
  }
}


};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
