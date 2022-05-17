// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DOF_ACCESS_HPP
#define DRAY_DOF_ACCESS_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <typename DofT> struct SharedDofPtr;
template <typename DofT> struct ReadDofPtr;


//
// SharedDofPtr - support for double indirection  val = dof_array[ele_offsets[dof_idx]];
//
template <typename DofT> struct SharedDofPtr
{
  const int32 *m_offset_ptr; // Points to element dof map, [dof_idx]-->offset
  const DofT *m_dof_ptr; // Beginning of dof data array, i.e. offset==0.

  // Iterator offset dereference operator.
  DRAY_EXEC const DofT &operator[] (const int32 i) const
  {
    return m_dof_ptr[m_offset_ptr[i]];
  }

  // Iterator offset operator.
  DRAY_EXEC SharedDofPtr operator+ (const int32 &i) const
  {
    return { m_offset_ptr + i, m_dof_ptr };
  }

  // Iterator pre-increment operator.
  DRAY_EXEC SharedDofPtr &operator++ ()
  {
    ++m_offset_ptr;
    return *this;
  }

  // Iterator dereference operator.
  DRAY_EXEC const DofT &operator* () const
  {
    return m_dof_ptr[*m_offset_ptr];
  }

  DRAY_EXEC operator ReadDofPtr<DofT>() const;
};


// For now duplicated SharedDofPtr and renamed ReadDofPtr, eventually should replace.
template <typename DofT> struct ReadDofPtr
{
  const int32 *m_offset_ptr; // Points to element dof map, [dof_idx]-->offset
  const DofT *m_dof_ptr; // Beginning of dof data array, i.e. offset==0.

  // Iterator offset dereference operator.
  DRAY_EXEC const DofT &operator[] (const int32 i) const
  {
    return m_dof_ptr[m_offset_ptr[i]];
  }

  // Iterator offset operator.
  DRAY_EXEC ReadDofPtr operator+ (const int32 &i) const
  {
    return { m_offset_ptr + i, m_dof_ptr };
  }

  // Iterator pre-increment operator.
  DRAY_EXEC ReadDofPtr &operator++ ()
  {
    ++m_offset_ptr;
    return *this;
  }

  // Iterator dereference operator.
  DRAY_EXEC const DofT &operator* () const
  {
    return m_dof_ptr[*m_offset_ptr];
  }

  DRAY_EXEC operator SharedDofPtr<DofT>() const;
};

// Implicit conversions now, renaming later.
template <typename DofT>
DRAY_EXEC
SharedDofPtr<DofT>::operator ReadDofPtr<DofT>() const { return {m_offset_ptr, m_dof_ptr}; }

template <typename DofT>
DRAY_EXEC
ReadDofPtr<DofT>::operator SharedDofPtr<DofT>() const { return {m_offset_ptr, m_dof_ptr}; }



template <typename DofT> struct WriteDofPtr
{
  const int32 *m_offset_ptr; // Points to element dof map, [dof_idx]-->offset
  DofT *m_dof_ptr; // Beginning of dof data array, i.e. offset==0.

  DRAY_EXEC SharedDofPtr<DofT> to_readonly_dof_ptr() const
  {
    return { m_offset_ptr, m_dof_ptr };
  }

  // Iterator offset dereference operator.
  DRAY_EXEC DofT &operator[] (const int32 i)
  {
    return m_dof_ptr[m_offset_ptr[i]];
  }

  // Iterator offset operator.
  DRAY_EXEC WriteDofPtr operator+ (const int32 &i) const
  {
    return { m_offset_ptr + i, m_dof_ptr };
  }

  // Iterator offset assignment.
  DRAY_EXEC WriteDofPtr &operator+= (const int32 i)
  {
    m_offset_ptr += i;
    return *this;
  }

  // Iterator pre-increment operator.
  DRAY_EXEC WriteDofPtr &operator++ ()
  {
    ++m_offset_ptr;
    return *this;
  }

  // Iterator dereference operator.
  DRAY_EXEC DofT &operator* ()
  {
    return m_dof_ptr[*m_offset_ptr];
  }
};

} //namespace dray

#endif // DRAY_ELEMENT_HPP
