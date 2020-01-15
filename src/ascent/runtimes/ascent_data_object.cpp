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
/// file: ascent_data_object.hpp
///
//-----------------------------------------------------------------------------

#include "ascent_data_object.hpp"

#if defined(ASCENT_VTKM_ENABLED)
#include "ascent_vtkh_collection.hpp"
#include "ascent_vtkh_data_adapter.hpp"
#endif

#include "ascent_mfem_data_adapter.hpp"
#include "ascent_transmogrifier.hpp"

#include <ascent_logging.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

#if defined(ASCENT_VTKM_ENABLED)
DataObject::DataObject(VTKHCollection *dataset)
  : m_low_bp(nullptr),
    m_high_bp(nullptr),
    m_vtkh(dataset),
    m_source(Source::VTKH)
{

}
#endif

DataObject::DataObject(conduit::Node *dataset)
  : m_low_bp(nullptr),
    m_high_bp(nullptr)
#if defined(ASCENT_VTKM_ENABLED)
    ,m_vtkh(nullptr)
#endif
{
  bool high_order = Transmogrifier::is_high_order(*dataset);
  std::shared_ptr<conduit::Node>  bp(dataset);
  if(high_order)
  {
    m_high_bp = bp;
    m_source = Source::HIGH_BP;
  }
  else
  {
    m_low_bp = bp;
    m_source = Source::LOW_BP;
  }
}

#if defined(ASCENT_VTKM_ENABLED)
std::shared_ptr<VTKHCollection> DataObject::as_vtkh_collection()
{
  if(m_vtkh != nullptr)
  {
    return m_vtkh;
  }
  else
  {
    if(m_source == Source::HIGH_BP && m_low_bp == nullptr)
    {
      std::shared_ptr<conduit::Node>  low_order(Transmogrifier::low_order(*m_high_bp));
      m_low_bp = low_order;
    }

    bool zero_copy = true;
    // convert to vtkh
    std::shared_ptr<VTKHCollection>
      vtkh_dset(VTKHDataAdapter::BlueprintToVTKHCollection(*m_low_bp, zero_copy));

     m_vtkh = vtkh_dset;

    return m_vtkh;
  }

  ASCENT_ERROR("this should never happen");
  return nullptr;
}
#endif

std::shared_ptr<conduit::Node>  DataObject::as_low_order_bp()
{
  if(m_low_bp != nullptr)
  {
    return m_low_bp;
  }

  if(m_source == Source::HIGH_BP)
  {
    std::shared_ptr<conduit::Node>  low_order(Transmogrifier::low_order(*m_high_bp));
    m_low_bp = low_order;
  }
#if defined(ASCENT_VTKM_ENABLED)
  else if(m_source == Source::VTKH)
  {
    conduit::Node *out_data = new conduit::Node();
    VTKHDataAdapter::VTKHCollectionToBlueprintDataSet(m_vtkh.get(), *out_data);

    std::shared_ptr<conduit::Node> bp(out_data);
    m_low_bp = bp;
  }
#endif

  return m_low_bp;
}

std::shared_ptr<conduit::Node>  DataObject::as_high_order_bp()
{
  if(m_high_bp!= nullptr)
  {
    return m_high_bp;
  }

  ASCENT_ERROR("converting from low order to high order is not currenlty supported");

  return nullptr;
}

std::shared_ptr<conduit::Node>  DataObject::as_node()
{
  if(m_source == Source::VTKH)
  {
    ASCENT_ERROR("Bad");
  }
  if(m_high_bp != nullptr)
  {
    return m_high_bp;
  }

  if(m_low_bp != nullptr)
  {
    return m_low_bp;
  }

  ASCENT_ERROR("this should never happen");
  return nullptr;
}

DataObject::Source DataObject::source() const
{
  return m_source;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
