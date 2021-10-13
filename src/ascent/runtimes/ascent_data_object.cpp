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
#include "ascent_metadata.hpp"

#if defined(ASCENT_VTKM_ENABLED)
#include "ascent_vtkh_collection.hpp"
#include "ascent_vtkh_data_adapter.hpp"
#endif

#if defined(ASCENT_DRAY_ENABLED)
#include <dray/data_model/collection.hpp>
#include <dray/io/blueprint_reader.hpp>
#endif

#include "ascent_transmogrifier.hpp"

#include <ascent_logging.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

namespace detail
{

void add_metadata(conduit::Node &dataset)
{
    int cycle = -1;
    double time = -1.;

    if(Metadata::n_metadata.has_path("cycle"))
    {
      cycle = Metadata::n_metadata["cycle"].to_int32();
    }
    if(Metadata::n_metadata.has_path("time"))
    {
      time = Metadata::n_metadata["time"].to_float64();
    }

    const int num_domains = dataset.number_of_children();
    for(int i = 0; i < num_domains; ++i)
    {
      conduit::Node &dom = dataset.child(i);
      if(cycle != -1)
      {
        dom["state/cycle"] = cycle;
      }
      if(time != -1.0)
      {
        dom["state/time"] = time;
      }
    }
}

} // namespace detail

DataObject::DataObject()
  : m_low_bp(nullptr),
    m_high_bp(nullptr),
#if defined(ASCENT_VTKM_ENABLED)
    m_vtkh(nullptr),
#endif
#if defined(ASCENT_DRAY_ENABLED)
    m_dray(nullptr),
#endif
    m_source(Source::INVALID)
{
  m_name = "default";
}

#if defined(ASCENT_VTKM_ENABLED)
DataObject::DataObject(VTKHCollection *dataset)
  : m_low_bp(nullptr),
    m_high_bp(nullptr),
    m_vtkh(dataset),
#if defined(ASCENT_DRAY_ENABLED)
    m_dray(nullptr),
#endif
    m_source(Source::VTKH)
{
  m_name = "default";
}
#endif

#if defined(ASCENT_DRAY_ENABLED)
DataObject::DataObject(dray::Collection *dataset)
  : m_low_bp(nullptr),
    m_high_bp(nullptr),
#if defined(ASCENT_VTKM_ENABLED)
    m_vtkh(nullptr),
#endif
    m_dray(dataset),
    m_source(Source::DRAY)
{
  m_name = "default";
}
#endif

DataObject::DataObject(conduit::Node *dataset)
  : m_low_bp(nullptr),
    m_high_bp(nullptr)
#if defined(ASCENT_VTKM_ENABLED)
    ,m_vtkh(nullptr)
#endif
#if defined(ASCENT_DRAY_ENABLED)
    ,m_dray(nullptr)
#endif
{
  reset(dataset);
  m_name = "default";
}

void DataObject::name(const std::string n)
{
  m_name = n;
}

std::string DataObject::name() const
{
  return m_name;
}

void DataObject::reset_all()
{
  m_source = Source::INVALID;
  std::shared_ptr<conduit::Node>  null_low(nullptr);
  std::shared_ptr<conduit::Node>  null_high(nullptr);
  m_low_bp = null_low;
  m_high_bp = null_high;

#if defined(ASCENT_VTKM_ENABLED)
  std::shared_ptr<VTKHCollection> null_vtkh(nullptr);
  m_vtkh = null_vtkh;
#endif

#if defined(ASCENT_DRAY_ENABLED)
  std::shared_ptr<dray::Collection> null_dray(nullptr);
  m_dray = null_dray;
#endif
}

void DataObject::reset(std::shared_ptr<conduit::Node> dataset)
{
  bool high_order = Transmogrifier::is_high_order(*dataset.get());

  std::shared_ptr<conduit::Node>  null_low(nullptr);
  std::shared_ptr<conduit::Node>  null_high(nullptr);
  m_low_bp = null_low;
  m_high_bp = null_high;

#if defined(ASCENT_VTKM_ENABLED)
  std::shared_ptr<VTKHCollection> null_vtkh(nullptr);
  m_vtkh = null_vtkh;
#endif

#if defined(ASCENT_DRAY_ENABLED)
  std::shared_ptr<dray::Collection> null_dray(nullptr);
  m_dray = null_dray;
#endif
  if(high_order)
  {
    m_high_bp = dataset;
    m_source = Source::HIGH_BP;
  }
  else
  {
    m_low_bp = dataset;
    m_source = Source::LOW_BP;
  }
}

void DataObject::reset(conduit::Node *dataset)
{
  bool high_order = Transmogrifier::is_high_order(*dataset);
  std::shared_ptr<conduit::Node>  bp(dataset);

  std::shared_ptr<conduit::Node>  null_low(nullptr);
  std::shared_ptr<conduit::Node>  null_high(nullptr);
  m_low_bp = null_low;
  m_high_bp = null_high;

#if defined(ASCENT_VTKM_ENABLED)
  std::shared_ptr<VTKHCollection> null_vtkh(nullptr);
  m_vtkh = null_vtkh;
#endif

#if defined(ASCENT_DRAY_ENABLED)
  std::shared_ptr<dray::Collection> null_dray(nullptr);
  m_dray = null_dray;
#endif

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

#if defined(ASCENT_DRAY_ENABLED)
std::shared_ptr<dray::Collection> DataObject::as_dray_collection()
{
  if(m_source == Source::INVALID)
  {
    ASCENT_ERROR("Source never initialized: default constructed");
  }

  if(m_dray != nullptr)
  {
    return m_dray;
  }
  else
  {
    if(m_source == Source::HIGH_BP)
    {
      std::shared_ptr<dray::Collection> collection(new dray::Collection());
      const int num_domains = m_high_bp->number_of_children();
      for(int i = 0; i < num_domains; ++i)
      {
        dray::DataSet dset = dray::BlueprintReader::blueprint_to_dray(m_high_bp->child(i));
        collection->add_domain(dset);
      }

      m_dray = collection;
      return m_dray;
    }
    else
    {
      // attempt to conver this to low order and go
      std::shared_ptr<conduit::Node> low_order = as_low_order_bp();
      std::shared_ptr<dray::Collection> collection(new dray::Collection());
      const int domains = low_order->number_of_children();
      for(int i = 0; i < domains; ++i)
      {
        dray::DataSet dset = dray::BlueprintReader::blueprint_to_dray(low_order->child(i));
        collection->add_domain(dset);
      }

      m_dray = collection;
      return m_dray;
    }

  }

  ASCENT_ERROR("this should never happen");
  return nullptr;
}
#endif

#if defined(ASCENT_VTKM_ENABLED)
std::shared_ptr<VTKHCollection> DataObject::as_vtkh_collection()
{
  if(m_source == Source::INVALID)
  {
    ASCENT_ERROR("Source never initialized: default constructed");
  }

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
    conduit::Node n_poly;
    conduit::Node *to_vtkh = nullptr;
    
    if (m_low_bp != nullptr)
    {
      if (Transmogrifier::is_poly(*m_low_bp))
      {
        Transmogrifier::to_poly(*m_low_bp, n_poly);
        to_vtkh = &n_poly;
        zero_copy = false;
      }
      else
      {
        to_vtkh = &(*m_low_bp);
      }
    }

    // convert to vtkh
    std::shared_ptr<VTKHCollection>
      vtkh_dset(VTKHDataAdapter::BlueprintToVTKHCollection(*to_vtkh, zero_copy));

    m_vtkh = vtkh_dset;
    
    return m_vtkh;
  }

  ASCENT_ERROR("this should never happen");
  return nullptr;
}

void DataObject::reset_vtkh_collection()
{
  if(m_source != Source::VTKH)
    m_vtkh.reset();
}
#endif

std::shared_ptr<conduit::Node>  DataObject::as_low_order_bp()
{
  if(m_source == Source::INVALID)
  {
    ASCENT_ERROR("Source never initialized: default constructed");
  }

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
    bool zero_copy = true;
    VTKHDataAdapter::VTKHCollectionToBlueprintDataSet(m_vtkh.get(), *out_data, true);

    detail::add_metadata(*out_data);
    std::shared_ptr<conduit::Node> bp(out_data);
    m_low_bp = bp;
  }
#endif

  return m_low_bp;
}

std::shared_ptr<conduit::Node>  DataObject::as_high_order_bp()
{
  if(m_source == Source::INVALID)
  {
    ASCENT_ERROR("Source never initialized: default constructed");
  }
#ifdef ASCENT_MFEM_ENABLED
  if(m_high_bp!= nullptr)
  {
    return m_high_bp;
  }

  ASCENT_ERROR("converting from low order to high order is not currenlty supported");
#else
  ASCENT_ERROR("Cannot provide high order blueprint. MFEM support not enabled.");
#endif

  return nullptr;
}

std::shared_ptr<conduit::Node>  DataObject::as_node()
{
  if(m_source == Source::INVALID)
  {
    ASCENT_ERROR("Source never initialized: default constructed");
  }
#if defined(ASCENT_VTKM_ENABLED)
  if(m_source == Source::VTKH && m_low_bp == nullptr)

  {
    conduit::Node *out_data = new conduit::Node();
    bool zero_copy = true;
    VTKHDataAdapter::VTKHCollectionToBlueprintDataSet(m_vtkh.get(), *out_data, true);

    detail::add_metadata(*out_data);
    std::shared_ptr<conduit::Node> bp(out_data);
    m_low_bp = bp;
  }
#endif
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

std::string DataObject::source_string() const
{
  std::string res;
  if(m_source == Source::INVALID)
  {
    res = "Invalid";
  }
  if(m_source == Source::VTKH)
  {
    res = "VTKH";
  }
  if(m_source == Source::LOW_BP)
  {
    res = "LOW_BP";
  }
  if(m_source == Source::HIGH_BP)
  {
    res = "HIGH_BP";
  }
  if(m_source == Source::DRAY)
  {
    res = "DRAY";
  }
  return res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
