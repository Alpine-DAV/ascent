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

#ifndef ASCENT_DATA_OBJECT_HPP
#define ASCENT_DATA_OBJECT_HPP

#include <ascent.hpp>
#include <conduit.hpp>
#include <memory>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


#if defined(ASCENT_VTKM_ENABLED)
  class VTKHCollection;
#endif

class DataObject
{
public:
  enum class Source { VTKH, LOW_BP, HIGH_BP};
  DataObject() = delete;
  //
  // Constructors take ownership of pointers
  //

  DataObject(conduit::Node *dataset);

#if defined(ASCENT_VTKM_ENABLED)
  DataObject(VTKHCollection *dataset);
  std::shared_ptr<VTKHCollection> as_vtkh_collection();
#endif
  std::shared_ptr<conduit::Node>  as_low_order_bp();
  std::shared_ptr<conduit::Node>  as_high_order_bp();
  std::shared_ptr<conduit::Node>  as_node();          // just return the coduit node
  DataObject::Source              source() const;
protected:
  std::shared_ptr<conduit::Node>  m_low_bp;
  std::shared_ptr<conduit::Node>  m_high_bp;
#if defined(ASCENT_VTKM_ENABLED)
  std::shared_ptr<VTKHCollection> m_vtkh;
#endif

  Source m_source;
};

//-----------------------------------------------------------------------------
};
#endif
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
