//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
// forward declare
#if defined(ASCENT_DRAY_ENABLED)
namespace dray
{
  class Collection;
} // namespace dray
#endif

namespace ascent
{


#if defined(ASCENT_VTKM_ENABLED)
// forward declare
class VTKHCollection;
#endif


class DataObject
{
public:
  enum class Source { VTKH, LOW_BP, HIGH_BP, DRAY, INVALID};
  DataObject();
  //
  // Constructors take ownership of pointers
  //

  DataObject(conduit::Node *dataset);
  void reset(conduit::Node *dataset);
  void reset(std::shared_ptr<conduit::Node> dataset);
  void reset_all();
  bool is_valid() const { return m_source != Source::INVALID;};
  void name(const std::string n);
  std::string name() const;

#if defined(ASCENT_VTKM_ENABLED)
  DataObject(VTKHCollection *dataset);
  std::shared_ptr<VTKHCollection> as_vtkh_collection();

  bool                            is_vtkh_coll_exists() const { return m_vtkh != nullptr; }
  void                            reset_vtkh_collection();

#endif
#if defined(ASCENT_DRAY_ENABLED)
  DataObject(dray::Collection *dataset);
  std::shared_ptr<dray::Collection> as_dray_collection();
#endif
  std::shared_ptr<conduit::Node>  as_low_order_bp();
  std::shared_ptr<conduit::Node>  as_high_order_bp();
  std::shared_ptr<conduit::Node>  as_node();          // just return the coduit node
  DataObject::Source              source() const;
  std::string source_string() const;
protected:
  std::shared_ptr<conduit::Node>  m_low_bp;
  std::shared_ptr<conduit::Node>  m_high_bp;
#if defined(ASCENT_VTKM_ENABLED)
  std::shared_ptr<VTKHCollection> m_vtkh;
#endif
#if defined(ASCENT_DRAY_ENABLED)
  std::shared_ptr<dray::Collection> m_dray;
#endif

  Source m_source;
  std::string m_name;
};

//-----------------------------------------------------------------------------
};
#endif
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
