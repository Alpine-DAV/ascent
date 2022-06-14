#include <dray/filters/clipfield.hpp>

#include <dray/dispatcher.hpp>
#include <dray/filters/subset.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{

}//namespace detail

ClipField::ClipField() : m_clip_value(0.), m_field_name(), m_invert(false)
{
}

ClipField::~ClipField()
{
}

void
ClipField::set_clip_value(Float value)
{
  m_clip_value = value;
}

void
ClipField::set_field(const std::string &field)
{
  m_field_name = field;
}

void
ClipField::set_invert_clip(bool value)
{
  m_invert = value;
}

Float
ClipField::clip_value() const
{
  return m_clip_value;
}

const std::string &
ClipField::field() const
{
  return m_field_name;
}

bool
ClipField::invert() const
{
  return m_invert;
}

Collection
ClipField::execute(Collection &collection)
{
   Collection result;
   return result;
}

}//namespace dray
