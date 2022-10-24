#include "isovolume.hpp"

#include <dray/error.hpp>
#include <dray/filters/clipfield.hpp>

namespace dray
{

Isovolume::Isovolume()
  : m_field(), m_range({0., 0.}), m_invert(false)
{
  // Do nothing
}

Isovolume::~Isovolume()
{
  // Do nothing
}

void
Isovolume::set_field(std::string field)
{
  m_field = field;
}

void
Isovolume::set_invert(bool invert)
{
  m_invert = invert;
}

void
Isovolume::set_range(RangeType range)
{
  m_range = range;
}

Collection
Isovolume::execute(Collection &c)
{
  if(m_field.empty())
  {
    DRAY_ERROR("Isovolume::execute() called with no isovalue set.");
  }

  if(!c.has_field(m_field))
  {
    DRAY_ERROR("The given collection does not contain a field called '"
      << m_field << "'.");
  }

  if(!(m_range[0] < m_range[1]))
  {
    DRAY_ERROR("The given range is invalid, " << m_range[0] << " is not less than "
      << m_range[1] << "!");
  }

  // First clip everything less than the max.
  ClipField clipmax;
  clipmax.set_clip_value(m_range[1]);
  clipmax.set_field(m_field);
  clipmax.set_invert_clip(m_invert);
  Collection retval = clipmax.execute(c);

  // Make sure the first clip operation resulted in data.
  bool hasdata = false;
  for(DataSet &domain : retval.domains())
  {
    if(domain.number_of_meshes() > 0)
    {
      hasdata = true;
      break;
    }
  }
  if(!hasdata)
  {
    // Q: Should we throw an exception? Currently returning the empty collection.
    return retval;
  }

  // Since there is data we will continue
  // Clip everything greater than the min.
  ClipField clipmin;
  clipmin.set_clip_value(m_range[0]);
  clipmin.set_field(m_field);
  clipmin.set_invert_clip(!m_invert);
  return clipmin.execute(retval);
}

}//namespace dray