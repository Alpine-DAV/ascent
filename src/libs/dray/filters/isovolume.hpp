#ifndef DRAY_ISOVOLUME_HPP
#define DRAY_ISOVOLUME_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class Isovolume
{
public:
  using RangeType = Vec<Float, 2>;
  Isovolume();
  virtual ~Isovolume();

  void set_field(std::string field_name);
  void set_invert(bool invert);
  void set_range(RangeType range);

  /**
   * @brief Clips the input dataset.
   * @param data_set The dataset.
   * @return A dataset containing the clipped cells.
   *
   */
  Collection execute(Collection &collection);

protected:
  std::string m_field;
  RangeType m_range;
  bool m_invert;
};

};//namespace dray

#endif//DRAY_CLIP_HPP
