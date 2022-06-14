#ifndef DRAY_CLIPFIELD_HPP
#define DRAY_CLIPFIELD_HPP

#include <dray/data_model/collection.hpp>
#include <string>

namespace dray
{

class ClipField
{
protected:
  Float m_clip_value;
  std::string   m_field_name;
  bool m_invert;
public:
  ClipField();
  ~ClipField();

  void set_clip_value(Float value);
  void set_field(const std::string &field);
  void set_invert_clip(bool value);

  Float clip_value() const;
  const std::string &field() const;
  bool invert() const;

  /**
   * @brief Extracts cells using a field as a guide.
   * @param data_set The dataset.
   * @return A dataset containing the preserved cells.
   *
   */
  Collection execute(Collection &collection);
};

};//namespace dray

#endif//DRAY_CLIPFIELD_HPP
