#ifndef DRAY_MESH_THRESHOLD_HPP
#define DRAY_MESH_THRESHOLD_HPP

#include <dray/data_model/collection.hpp>
#include <string>

namespace dray
{

class MeshThreshold
{
protected:
  Range m_range;
  std::string m_field_name;
  bool m_return_all_in_range;
public:
  MeshThreshold();
  ~MeshThreshold();

  // NOTE: Some other parts of the code would say: void upper_threshold(double value)
  //       to set.
  void set_upper_threshold(Float value);
  void set_lower_threshold(Float value);
  void set_field(const std::string &field_name);
  void set_all_in_range(bool value);

  Float get_upper_threshold() const;
  Float get_lower_threshold() const;
  bool get_all_in_range() const;
  const std::string &get_field() const;

  /**
   * @brief Extracts cells using a field as a guide.
   * @param data_set The dataset.
   * @return A dataset containing the preserved cells.
   *
   */
  Collection execute(Collection &collection);
};

};//namespace dray

#endif//DRAY_MESH_THRESHOLD_HPP
