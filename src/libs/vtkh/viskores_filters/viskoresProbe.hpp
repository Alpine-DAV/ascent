#ifndef VTK_H_VISKORES_PROBE_HPP
#define VTK_H_VISKORES_PROBE_HPP

#include <viskores/cont/DataSet.h>

namespace vtkh
{

class viskoresProbe
{
public:
  viskoresProbe();
  ~viskoresProbe();

  void setPoints(viskores::cont::ArrayHandle<viskores::Float64> xs,
                 viskores::cont::ArrayHandle<viskores::Float64> ys,
                 viskores::cont::ArrayHandle<viskores::Float64> zs);

  void setBoxDims(const viskores::Vec<viskores::Float64,3> dims);
  void setBoxOrigin(const viskores::Vec<viskores::Float64,3> origin);
  void setBoxSpacing(const viskores::Vec<viskores::Float64,3> spacing);
  void setGeometry(const viskores::cont::DataSet &geometry);

  void setInvalidValue(const viskores::Float64 invalid_value);

  viskores::cont::DataSet Run(viskores::cont::DataSet &input);

protected:
  enum SampleMode { 
                    NONE,
                    POINTS,
                    BOX,
                    GEOMETRY
                  };

  int m_mode;

  viskores::Float64 m_invalid_value;

  // points details
  viskores::cont::ArrayHandle<viskores::Float64> m_points_xs;
  viskores::cont::ArrayHandle<viskores::Float64> m_points_ys;
  viskores::cont::ArrayHandle<viskores::Float64> m_points_zs;

  // box details
  viskores::Vec<viskores::Float64,3> m_box_dims;
  viskores::Vec<viskores::Float64,3> m_box_origin;
  viskores::Vec<viskores::Float64,3> m_box_spacing;

  viskores::cont::DataSet m_geometry;

};
}
#endif
