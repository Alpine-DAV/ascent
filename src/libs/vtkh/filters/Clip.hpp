#ifndef VTK_H_CLIP_HPP
#define VTK_H_CLIP_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>
#include <memory>

namespace vtkh
{

class VTKH_API Clip: public Filter
{
public:
  Clip();
  virtual ~Clip();
  std::string GetName() const override;
  void SetBoxClip(const vtkm::Bounds &clipping_bounds);
  void SetSphereClip(const double center[3], const double radius);
  void SetPlaneClip(const double origin[3], const double normal[3]);

  void Set2PlaneClip(const double origin1[3],
                     const double normal1[3],
                     const double origin2[3],
                     const double normal2[3]);

  void Set3PlaneClip(const double origin1[3],
                     const double normal1[3],
                     const double origin2[3],
                     const double normal2[3],
                     const double origin3[3],
                     const double normal3[3]);

  void SetCellSetIndex(vtkm::Id index);
  void SetInvertClip(bool invert);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  struct InternalsType;
  std::shared_ptr<InternalsType> m_internals;
  bool m_invert;
  bool m_do_multi_plane;
};

} //namespace vtkh
#endif
