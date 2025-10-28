#ifndef VTK_H_SLICE_HPP
#define VTK_H_SLICE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <viskores/rendering/Camera.h>


namespace vtkh
{

typedef viskores::rendering::Camera viskoresCamera;

class VTKH_API Slice : public Filter
{
public:
  Slice();
  virtual ~Slice();
  std::string GetName() const override;
  void AddPlane(viskores::Vec<viskores::Float32,3> point, viskores::Vec<viskores::Float32,3> normal);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::vector<viskores::Vec<viskores::Float32,3>> m_points;
  std::vector<viskores::Vec<viskores::Float32,3>> m_normals;
};


///
/// Slice Filter that uses Viskores support for Implicit Functions
/// to support Spherical and Cylindrical slicing
///
class VTKH_API SliceImplicit : public Filter
{
public:
  SliceImplicit();
  virtual ~SliceImplicit();
  std::string GetName() const override;

  void SetBoxSlice(const viskores::Bounds &slice_bounds);
  void SetSphereSlice(const double center[3], const double radius);
  void SetCylinderSlice(const double center[3],
                        const double axis[3],
                        const double radius);
  void SetPlaneSlice(const double origin[3], const double normal[3]);

  //
  // TODO: multi plane needs more work
  // void Set2PlaneSlice(const double origin1[3],
  //                     const double normal1[3],
  //                     const double origin2[3],
  //                     const double normal2[3]);
  //
  // void Set3PlaneSlice(const double origin1[3],
  //                     const double normal1[3],
  //                     const double origin2[3],
  //                     const double normal2[3],
  //                     const double origin3[3],
  //                     const double normal3[3]);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  struct InternalsType;
  bool m_do_multi_plane;
  std::shared_ptr<InternalsType> m_internals;
};


class VTKH_API AutoSliceLevels : public Filter
{
public:
  AutoSliceLevels();
  virtual ~AutoSliceLevels();
  std::string GetName() const override;
  void SetNormal(viskores::Vec<viskores::Float32,3> normal);
  void SetLevels(int levels);
  void SetField(std::string field_name);
  viskoresCamera* GetCamera();
  viskores::Bounds GetDataBounds();
  viskores::Float32 GetRadius();
  viskores::Vec<viskores::Float32,3> GetNormal();
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::vector<viskores::Vec<viskores::Float32,3>> m_normals;
  int m_levels;
  std::string m_field_name;
  viskoresCamera *m_camera;
  viskores::Bounds m_bounds;
  viskores::Float32 m_radius;
  viskores::Vec<viskores::Float32,3> m_normal;
};

} //namespace vtkh
#endif
