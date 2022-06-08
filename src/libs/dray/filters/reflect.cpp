#include <dray/filters/reflect.hpp>

#include <dray/dispatcher.hpp>
#include <dray/data_model/elem_utils.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{

namespace detail
{

template<typename MeshElem>
DataSet
reflect_execute(UnstructuredMesh<MeshElem> &mesh,
                const Vec<Float,3> point,
                const Vec<Float,3> normal)
{
  DRAY_LOG_OPEN("reflect");

  // im afraid of lambda capture
  const Vec<Float,3> lpoint = point;
  const Vec<Float,3> lnormal = normal;

  GridFunction<3u> input_gf = mesh.get_dof_data();
  // shallow copy everything
  GridFunction<3u> output_gf = input_gf;
  // deep copy values
  Array<Vec<Float, 3>> points;
  array_copy (points, input_gf.m_values);

  Vec<Float,3> *points_ptr = points.get_device_ptr();
  const int size = points.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<Float,3> dof = points_ptr[i];
    Vec<Float,3> dir = dof - lpoint;
    Float dist = dot(dir, lnormal);
    dof = dof - Float(2.f) * dist * lnormal;
    points_ptr[i] = dof;
  });

  // replace the input values
  output_gf.m_values = points;

  UnstructuredMesh<MeshElem> out_mesh(output_gf, mesh.order());
  std::shared_ptr<UnstructuredMesh<MeshElem>> omesh
    = std::make_shared<UnstructuredMesh<MeshElem>>(out_mesh);
  DataSet dataset(omesh);

  DRAY_LOG_CLOSE();
  return dataset;
}

struct ReflectFunctor
{
  DataSet m_res;
  Vec<Float,3> m_point;
  Vec<Float,3> m_normal;
  ReflectFunctor(const Vec<float32,3> &point,
                 const Vec<float32,3> &normal)
  {
    for(int i = 0; i < 3; ++i)
    {
      m_point[i] = static_cast<Float>(point[i]);
      m_normal[i] = static_cast<Float>(normal[i]);
    }
    // ensure that this normalized
    m_normal.normalize();
  }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    m_res = detail::reflect_execute(mesh, m_point, m_normal);
  }
};

}//namespace detail

Reflect::Reflect()
  : m_point({0.f,0.f,0.f}),
    m_normal({0.f, 1.f, 0.f})
{
}

void
Reflect::plane(const Vec<float32,3> &point, const Vec<float32,3> &normal)
{
  m_point = point;
  m_normal = normal;
}

Collection
Reflect::execute(Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    detail::ReflectFunctor func(m_point, m_normal);
    dispatch(data_set.mesh(), func);

    // pass through all in the input fields
    const int num_fields = data_set.number_of_fields();
    for(int i = 0; i < num_fields; ++i)
    {
      func.m_res.add_field(data_set.field_shared(i));
    }
    res.add_domain(func.m_res);
  }
  return res;
}


}//namespace dray
