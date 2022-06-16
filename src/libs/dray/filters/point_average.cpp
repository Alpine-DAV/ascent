#include "point_average.hpp"

#include <dray/data_model/data_set.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/dispatcher.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <iostream>

// Start internal implementation
namespace
{

using namespace dray;

// Could template off of Mesh GridFunction dimension but it always seems to be 3.
template<typename FieldElemType>
static std::shared_ptr<Field>
compute_point_average(const GridFunction<3> &in_mesh_gf,
                      const UnstructuredField<FieldElemType> &in_field,
                      const std::string &name)
{
  using OutElemType = Element<FieldElemType::get_dim(),
                              FieldElemType::get_ncomp(),
                              FieldElemType::get_etype(),
                              Order::Linear>;
  using VecType = Vec<Float, OutElemType::get_ncomp()>;

  // Retrieve important input information
  constexpr auto ncomp = FieldElemType::get_ncomp();
  const GridFunction<ncomp> &in_field_gf = in_field.get_dof_data();
  const auto *in_data_ptr = in_field_gf.m_values.get_device_ptr_const();
  const auto *in_idx_ptr = in_field_gf.m_ctrl_idx.get_device_ptr_const();
  const auto *in_conn_ptr = in_mesh_gf.m_ctrl_idx.get_device_ptr_const();

  // Start to build output information
  const int ndof = in_mesh_gf.m_el_dofs;
  const int nelem = in_mesh_gf.m_size_el;
  const int nvalues = in_mesh_gf.m_values.size();
  GridFunction<OutElemType::get_ncomp()> out_gf;
  out_gf.m_el_dofs = ndof;
  out_gf.m_size_el = nelem;
  out_gf.m_size_ctrl = in_mesh_gf.m_size_ctrl;
  out_gf.m_ctrl_idx = in_mesh_gf.m_ctrl_idx;
  out_gf.m_values.resize(nvalues);
  auto *out_data_ptr = out_gf.m_values.get_device_ptr();

  Array<int> ncells_per_point;
  ncells_per_point.resize(nvalues);
  int *ncells_data = ncells_per_point.get_device_ptr();

  // Is there a good way to zero initialize?
  const RAJA::RangeSegment range_points(0, nvalues);
  RAJA::forall<for_policy>(range_points,
    [=] DRAY_LAMBDA (int i)
    {
      for(int c = 0; c < ncomp; c++)
      {
        out_data_ptr[i][c] = 0.;
      }
      ncells_data[i] = 0;
    });

  // For each cell, add the cell value to the output point array
  const RAJA::RangeSegment range_cells(0, nelem);
  RAJA::forall<for_policy>(range_cells,
    [=] DRAY_LAMBDA (int i)
    {
      const int conn_idx = i * ndof;
      const int in_data_idx = in_idx_ptr[i];
      for(int dof = 0; dof < ndof; dof++)
      {
        const int dof_idx = in_conn_ptr[conn_idx + dof];
        for(int c = 0; c < ncomp; c++)
        {
          RAJA::atomicAdd<atomic_policy>(&out_data_ptr[dof_idx][c],
                                          in_data_ptr[in_data_idx][c]);
        }
        RAJA::atomicAdd<atomic_policy>(&ncells_data[dof_idx], 1);
      }
    });

  // For each point, divide by the number of cells that touch that point
  RAJA::forall<for_policy>(range_points,
    [=] DRAY_LAMBDA (int i)
    {
      for(int c = 0; c < ncomp; c++)
      {
        out_data_ptr[i][c] = out_data_ptr[i][c] / ncells_data[i];
      }
    });

  return std::make_shared<UnstructuredField<OutElemType>>(out_gf, Order::Linear, name);
}

/**
  @brief Need to ensure the output field created by this filter is on
         the output dataset. This means we cannot copy any field from
         the input that has out_name as its name.
*/
DataSet
initialize_output_domain(DataSet &domain, const std::string &out_name)
{
  DataSet retval(domain);

  retval.clear_fields();
  const int nfields = domain.number_of_fields();
  for(int i = 0; i < nfields; i++)
  {
    const auto ptr = domain.field_shared(i);
    if(ptr->name() == out_name)
    {
      // Skip the field with the same name as out_field
      continue;
    }
    retval.add_field(ptr);
  }
  return retval;
}

struct PointAverageFunctor
{
  PointAverageFunctor() = delete;
  PointAverageFunctor(Mesh *mesh, Field *field, const std::string &name);
  ~PointAverageFunctor() = default;

  void execute();
  inline std::shared_ptr<Field> output() { return m_output; }

  template<typename FieldType>
  void operator()(FieldType &mesh);

  std::shared_ptr<Field> m_output;
  std::string m_name;
  Mesh *m_mesh;
  Field *m_field;
  GridFunction<3> *m_mesh_gf;
};

PointAverageFunctor::PointAverageFunctor(Mesh *mesh,
                                         Field *field,
                                         const std::string &name)
  : m_output(), m_name(name), m_mesh(mesh), m_field(field), m_mesh_gf(nullptr)
{
  // Do nothing
}

void
PointAverageFunctor::execute()
{
  GridFunction<3> gf = detail::get_dof_data(m_mesh);
  m_mesh_gf = &gf;
  dispatch(m_field, *this);
  m_mesh_gf = nullptr;
}

template<typename FieldType>
void
PointAverageFunctor::operator()(FieldType &field)
{
  m_output = compute_point_average(*m_mesh_gf, field, m_name);
}

// End internal implementation
}

// Public interface
namespace dray
{

PointAverage::PointAverage()
  : in_field(), out_field(), ref_mesh()
{
  // Do nothing
}

PointAverage::~PointAverage()
{
  // Do nothing
}

void
PointAverage::set_field(const std::string &name)
{
  in_field = name;
}

void
PointAverage::set_output_field(const std::string &name)
{
  out_field = name;
}

void
PointAverage::set_mesh(const std::string &name)
{
  ref_mesh = name;
}

Collection
PointAverage::execute(Collection &input)
{
  if(!input.has_field(in_field))
  {
    DRAY_ERROR("Cannot execute PointAverage for variable '" << in_field
            << "' because it does not exist in the given collection.");
  }

  // Default output name to input name.
  // If the user set a custom output name then use that.
  std::string out_name(in_field);
  if(!out_field.empty())
  {
    out_name = out_field;
  }

  Collection output;
  auto domains = input.domains();
  for(DataSet &domain : domains)
  {
    DataSet out_dom = initialize_output_domain(domain, out_name);
    if(domain.has_field(in_field))
    {
      // Get the mesh, default to mesh 0.
      Mesh  *mesh = mesh = domain.mesh();
      if(!ref_mesh.empty())
      {
        if(!domain.has_mesh(ref_mesh))
        {
          // Issue warning? A mesh was picked by the user but it
          // doesn't exist on this domain.
          continue;
        }
        mesh = domain.mesh(ref_mesh);
      }
      if(!mesh)
      {
        DRAY_ERROR("PointAverage cannot be executed on a domain with no mesh. Domain " << domain.domain_id() << ".")
      }

      Field *field = domain.field(in_field);
      PointAverageFunctor pointavg(mesh, field, out_name);
      pointavg.execute();
      out_dom.add_field(pointavg.output());
    }
    output.add_domain(out_dom);
  }
  return output;
}

}
