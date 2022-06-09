#include "cell_average.hpp"

#include <dray/data_model/data_set.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/dispatcher.hpp>

#include <iostream>

namespace
{

// I'm not sure the best way todo this... using a macro for now
#define OUT_ELEM_TYPE_DEF dray::Element<FieldElemType::get_dim(), \
                                        FieldElemType::get_ncomp(), \
                                        FieldElemType::get_etype(), \
                                        dray::Order::Constant>

template<typename FieldElemType>
static dray::UnstructuredField<OUT_ELEM_TYPE_DEF>
compute_cell_average(const dray::UnstructuredField<FieldElemType> &in_field,
                     const std::string &name)
{
  using OutElemType = OUT_ELEM_TYPE_DEF;

  // Host implementation for now
  const auto nvalues = in_field.get_num_elem();
  dray::Array<dray::Vec<dray::Float, OutElemType::get_ncomp()>> out_data;
  out_data.resize(nvalues);
  auto *out_data_ptr = out_data.get_host_ptr();
  const RAJA::RangeSegment range(0, nvalues);
  RAJA::forall<RAJA::seq_exec>(range,
    [=](int i)
    {
      out_data_ptr[i][0] = 1;
    });

  // Build return value
  dray::GridFunction<OutElemType::get_ncomp()> out_gf;
  out_gf.m_el_dofs = 1;
  out_gf.m_size_el = nvalues;
  // Q: Do we need conn for this?
  out_gf.m_size_ctrl = nvalues;
  out_gf.m_values = std::move(out_data);
  dray::UnstructuredField<OutElemType> out_field(out_gf, dray::Order::Constant, name);
  return out_field;
}

#undef OUT_ELEM_TYPE_DEF

#if 0
template<typename MeshElemType, typename FieldElemType>
static dray::UnstructuredField<FieldElemType>
compute_cell_averages(dray::UnstructuredMesh<MeshElemType> mesh,
                      dray::UnstructuredField<FieldElemType> field)
{
  dray::DeviceField<FieldElemType> in_field(field);
  dray::DeviceMesh<MeshElemType> in_mesh(mesh);

  dray::Array<

  // Here's what I want to write
  auto elements = mesh.get_dof_data().m_values;
  auto values = field.get_dof_data().m_values;
  const int Nelem = mesh.cells();
  const int Nper_elem = MeshType::ElementType::get_ncomp();
  double *output = new double[Nelem];
  for(int i = 0; i < Nelem; i++)
  {
    const auto element = elements[i];
    for(int j = 0; j < Nper_elem; j++)
    {
      output[i] += values[element[j]][0];
    }
    output[i] = output[i] / double(Nper_elem);
  }

  std::cout << "Result:\n";
  for(int i = 0; i < Nelem; i++)
  {
    std::cout << "  " << output[i] << "\n";
  }
  std::cout << std::endl;
  delete[] output;
}
#endif

struct CellAverageFunctor
{
  CellAverageFunctor() = delete;
  CellAverageFunctor(dray::DataSet input,
                     const std::string &in_field,
                     const std::string &out_field);
  ~CellAverageFunctor() = default;

  dray::DataSet execute();

  // This method gets invoked by dispatch, which will have converted the Mesh
  // into a concrete derived type like UnstructuredMesh<Hex_P1> so this method
  // is able to call methods on the derived type with no virtual calls.
  template<typename MeshType, typename ScalarField>
  void operator()(MeshType &mesh, ScalarField &field);

  dray::DataSet m_input;
  dray::DataSet m_output;
  std::string m_ifield;
  std::string m_ofield;
};

CellAverageFunctor::CellAverageFunctor(dray::DataSet input,
                                       const std::string &in_field,
                                       const std::string &out_field)
  : m_input(input), m_output(input), m_ifield(in_field), m_ofield(out_field)
{
}

dray::DataSet
CellAverageFunctor::execute()
{
  // This iterates over the product of possible mesh and scalar field types
  // to call the operator() function that can operate on concrete types.
  dray::Field *field = m_input.field(m_ifield);
  if(field != nullptr && field->components() == 1)
  {
    dispatch(m_input.mesh(), field, *this);
  }
  return m_output;
}

template<typename MeshType, typename FieldType>
inline void
CellAverageFunctor::operator()(MeshType &mesh, FieldType &field)
{
  std::cout << "Dispatched!" << std::endl;
  conduit::Node n_mesh;
  mesh.to_node(n_mesh);
  n_mesh.print();

  conduit::Node n_field;
  field.to_node(n_field);
  n_field.print();

  auto out_field = compute_cell_average(field, m_ofield);
  conduit::Node n_out_field;
  out_field.to_node(n_out_field);
  std::cout << "Out Field:" << std::endl;
  n_out_field.print();
}

}

// Public interface
namespace dray
{

CellAverage::CellAverage()
  : in_field(), out_field()
{
  // Do nothing
}

CellAverage::~CellAverage()
{
  // Do nothing
}

void
CellAverage::set_field(const std::string &name)
{
  in_field = name;
}

void
CellAverage::set_output_field(const std::string &name)
{
  out_field = name;
}

Collection
CellAverage::execute(Collection &input)
{
  if(!input.has_field(in_field))
  {
    DRAY_ERROR("Cannot execute CellAverage for variable '" << in_field
            << "' because it does not exist in the given collection.");
  }

  if(out_field.empty())
  {
    out_field = in_field;
  }

  Collection output;
  const int N = input.local_size();
  for(int i = 0; i < N; i++)
  {
    CellAverageFunctor impl(input.domain(i), in_field, out_field);
    output.add_domain(impl.execute());
  }
  return output;
}

}
