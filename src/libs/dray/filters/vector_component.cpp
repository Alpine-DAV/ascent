#include <dray/filters/vector_component.hpp>

#include <dray/dispatcher.hpp>
#include <dray/data_model/elem_utils.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#include <memory>


namespace dray
{

namespace detail
{

template<typename ElemType>
std::shared_ptr<Field>
vector_comp_execute(UnstructuredField<ElemType> &field,
                    const int32 component)
{
  DRAY_LOG_OPEN("vector_component");

  GridFunction<ElemType::get_ncomp()> input_gf = field.get_dof_data();
  GridFunction<1> output_gf;
  // the output will have the same params as the input, just a different
  // values type
  output_gf.m_ctrl_idx = input_gf.m_ctrl_idx;
  output_gf.m_el_dofs = input_gf.m_el_dofs;
  output_gf.m_size_el = input_gf.m_size_el;
  output_gf.m_size_ctrl = input_gf.m_size_ctrl;
  output_gf.m_values.resize(input_gf.m_values.size());

  Vec<Float,ElemType::get_ncomp()> *in_ptr = input_gf.m_values.get_device_ptr();
  Vec<Float,1> *out_ptr = output_gf.m_values.get_device_ptr();
  const int size = input_gf.m_values.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    out_ptr[i][0] = in_ptr[i][component];
  });

  using OutElemT = Element<ElemType::get_dim(),
                           1,
                           ElemType::get_etype(),
                           ElemType::get_P()>;

  UnstructuredField<OutElemT> foutput(output_gf, field.order(), "");
  std::shared_ptr<Field> output = std::make_shared<UnstructuredField<OutElemT>>(foutput);

  DRAY_LOG_CLOSE();
  return output;
}

struct VectorComponentFunctor
{
  int32 m_component;
  std::shared_ptr<Field> m_output;
  VectorComponentFunctor(const int32 component)
    : m_component(component)

  {

  }

  template<typename FieldType>
  void operator()(FieldType &field)
  {
    m_output = detail::vector_comp_execute(field, m_component);
  }
};
}//namespace detail

VectorComponent::VectorComponent()
  : m_component(-1)
{
}

void
VectorComponent::component(const int32 comp)
{
  if(comp < 0 || comp > 2)
  {
    DRAY_ERROR("Vector component must be in range [0,2] given '"<<comp<<"'");
  }
  m_component = comp;
}

void
VectorComponent::output_name(const std::string &name)
{
  m_output_name = name;
}

void
VectorComponent::field(const std::string &name)
{
  m_field_name = name;
}

Collection VectorComponent::decompose_all(Collection &input)
{
  Collection res;
  for(int32 i = 0; i < input.local_size(); ++i)
  {
    DataSet data_set = input.domain(i);
    DataSet decomped = decompose_all(data_set);
    res.add_domain(decomped);
  }
  return res;
}

Collection VectorComponent::decompose_field(Collection &input, const std::string &field_name)
{
  if(!input.has_field(field_name))
  {
    DRAY_ERROR("Cannot decompose non-existant field '"<<field_name<<"'");
  }

  Collection res;
  for(int32 i = 0; i < input.local_size(); ++i)
  {
    DataSet data_set = input.domain(i);
    if(data_set.has_field(field_name))
    {
      DataSet decomped = decompose_field(data_set, field_name);
      res.add_domain(decomped);
    }
    else
    {
      // just pass it through
      res.add_domain(data_set);
    }
  }
  return res;
}

DataSet VectorComponent::decompose_field(DataSet &input, const std::string &field_name)
{
  std::vector<std::string> suffix({"_x", "_y", "_z"});
  DataSet res = input;

  std::shared_ptr<Field> field = input.field_shared(field_name);
  int32 comps = field->components();

  if(comps == 3 || comps == 2)
  {
    for(int32 comp = 0; comp < comps; ++comp)
    {
      std::shared_ptr<Field> component = execute(field.get(), comp);
      component->name(field_name+suffix[comp]);
      res.add_field(component);
    }
  }
  else
  {
    DRAY_ERROR("Cannot decompose field that doesn not have 3 or 2 components");
  }
  return res;
}

DataSet VectorComponent::decompose_all(DataSet &input)
{
  std::vector<std::string> suffix({"_x", "_y", "_z"});
  DataSet res = input;
  res.clear_fields();
  for(int32 i = 0; i < input.number_of_fields(); ++i)
  {
    std::shared_ptr<Field> field = input.field_shared(i);
    int32 comps = field->components();
    std::string fname = field->name();
    if(comps > 1)
    {
      for(int32 comp = 0; comp < comps; ++comp)
      {
        std::shared_ptr<Field> component = execute(field.get(), comp);
        component->name(fname+suffix[comp]);
        res.add_field(component);
      }
    }
    else
    {
      res.add_field(field);
    }
  }
  return res;
}

std::shared_ptr<Field>
VectorComponent::execute(Field *field, const int32 comp)
{
  detail::VectorComponentFunctor func(comp);
  dispatch_vector(field, func);
  return func.m_output;
}

Collection
VectorComponent::execute(Collection &collection)
{
  if(m_component == -1)
  {
    DRAY_ERROR("Component never set");
  }

  if(m_field_name == "")
  {
    DRAY_ERROR("Must specify an field name");
  }

  if(!collection.has_field(m_field_name))
  {
    DRAY_ERROR("No field named '"<<m_field_name<<"'");
  }

  if(m_output_name == "")
  {
    DRAY_ERROR("Must specify an output  field name");
  }

  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    auto output = execute(data_set.field(m_field_name), m_component);
    output->name(m_output_name);
    data_set.add_field(output);

    // pass through
    res.add_domain(data_set);
  }
  return res;
}

}//namespace dray
