#include <vtkh/filters/Filter.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <ascent_annotations.hpp>

namespace vtkh
{

namespace detail
{
  void addField(std::vector<std::string>& vec, const std::string& str) {
    if (std::find(vec.begin(), vec.end(), str) == vec.end()) {
        vec.push_back(str);
    }
  }

}//end detail

Filter::Filter()
{
  m_input = nullptr;
  m_output = nullptr;
}

Filter::~Filter()
{
};

void
Filter::SetInput(DataSet *input)
{
  m_input = input;
}

DataSet*
Filter::GetOutput()
{
  return m_output;
}

DataSet*
Filter::Update()
{
  // Declare values before the filter scope so they remain active when
  // the scope's end snapshot is recorded. Output metrics are populated after
  // execution, once m_output is available.
  ascent::annotations::Value filter_name("ascent.vtkh.filter");
  ascent::annotations::Value backend("ascent.vtkh.backend");
  ascent::annotations::Value input_cells("ascent.vtkh.input.cells");
  ascent::annotations::Value input_domains("ascent.vtkh.input.domains");
  ascent::annotations::Value output_cells("ascent.vtkh.output.cells");
  ascent::annotations::Value output_domains("ascent.vtkh.output.domains");

  if(ascent::annotations::supported())
  {
    filter_name.set(this->GetName());
    backend.set(GetCurrentDevice());

    if(this->m_input != nullptr)
    {
      input_cells.set(static_cast<std::uint64_t>(this->m_input->GetNumberOfCells()));
      input_domains.set(static_cast<std::uint64_t>(this->m_input->GetNumberOfDomains()));
    }
  }

  ascent::annotations::Scope filter_scope("vtkh filter: " + this->GetName());
  VTKH_DATA_OPEN(this->GetName());
#ifdef VTKH_ENABLE_LOGGING
  VTKH_DATA_ADD("device", GetCurrentDevice());
  long long int in_cells = this->m_input->GetNumberOfCells();
  VTKH_DATA_ADD("input_cells", in_cells);
  VTKH_DATA_ADD("input_domains", this->m_input->GetNumberOfDomains());
  int in_topo_dims;
  bool in_structured = this->m_input->IsStructured(in_topo_dims);
  if(in_structured)
  {
    VTKH_DATA_ADD("in_topology", "structured");
  }
  else
  {
    VTKH_DATA_ADD("in_topology", "unstructured");
  }
#endif
  {
    ascent::annotations::Scope phase("vtkh pre execute");
    PreExecute();
  }
  {
    ascent::annotations::Scope phase("vtkh execute");
    DoExecute();
  }
  {
    ascent::annotations::Scope phase("vtkh post execute");
    PostExecute();
  }

  if(ascent::annotations::supported() && this->m_output != nullptr)
  {
    output_cells.set(static_cast<std::uint64_t>(this->m_output->GetNumberOfCells()));
    output_domains.set(static_cast<std::uint64_t>(this->m_output->GetNumberOfDomains()));
  }
#ifdef VTKH_ENABLE_LOGGING
  long long int out_cells = this->m_output->GetNumberOfCells();
  VTKH_DATA_ADD("output_cells", out_cells);
  VTKH_DATA_ADD("output_domains", this->m_output->GetNumberOfDomains());
  int out_topo_dims;
  bool out_structured = this->m_output->IsStructured(out_topo_dims);

  if(out_structured)
  {
    VTKH_DATA_ADD("output_topology", "structured");
  }
  else
  {
    VTKH_DATA_ADD("output_topology", "unstructured");
  }
#endif
  VTKH_DATA_CLOSE();
  return m_output;
}

void
Filter::AddMapField(const std::string &field_name)
{
  m_map_fields.push_back(field_name);
}

void
Filter::ClearMapFields()
{
  m_map_fields.clear();
}

void
Filter::PreExecute()
{
  if(m_input == nullptr)
  {
    std::stringstream msg;
    msg<<"Input for vtkh filter '"<<this->GetName()<<"' is null.";
    throw Error(msg.str());
  }

  if(m_map_fields.size() == 0)
  {
    this->MapAllFields();
  }

};

void
Filter::PostExecute()
{
  this->PropagateMetadata();
};

void
Filter::MapAllFields()
{
    int num_domains = m_input->GetNumberOfDomains();
    if(num_domains > 0)
    {
        for(int i = 0; i < num_domains; i++)
        {
            viskores::cont::DataSet dom = m_input->GetDomain(i);
            viskores::IdComponent num_fields = dom.GetNumberOfFields();
            for(viskores::IdComponent i = 0; i < num_fields; ++i)
            {
                std::string field_name = dom.GetField(i).GetName();
                detail::addField(m_map_fields,field_name);
            }
        }
    }
}

void
Filter::CheckForRequiredField(const std::string &field_name)
{
  if(m_input == nullptr)
  {
    std::stringstream msg;
    msg<<"Cannot verify required field '"<<field_name;
    msg<<"' for vkth filter '"<<this->GetName()<<"' because input is null.";
    throw Error(msg.str());
  }

  if(!m_input->GlobalFieldExists(field_name))
  {
    std::stringstream msg;
    msg<<"Required field '"<<field_name;
    msg<<"' for vkth filter '"<<this->GetName()<<"' does not exist";
    throw Error(msg.str());
  }
}

void
Filter::PropagateMetadata()
{
  m_output->SetCycle(m_input->GetCycle());
}


viskores::filter::FieldSelection
Filter::GetFieldSelection() const
{
  viskores::filter::FieldSelection sel;
  for (const auto& str : this->m_map_fields)
  {
    sel.AddField(str);
  }
  return sel;
}


} //namespace vtkh
