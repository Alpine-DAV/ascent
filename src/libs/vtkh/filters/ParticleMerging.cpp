#include <vtkm/filter/clean_grid/worklet/PointMerge.h>
//TODO: Header for wrapped filter
//#include <vtkm/filter/clean_grid/CleanGrid.h>
#include <vtkh/Error.hpp>
#include <vtkh/filters/ParticleMerging.hpp>

namespace vtkh
{

ParticleMerging::ParticleMerging()
  : m_radius(-1)
{

}

ParticleMerging::~ParticleMerging()
{

}

void
ParticleMerging::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
ParticleMerging::SetRadius(const vtkm::Float64 radius)
{
  if(radius <= 0)
  {
    throw Error("Particle merging: radius must be greater than zero");
  }

  m_radius = radius;
}

void ParticleMerging::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
  if(!this->m_input->IsPointMesh())
  {
    throw Error("Particle Merging: input must be a point mesh");
  }
  if(m_radius == -1.)
  {
    throw Error("Particle merging: radius never set");
  }
}

void ParticleMerging::PostExecute()
{
  Filter::PostExecute();
}

void ParticleMerging::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    // insert interesting stuff
    auto coords = dom.GetCoordinateSystem().GetData();
    std::string coords_name = dom.GetCoordinateSystem().GetName();
    vtkm::Bounds bounds = dom.GetCoordinateSystem().GetBounds();

    bool fast_merge = true;
    vtkm::worklet::PointMerge merger;
    merger.Run(m_radius * 2. , fast_merge, bounds, coords);
    vtkm::cont::CoordinateSystem mcoords = vtkm::cont::CoordinateSystem(coords_name, coords);

    // this field could be associated with cells or points,
    // but this is a point mesh so those are the samae
    vtkm::cont::Field field = dom.GetField(m_field_name);
    auto in_field = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
    vtkm::cont::Field mfield(field.GetName(),
                             field.GetAssociation(),
                              merger.MapPointField(in_field));

    const vtkm::Id num_cells = mcoords.GetNumberOfPoints();
    vtkm::cont::ArrayHandleCounting<vtkm::Id> cconn(0,1,num_cells);
    vtkm::cont::ArrayHandle<vtkm::Id> conn;
    vtkm::cont::ArrayCopy(cconn, conn);
    vtkm::cont::CellSetSingleType<> cellset;
    cellset.Fill(num_cells,vtkm::CELL_SHAPE_VERTEX,1,conn);

    vtkm::cont::DataSet out;
    out.AddCoordinateSystem(mcoords);
    out.AddField(mfield);
    out.SetCellSet(cellset);

    m_output->AddDomain(out, domain_id);

  }
}

std::string
ParticleMerging::GetName() const
{
  return "vtkh::ParticleMerging";
}

} //  namespace vtkh
