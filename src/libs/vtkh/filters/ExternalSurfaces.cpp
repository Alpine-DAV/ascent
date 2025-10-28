#include <vtkh/filters/ExternalSurfaces.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/utils/viskores_dataset_info.hpp>
#include <vtkh/viskores_filters/viskoresCleanGrid.hpp>
#include <viskores/filter/entity_extraction/ExternalFaces.h>

//---------------------------------------------------------------------------//
namespace vtkh
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
ExternalSurfaces::ExternalSurfaces()
{

}

//---------------------------------------------------------------------------//
ExternalSurfaces::~ExternalSurfaces()
{

}

//---------------------------------------------------------------------------//
void
ExternalSurfaces::PreExecute()
{
    Filter::PreExecute();
}

//---------------------------------------------------------------------------//
void
ExternalSurfaces::PostExecute()
{
    Filter::PostExecute();
}

//---------------------------------------------------------------------------//
void
ExternalSurfaces::DoExecute()
{
    this->m_output = new DataSet();

    const int num_domains = this->m_input->GetNumberOfDomains();

    for(int i = 0; i < num_domains; ++i)
    {
        viskores::Id domain_id;
        viskores::cont::DataSet dom;
        this->m_input->GetDomain(i, dom, domain_id);

        viskores::filter::entity_extraction::ExternalFaces ext_faces;
        ext_faces.SetCompactPoints(true);
        ext_faces.SetPassPolyData(true);
        viskores::cont::DataSet ds_output = ext_faces.Execute(dom);

        // TODO , do we need to clean the grid?
        vtkh::viskoresCleanGrid cleaner;
        auto clout = cleaner.Run(ds_output, this->GetFieldSelection());
        m_output->AddDomain(clout, domain_id);

  }
}

//---------------------------------------------------------------------------//
std::string
ExternalSurfaces::GetName() const
{
    return "vtkh::ExternalSurfaces";
}
//---------------------------------------------------------------------------//
} //  namespace vtkh
//---------------------------------------------------------------------------//

