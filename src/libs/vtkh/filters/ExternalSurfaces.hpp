#ifndef VTK_H_EXTERNAL_SURFACES_HPP
#define VTK_H_EXTERNAL_SURFACES_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>

//---------------------------------------------------------------------------//
namespace vtkh
{
//---------------------------------------------------------------------------//
    
//---------------------------------------------------------------------------//
class VTKH_API ExternalSurfaces : public Filter
{
public:
    ExternalSurfaces();
    virtual ~ExternalSurfaces();
    std::string GetName() const override;

protected:
    void PreExecute() override;
    void PostExecute() override;
    void DoExecute() override;
};

//---------------------------------------------------------------------------//
} //namespace vtkh
//---------------------------------------------------------------------------//

#endif
