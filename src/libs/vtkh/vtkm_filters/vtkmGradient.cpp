#include "vtkmGradient.hpp"
#include <vtkm/filter/vector_analysis/Gradient.h>

namespace vtkh
{


vtkm::cont::DataSet
vtkmGradient::Run(vtkm::cont::DataSet &input,
                  std::string field_name,
                  GradientParameters params,
                  vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::vector_analysis::Gradient grad;
  grad.SetOutputFieldName(params.output_name);
  grad.SetFieldsToPass(map_fields);
  grad.SetActiveField(field_name);

  grad.SetComputePointGradient(params.use_point_gradient);

  grad.SetComputeDivergence(params.compute_divergence);
  grad.SetDivergenceName(params.divergence_name);

  grad.SetComputeVorticity(params.compute_vorticity);
  grad.SetVorticityName(params.vorticity_name);

  grad.SetComputeQCriterion(params.compute_qcriterion);
  grad.SetQCriterionName(params.qcriterion_name);

  auto output = grad.Execute(input);
  return output;
}

} // namespace vtkh
