#ifndef VTK_H_GRADIENT_PARAMETERS_HPP
#define VTK_H_GRADIENT_PARAMETERS_HPP

#include <string>

namespace vtkh
{

struct GradientParameters
{
  bool use_point_gradient = true;
  bool compute_divergence = false;
  bool compute_vorticity  = false;
  bool compute_qcriterion = false;

  std::string output_name = "gradient";
  std::string divergence_name = "divergence";
  std::string vorticity_name = "vorticity";
  std::string qcriterion_name = "qcriterion";

};

}
#endif
