#include <iostream>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <viskores/filter/flow/ParticleAdvection.h>
#include <viskores/cont/EnvironmentTracker.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>

#if VTKH_PARALLEL
#include <viskores/thirdparty/diy/diy.h>
#include <viskores/thirdparty/diy/mpi-cast.h>
#include <mpi.h>
#endif

namespace vtkh
{

ParticleAdvection::ParticleAdvection()
{
}

ParticleAdvection::~ParticleAdvection()
{

}

void ParticleAdvection::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void ParticleAdvection::PostExecute()
{
  Filter::PostExecute();
}

void ParticleAdvection::DoExecute()
{
  this->m_output = new DataSet();

#ifndef VTKH_BYPASS_VISKORES_BIH

#ifdef VTKH_PARALLEL
  // Setup VTK-h and Viskores comm.
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  viskores::cont::EnvironmentTracker::SetCommunicator(viskoresdiy::mpi::communicator(viskoresdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
#endif

  //Make sure that the field exists on any domain.
  if (!this->m_input->GlobalFieldExists(m_field_name))
  {
    throw Error("Domain does not contain specified vector field for ParticleAdvection analysis.");
  }

  viskores::cont::PartitionedDataSet inputs;

  //Create a partitioned dataset for all domains with the field.
  if (this->m_input->FieldExists(m_field_name))
  {
    const int num_domains = this->m_input->GetNumberOfDomains();
    for (int i = 0; i < num_domains; i++)
    {
      viskores::Id domain_id;
      viskores::cont::DataSet dom;
      this->m_input->GetDomain(i, dom, domain_id);
      if(dom.HasField(m_field_name))
      {
        using vectorField_d = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64, 3>>;
        using vectorField_f = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32, 3>>;
        auto field = dom.GetField(m_field_name).GetData();
        if(field.IsType<vectorField_d>() && !field.IsType<vectorField_f>())
        {
          inputs.AppendPartition(dom);
        }
      }
    }
  }

  bool validField = (inputs.GetNumberOfPartitions() > 0);

#ifdef VTKH_PARALLEL
  int localNum = static_cast<int>(inputs.GetNumberOfPartitions());
  int globalNum = 0;
  MPI_Allreduce((void *)(&localNum),
                (void *)(&globalNum),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);
  validField = (globalNum > 0);
#endif

  if (!validField)
  {
    throw Error("Vector field type does not match <viskores::Vec<viskores::Float32,3>> or <viskores::Vec<viskores::Float64,3>>");
  }

  //Everything is valid. Call the Viskores filter.

  viskores::filter::flow::ParticleAdvection particleAdvectionFilter;
  auto seedsAH = viskores::cont::make_ArrayHandle(m_seeds, viskores::CopyFlag::Off);

  particleAdvectionFilter.SetStepSize(m_step_size);
  particleAdvectionFilter.SetActiveField(m_field_name);
  particleAdvectionFilter.SetSeeds(seedsAH);
  particleAdvectionFilter.SetNumberOfSteps(m_num_steps);
  auto out = particleAdvectionFilter.Execute(inputs);

  for (viskores::Id i = 0; i < out.GetNumberOfPartitions(); i++)
  {
    this->m_output->AddDomain(out.GetPartition(i), i);
  }
#endif
}

} //  namespace vtkh
