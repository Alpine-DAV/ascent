#include <iostream>
#include <vtkh/filters/Streamline.hpp>
#include <vtkm/filter/flow/Streamline.h>
#include <vtkm/cont/EnvironmentTracker.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>

#if VTKH_PARALLEL
#include <vtkm/thirdparty/diy/diy.h>
#include <vtkm/thirdparty/diy/mpi-cast.h>
#include <mpi.h>
#endif

namespace vtkh
{

Streamline::Streamline()
{
}

Streamline::~Streamline()
{

}

void Streamline::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void Streamline::PostExecute()
{
  Filter::PostExecute();
}

void Streamline::DoExecute()
{
  this->m_output = new DataSet();

#ifndef VTKH_BYPASS_VTKM_BIH

#ifdef VTKH_PARALLEL
  // Setup VTK-h and VTK-m comm.
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
#endif

  //Make sure that the field exists on any domain.
  if (!this->m_input->GlobalFieldExists(m_field_name))
  {
    throw Error("Domain does not contain specified vector field for ParticleAdvection analysis.");
  }

  vtkm::cont::PartitionedDataSet inputs;

  //Create a partitioned dataset for all domains with the field.
  if (this->m_input->FieldExists(m_field_name))
  {
    const int num_domains = this->m_input->GetNumberOfDomains();
    for (int i = 0; i < num_domains; i++)
    {
      vtkm::Id domain_id;
      vtkm::cont::DataSet dom;
      this->m_input->GetDomain(i, dom, domain_id);
      if(dom.HasField(m_field_name))
      {
        using vectorField_d = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>>;
        using vectorField_f = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3>>;
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
    throw Error("Vector field type does not match <vtkm::Vec<vtkm::Float32,3>> or <vtkm::Vec<vtkm::Float64,3>>");
  }

  //Everything is valid. Call the VTKm filter.

  vtkm::filter::flow::Streamline streamlineFilter;
  auto seedsAH = vtkm::cont::make_ArrayHandle(m_seeds, vtkm::CopyFlag::Off);

  streamlineFilter.SetStepSize(m_step_size);
  streamlineFilter.SetActiveField(m_field_name);
  streamlineFilter.SetSeeds(seedsAH);
  streamlineFilter.SetNumberOfSteps(m_num_steps);
  auto out = streamlineFilter.Execute(inputs);

  for (vtkm::Id i = 0; i < out.GetNumberOfPartitions(); i++)
  {
    this->m_output->AddDomain(out.GetPartition(i), i);
  }
#endif
}

} //  namespace vtkh
