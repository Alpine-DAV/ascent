#include <iostream>
#include <vtkh/filters/Streamline.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <viskores/filter/flow/Streamline.h>
#include <viskores/cont/EnvironmentTracker.h>
#include <viskores/filter/geometry_refinement/Tube.h>

#if VTKH_PARALLEL
#include <viskores/thirdparty/diy/diy.h>
#include <viskores/thirdparty/diy/mpi-cast.h>
#include <mpi.h>
#endif

namespace vtkh
{

Streamline::Streamline()
:  m_tubes(true),
   m_radius_set(false),
   m_tube_sides(3.0),
   m_tube_capping(true),
   m_tube_value(0.0)
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
        auto field = dom.GetField(m_field_name).GetData();
        if(field.GetNumberOfComponentsFlat() == 3)
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

  viskores::filter::flow::Streamline streamlineFilter;
  auto seedsAH = viskores::cont::make_ArrayHandle(m_seeds, viskores::CopyFlag::Off);

  streamlineFilter.SetStepSize(m_step_size);
  streamlineFilter.SetActiveField(m_field_name);
  streamlineFilter.SetSeeds(seedsAH);
  streamlineFilter.SetNumberOfSteps(m_num_steps);
  auto out = streamlineFilter.Execute(inputs);

  //call tube filter if we want to render output
  if(m_tubes)
  {

    if(!m_radius_set)
    {
      viskores::Float32 radius = 0.0;
      viskores::Bounds coordBounds = out.GetPartition(0).GetCoordinateSystem().GetBounds();
      // set a default radius
      viskores::Float64 lx = coordBounds.X.Length();
      viskores::Float64 ly = coordBounds.Y.Length();
      viskores::Float64 lz = coordBounds.Z.Length();
      viskores::Float64 mag = viskores::Sqrt(lx * lx + ly * ly + lz * lz);
      // same as used in vtk ospray
      constexpr viskores::Float64 heuristic = 1000.;
      radius = static_cast<viskores::Float32>(mag / heuristic);
      m_tube_size = radius;
    }

    //if the tubes are too small they cannot be rendered
    float min_tube_size = 0.00000001;
    if(m_tube_size < min_tube_size)
    {
      int num_domains = out.GetNumberOfPartitions();
      for (viskores::Id i = 0; i < num_domains; i++)
      {
        this->m_output->AddDomain(out.GetPartition(i), i);
      }
      return;
    }

    viskores::filter::geometry_refinement::Tube tubeFilter;
    tubeFilter.SetCapping(m_tube_capping);
    tubeFilter.SetNumberOfSides(m_tube_sides);
    tubeFilter.SetRadius(m_tube_size);

    auto tubeOut = tubeFilter.Execute(out);

    for (viskores::Id i = 0; i < tubeOut.GetNumberOfPartitions(); i++)
    {
      this->m_output->AddDomain(tubeOut.GetPartition(i), i);
    }
    this->m_output->AddConstantPointField(m_tube_value, m_output_field_name);
  }
  else
  {
    for (viskores::Id i = 0; i < out.GetNumberOfPartitions(); i++)
    {
      this->m_output->AddDomain(out.GetPartition(i), i);
    }
  }
#endif
}

} //  namespace vtkh
