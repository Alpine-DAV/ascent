#include <iostream>
#include <vtkh/filters/WarpXStreamline.hpp>
#include <vtkm/filter/flow/WarpXStreamline.h>
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

namespace detail
{

void GenerateChargedParticles(const vtkm::cont::ArrayHandle<vtkm::Vec3f>& pos,
                              const vtkm::cont::ArrayHandle<vtkm::Vec3f>& mom,
                              const vtkm::cont::ArrayHandle<vtkm::Float64>& mass,
                              const vtkm::cont::ArrayHandle<vtkm::Float64>& charge,
                              const vtkm::cont::ArrayHandle<vtkm::Float64>& weight,
                              vtkm::cont::ArrayHandle<vtkm::ChargedParticle>& seeds,
			      const int id_offset)
{
  auto pPortal = pos.ReadPortal();
  auto uPortal = mom.ReadPortal();
  auto mPortal = mass.ReadPortal();
  auto qPortal = charge.ReadPortal();
  auto wPortal = weight.ReadPortal();

  auto numValues = pos.GetNumberOfValues();

  seeds.Allocate(numValues);
  auto sPortal = seeds.WritePortal();

  for (vtkm::Id i = 0; i < numValues; i++)
  {
    vtkm::ChargedParticle electron(
      pPortal.Get(i), i, mPortal.Get(i), qPortal.Get(i), wPortal.Get(i), uPortal.Get(i));
    sPortal.Set(i + id_offset, electron);
  }
}

} //end detail

WarpXStreamline::WarpXStreamline()
{
}

WarpXStreamline::~WarpXStreamline()
{

}

void WarpXStreamline::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void WarpXStreamline::PostExecute()
{
  Filter::PostExecute();
}

void WarpXStreamline::DoExecute()
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
    throw Error("Domain does not contain specified vector field for WarpXStreamline analysis.");
  }

  vtkm::cont::PartitionedDataSet inputs;

  vtkm::cont::ArrayHandle<vtkm::ChargedParticle> seeds;
  //Create charged particles for all domains with the vel and particle fields.
  if (this->m_input->FieldExists(m_field_name))
  {
    const int num_domains = this->m_input->GetNumberOfDomains();
    int id_offset = 0;
    for (int i = 0; i < num_domains; i++)
    {
      vtkm::Id domain_id;
      vtkm::cont::DataSet dom;
      this->m_input->GetDomain(i, dom, domain_id);
      if(dom.HasField("Momentum"))
      {
        vtkm::cont::ArrayHandle<vtkm::Vec3f> pos, mom;
        vtkm::cont::ArrayHandle<vtkm::Float64> mass, charge, w;
        dom.GetCoordinateSystem().GetData().AsArrayHandle(pos);
        dom.GetField("Momentum").GetData().AsArrayHandle(mom);
        dom.GetField("Mass").GetData().AsArrayHandle(mass);
        dom.GetField("Charge").GetData().AsArrayHandle(charge);
        dom.GetField("Weighting").GetData().AsArrayHandle(w);
        detail::GenerateChargedParticle(pos,mom,mass,charge,w,seeds, id_offset);
	//Actual: local unique ids
	//Question: do we global unique ids? 
	id_offset += pos.GetNumberOfValues();
      }
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

  vtkm::filter::flow::WarpXStreamline warpxStreamlineFilter;

  warpxStreamlineFilter.SetStepSize(m_step_size);
  warpxStreamlineFilter.SetActiveField(m_field_name);
  warpxStreamlineFilter.SetSeeds(seeds);
  warpxStreamlineFilter.SetNumberOfSteps(m_num_steps);
  auto out = warpxStreamlineFilter.Execute(inputs);

  for (vtkm::Id i = 0; i < out.GetNumberOfPartitions(); i++)
  {
    this->m_output->AddDomain(out.GetPartition(i), i);
  }
#endif
}

} //  namespace vtkh
