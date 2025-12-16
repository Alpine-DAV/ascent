#include <iostream>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/filters/WarpXStreamline.hpp>
#include <vtkh/filters/PointTransform.hpp>
#include <viskores/filter/flow/WarpXStreamline.h>
#include <viskores/cont/EnvironmentTracker.h>
#include <viskores/filter/geometry_refinement/Tube.h>

#if VTKH_PARALLEL
#include <viskores/thirdparty/diy/diy.h>
#include <viskores/thirdparty/diy/mpi-cast.h>
#include <mpi.h>
#endif


#include <viskores/rendering/Actor.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperWireframer.h>
#include <viskores/rendering/MapperRayTracer.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>



namespace vtkh
{

namespace detail
{

void GenerateChargedParticles(const viskores::cont::ArrayHandle<viskores::Vec3f>& pos,
                              const viskores::cont::ArrayHandle<viskores::Vec3f>& mom,
                              const viskores::cont::ArrayHandle<viskores::Float64>& mass,
                              const viskores::cont::ArrayHandle<viskores::Float64>& charge,
                              const viskores::cont::ArrayHandle<viskores::Float64>& weight,
                              viskores::cont::ArrayHandle<viskores::ChargedParticle>& seeds,
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

  for (viskores::Id i = 0; i < numValues; i++)
  {
    viskores::ChargedParticle electron(
      pPortal.Get(i), i, mPortal.Get(i), qPortal.Get(i), wPortal.Get(i), uPortal.Get(i));
    sPortal.Set(i + id_offset, electron);
  }
  
}


} //end detail

WarpXStreamline::WarpXStreamline()
: m_e_field_name("E"), 
	m_b_field_name("B"), 
	m_charge_field_name("Charge"), 
	m_mass_field_name("Mass"),
	m_momentum_field_name("Momentum"), 
	m_weighting_field_name("Weighting"),
	m_tubes(false),
	m_radius_set(false),
	m_tube_sides(3.0),
	m_tube_capping(true),
	m_tube_value(0.0),
	m_output_field_name("E_B_streamlines")
{
	
}

WarpXStreamline::~WarpXStreamline()
{

}

void WarpXStreamline::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_b_field_name);
  Filter::CheckForRequiredField(m_e_field_name);
  Filter::CheckForRequiredField(m_charge_field_name);
  Filter::CheckForRequiredField(m_mass_field_name);
  Filter::CheckForRequiredField(m_momentum_field_name);
  Filter::CheckForRequiredField(m_weighting_field_name);

}

void WarpXStreamline::PostExecute()
{
  Filter::PostExecute();
}

void WarpXStreamline::DoExecute()
{
    this->m_output = new DataSet();

#ifndef VTKH_BYPASS_VISKORES_BIH

#ifdef VTKH_PARALLEL
    // Setup VTK-h and Viskores comm.
    MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
    viskores::cont::EnvironmentTracker::SetCommunicator(viskoresdiy::mpi::communicator(viskoresdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
#endif

    //Make sure that the E field exists on any domain.
    if (!this->m_input->GlobalFieldExists(m_e_field_name))
    {
        throw Error("Domain does not contain specified E vector field for WarpXStreamline analysis.");
    }
    //Make sure that the B field exists on any domain.
    if (!this->m_input->GlobalFieldExists(m_b_field_name))
    {
        throw Error("Domain does not contain specified B vector field for WarpXStreamline analysis.");
    }

    viskores::cont::PartitionedDataSet inputs;

    viskores::cont::ArrayHandle<viskores::ChargedParticle> seeds;
    //Create charged particles for all domains with the particle spec fields.
    //TODO: user specified momentum,mass,charge,weighting?
    if (this->m_input->FieldExists(m_momentum_field_name))
    {
        const int num_domains = this->m_input->GetNumberOfDomains();
        int id_offset = 0;
        for (int i = 0; i < num_domains; i++)
        {
            viskores::Id domain_id;
            viskores::cont::DataSet dom;
            this->m_input->GetDomain(i, dom, domain_id);
            if(dom.HasField(m_momentum_field_name))
            {
                viskores::cont::ArrayHandle<viskores::Vec3f> pos, mom;
                viskores::cont::ArrayHandle<viskores::Float64> mass, charge, w;
                dom.GetCoordinateSystem().GetData().AsArrayHandle(pos);
                dom.GetField(m_momentum_field_name).GetData().AsArrayHandle(mom);
                dom.GetField(m_mass_field_name).GetData().AsArrayHandle(mass);
                dom.GetField(m_charge_field_name).GetData().AsArrayHandle(charge);
                dom.GetField(m_weighting_field_name).GetData().AsArrayHandle(w);
                detail::GenerateChargedParticles(pos,mom,mass,charge,w,seeds, id_offset);
                //Actual: local unique ids
                //Question: do we global unique ids? 
                id_offset += pos.GetNumberOfValues();
            }
            if(dom.HasField(m_b_field_name))
            {
                //add our vector fields as input
                inputs.AppendPartition(dom);
            }
        }
    }
    else
    {
        throw Error("Domain is missing one or more neccessary fields to create a charged particle: \
         Charge, Mass, Momentum, and/or Weighting.");
    }

    bool validField = (inputs.GetNumberOfPartitions() > 0);
//Don't really need this check
//since we got rid of the other check
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
        throw Error("Vector field type does not match a supportable type.");
    }

    //Everything is valid. Call the Viskores filter.

    viskores::filter::flow::WarpXStreamline warpxStreamlineFilter;
    //std::cerr << "intputs OUTTTT" << std::endl;
    //inputs.PrintSummary(std::cerr);

    warpxStreamlineFilter.SetStepSize(m_step_size);
    warpxStreamlineFilter.SetBField(m_b_field_name);
    warpxStreamlineFilter.SetEField(m_e_field_name);
    warpxStreamlineFilter.SetSeeds(seeds);
    warpxStreamlineFilter.SetNumberOfSteps(m_num_steps);
    warpxStreamlineFilter.SetOutputFieldName(m_output_field_name);
    auto out = warpxStreamlineFilter.Execute(inputs);

    //std::cerr << "m_output_field_name: " << m_output_field_name << std::endl;
    //std::cerr << "streamline output:" << std::endl;
    //out.PrintSummary(std::cerr);

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
            //std::cerr << "tube size too small" << std::endl;
            int num_domains = out.GetNumberOfPartitions();
            for (viskores::Id i = 0; i < num_domains; i++)
            {
              this->m_output->AddDomain(out.GetPartition(i), i);
            }
            return;
        }

        viskores::filter::geometry_refinement::Tube tubeFilter;
        tubeFilter.SetCapping(m_tube_capping);
        //tubeFilter.SetNumberOfSides(m_tube_sides);
        tubeFilter.SetNumberOfSides(4);
        tubeFilter.SetRadius(m_tube_size);
        //std::cerr << "tube size: " << radius << std::endl;
        auto tubeOut = tubeFilter.Execute(out);
      
        int num_domains = tubeOut.GetNumberOfPartitions();
        for (viskores::Id i = 0; i < num_domains; i++)
        {
          this->m_output->AddDomain(tubeOut.GetPartition(i), i);
        }
        this->m_output->AddConstantPointField(m_tube_value, m_output_field_name);
	
	//TODO: Add a linear option? 
	//Makes the streamlines pretty
        //this->m_output->AddLinearPointField(m_tube_value, m_output_field_name);
    }
    else
    {
      int num_domains = out.GetNumberOfPartitions();
      for (viskores::Id i = 0; i < num_domains; i++)
      {
        this->m_output->AddDomain(out.GetPartition(i), i);
      }
    }
#endif
}

} //  namespace vtkh
