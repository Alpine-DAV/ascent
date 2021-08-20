//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_blueprint_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_blueprint_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <runtimes/flow_filters/ascent_runtime_relay_filters.hpp>
#include <Genten_HigherMoments.hpp>
#include <expressions/ascent_blueprint_architect.hpp>
#include <ascent_mpi_utils.hpp>
#include <math.h>
#include <limits>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <runtimes/ascent_data_object.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/DataSet.h>
#include <ascent_vtkh_data_adapter.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/DataSet.hpp>
#endif

using namespace conduit;
using namespace std;

using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
//
namespace filters
{
namespace detail
{
void write_metric(conduit::Node &input,
                  const std::string topo,
                  std::vector<std::vector<std::pair<std::string,double>>> &mesh_data)
{
  conduit::Node output;
  for(int i = 0; i < input.number_of_children(); ++i)
  {
    conduit::Node &dom = input.child(i);
    conduit::Node &out_dom = output.append();
    out_dom["state"] = dom["state"];
    out_dom["topologies"] = dom["topologies"];
    std::string coords_name = dom["topologies/"+topo+"/coordset"].as_string();
    conduit::Node &in_coords = dom["coordsets/"+coords_name];
    double in_i = in_coords["dims/i"].to_float64();
    double in_j = in_coords["dims/j"].to_float64();
    double in_k = in_coords["dims/k"].to_float64();

    conduit::Node &out_coords = out_dom["coordsets/"+coords_name];
    out_coords["type"] = "uniform";
    out_coords["dims/i"] = 2;
    out_coords["dims/j"] = 2;
    out_coords["dims/k"] = 2;
    out_coords["origin"] = in_coords["origin"];
    out_coords["spacing/dx"] = in_i * in_coords["spacing/dx"].to_float64();
    out_coords["spacing/dy"] = in_j * in_coords["spacing/dy"].to_float64();
    out_coords["spacing/dz"] = in_k * in_coords["spacing/dz"].to_float64();


    std::vector<std::pair<std::string,double>> &dom_scalars = mesh_data[i];
    for(int s = 0; s < dom_scalars.size(); ++s)
    {
      std::string field_name = dom_scalars[s].first;
      conduit::Node &field = out_dom["fields/"+field_name];
      field["association"] = "element";
      field["topology"] = topo;
      field["values"].set(conduit::DataType::float64(1));
      conduit::float64_array array = field["values"].value();
      array[0] = dom_scalars[s].second;;
    }
    if(mpi_rank() == 0 && i ==0)
    {
      out_dom.print();
    }
  }
  conduit::Node info;
  bool is_valid = blueprint::mesh::verify(output, info);
  if(!is_valid && mpi_rank() == 0)
  {
    info.print();
  }
  std::string result_path;
  mesh_blueprint_save(output,
                      "small_spatial_metric",
                      "hdf5",
                      100,
                      result_path);
}

} //  namespace detail

//-----------------------------------------------------------------------------
BlueprintVerify::BlueprintVerify()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
BlueprintVerify::~BlueprintVerify()
{
// empty
}

//-----------------------------------------------------------------------------
void
BlueprintVerify::declare_interface(Node &i)
{
    i["type_name"]   = "blueprint_verify";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
BlueprintVerify::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("protocol") ||
       ! params["protocol"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'protocol'";
    }

    return res;
}


//-----------------------------------------------------------------------------
void
BlueprintVerify::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_verify input must be a DataObject");
    }

    std::string protocol = params()["protocol"].as_string();

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    // some MPI tasks may not have data, that is fine
    // but blueprint verify will fail, so if the
    // input node is empty skip verify
    int local_verify_ok = 0;
    if(!n_input->dtype().is_empty())
    {
        if(!conduit::blueprint::verify(protocol,
                                       *n_input,
                                       v_info))
        {
            n_input->schema().print();
            v_info.print();
            ASCENT_ERROR("blueprint verify failed for protocol"
                          << protocol << std::endl
                          << "details:" << std::endl
                          << v_info.to_json());
        }
        else
        {
            local_verify_ok = 1;
        }
    }

    // make sure some MPI task actually had bp data
#ifdef ASCENT_MPI_ENABLED
    int global_verify_ok = 0;
    MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    MPI_Allreduce((void *)(&local_verify_ok),
                (void *)(&global_verify_ok),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);
    local_verify_ok = global_verify_ok;
#endif

    if(local_verify_ok == 0)
    {
        ASCENT_ERROR("blueprint verify failed: published data is empty");
    }

    set_output<DataObject>(d_input);
}

//-----------------------------------------------------------------------------
Learn::Learn()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Learn::~Learn()
{
// empty
}

//-----------------------------------------------------------------------------
void
Learn::declare_interface(Node &i)
{
    i["type_name"]   = "blueprint_learn";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
Learn::verify_params(const conduit::Node &params,
                     conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("protocol") ||
       ! params["protocol"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'protocol'";
    }

    return res;
}
//-----------------------------------------------------------------------------
void compute_fmms(const int & nfields,
                  const double * kvecs,
                  double * eigenvalues,
                  double * fmms) // output of size fields
{

   //Now that we have the eigenvalues/vectores, compute the fmms

   double sum_eigvals = 0.0;
   for(int i = 0; i<nfields; i++)
   {
     fmms[i] = 0.0; //important initialization
     //This could be better, probably, with std::transform
     eigenvalues[i] = std::sqrt(std::abs(eigenvalues[i])); //Need to include cmath.h
     sum_eigvals += eigenvalues[i];
   }

   for(int j = 0; j<nfields; j++)
   {
     for(int i = 0; i<nfields; i++)
     {
       fmms[i] += eigenvalues[j] * ( kvecs[i + j*nfields]
                                    *kvecs[i + j*nfields]) ;
     }
   }

   //Normalise
   for(int i = 0; i<nfields; i++)
   {
     fmms[i] = std::abs(fmms[i]) / sum_eigvals;
   }

   //If everything is correct sum of fmms should be close to 1.0
   //Consider an assert to check that sum is in neighbourhood of 1.0


}
void
Learn::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_learn input must be a DataObject");
    }
    if(mpi_rank() == 0)
    {
      std::cout<<"******************* LEARN *******************\n";
    }
    //std::string protocol = params()["protocol"].as_string();

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_low_order_bp();

    double threshold = 0.7;
    if(params().has_path("threshold"))
    {
      threshold = params()["threshold"].to_float64();
    }

    std::vector<std::string> field_selection;
    if(params().has_path("fields"))
    {
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("Learn: field list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("Learn: field list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
    }
    else
    {
      ASCENT_ERROR("Learn: missing field list");
    }
    const int num_fields = field_selection.size();

    std::string assoc =  "";
    std::string topo =  "";
    //int field_size = 0;
    std::vector<int> field_sizes;
    field_sizes.resize(n_input->number_of_children());


    for(int i = 0; i < n_input->number_of_children(); ++i)
    {
      const conduit::Node &dom = n_input->child(i);
      for(int f = 0; f < num_fields; ++f)
      {
        std::string fpath = "fields/"+field_selection[f];
        if(!dom.has_path(fpath))
        {
          ASCENT_ERROR("Learn: no field named '"<<field_selection[f]<<"'");
        }
        std::string f_assoc = dom[fpath + "/association"].as_string();
        std::string f_topo = dom[fpath + "/topology"].as_string();
        if(f == 0)
        {
          // todo: this is not totally right.
          // we need to check that all domains and all fields are cool
          assoc = f_assoc;
          topo = f_topo;
          field_sizes[i] = dom[fpath + "/values"].dtype().number_of_elements();
        }
        else
        {
          if(f_assoc != assoc || f_topo != topo)
          {
            ASCENT_ERROR("Learn: field topology mismatch");
          }
        }
      }
    }
    int rank = 0;
    int comm_size = 1;
#ifdef ASCENT_MPI_ENABLED
    int comm_id = flow::Workspace::default_mpi_comm();

    MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
    MPI_Comm_rank(mpi_comm,&rank);

    MPI_Comm_size(mpi_comm, &comm_size);
#endif

    std::vector<double> field_mins;
    std::vector<double> field_lengths;

    field_mins.resize(num_fields);
    field_lengths.resize(num_fields);
    for(int f = 0; f < num_fields; ++f)
    {
      conduit::Node n_min, n_max;
      n_min = expressions::field_min(*n_input,field_selection[f]);
      n_max = expressions::field_max(*n_input,field_selection[f]);
      double minv = n_min["value"].to_float64();
      double maxv = n_max["value"].to_float64();
      // we are going to normalize, so
      // protect against 1/length being 1/0
      double length = maxv == minv ? 1.0 : maxv - minv;
      field_mins[f] = minv;
      field_lengths[f] = length;
      if(rank == 0)
      {
        std::cout<<field_selection[f]
                 <<" min "<<field_mins[f]
                 <<" len "<<field_lengths[f]<<"\n";
      }
    }

    std::stringstream d_name;
    d_name<<"rank_"<<rank;
    std::ofstream debug;
    debug.open(d_name.str());
    //debug<<"Field size "<<field_size<<"\n";

    const int num_domains = n_input->number_of_children();
    int global_blocks = num_domains;
    //
    // data for visual debugging
    using Data = std::pair<std::string,double>;
    std::vector<std::vector<Data>> mesh_data;
    mesh_data.resize(num_domains);

#ifdef ASCENT_MPI_ENABLED
    MPI_Allreduce(&num_domains, &global_blocks, 1, MPI_INT, MPI_SUM, mpi_comm);
#endif

    double *kVecs = new double[num_fields*num_fields]; // TODO: need one per domain!!!!
    double *eigvals = new double[num_fields];
    double *norm_eigv = new double[num_domains];
    double *fmms = new double[num_fields * num_domains];
    //double *anomaly_metric = new double

    higher_moments_init();
    //Kokkos::initialize();
    for(int i = 0; i < num_domains; ++i)
    {
      std::cout<<"Number of values "<<field_sizes[i]<<"\n";
      const conduit::Node &dom = n_input->child(i);

      // visual debugging data
      std::vector<Data> &domain_data = mesh_data[i];

      double min_value = std::numeric_limits<double>::max();
      double max_value = std::numeric_limits<double>::lowest();

      std::vector<const double*> fields;
      for(int f = 0; f < num_fields; ++f)
      {
        std::string fpath = "fields/"+field_selection[f] + "/values";
        const double * field_ptr = dom[fpath].value();
        fields.push_back(field_ptr);
      }
      double *A = new double[field_sizes[i]*num_fields];
      for(int a = 0; a < field_sizes[i]; ++a)
      {
        int offset = a * num_fields;
        for(int f = 0; f < num_fields; ++f)
        {
          double val = fields[f][a];
          val = (val - field_mins[f]) / field_lengths[f];
          A[offset + f] = val;
          min_value = std::min(min_value, fields[f][a]);
          max_value = std::max(max_value, fields[f][a]);
        }
      }
      int order = 4; // default value in this function
      //double *fmms = FormRawMomentTensor(A, field_sizes[0], num_fields, order);

      double *pvecs = new double[num_fields * num_fields];
      double *pvals = new double[num_fields];
      ComputePrincipalKurtosisVectors(A, field_sizes[i], num_fields, pvecs, pvals);
      double * domain_fmms = fmms + num_fields * i;
      compute_fmms(num_fields, pvecs, pvals, domain_fmms);
      delete[] pvecs;
      delete[] pvals;
    } // ends domain loop

    double *average_fmms = new double[num_fields];
    double *local_sum = new double[num_fields];
    for(int i = 0; i < num_fields; ++i)
    {
      local_sum[i] = 0.;
    }

    // Field-by-field average over all the blocks
    for(int i = 0; i < num_domains; ++i)
    {

      if(norm_eigv[i] > 1e-18)
      {

        int offset = i * num_fields;
        for(int f = 0; f < num_fields; f++)
        {
          double val = fmms[offset + f];
          local_sum[f] += val;
        }
        debug<<"keep domain "<<i<<" fmms "<< fmms[offset] << " " << fmms[offset+1] << " ";
        debug<< fmms[offset+2] << " " << fmms[offset+3] << "\n";
      }
    }
    for(int f = 0; f < num_fields; f++)
    {
      debug<<"field "<<f<<" block sum "<<local_sum[f]<<"\n";
    }
#ifdef ASCENT_MPI_ENABLED
    //int *domains_per_rank = new int[comm_size];
    MPI_Allreduce(local_sum, average_fmms, num_fields, MPI_DOUBLE, MPI_SUM, mpi_comm);
#else
    for(int f = 0; f < num_fields; f++)
    {
      average_fmms[f] = local_sum[f]/double(num_domains);
    }
#endif //MPI

    for(int f = 0; f< num_fields; f++)
    {
      average_fmms[f] /= double(global_blocks);
    }

    if(rank  == 0)
    {
      for(int f = 0; f < num_fields; f++)
      {
        std::cout<<field_selection[f]<<" average fmms "<< average_fmms[f] <<"\n";
      }
    }

    //Compute the spatial anomaly metric for each domain. If metric is above threshold = 0.7
    //the domain is anomalous, so paint all its cells 'red'
    double *spatial_metric = new double[num_domains];

    bool triggered = false;
    double min_metric = std::numeric_limits<double>::max();
    double max_metric = std::numeric_limits<double>::lowest();

    for(int i = 0; i < num_domains; ++i)
    {
      spatial_metric[i] = 0.0;
      if(norm_eigv[i] > 1e-18)
      {

      double * domain_fmms = fmms + num_fields * i;

      //Compute Hellinger distance between domain fmms and average fmms
      for(int f = 0; f< num_fields; f++)
      {
        spatial_metric[i] += ( std::sqrt(domain_fmms[f]) - std::sqrt(average_fmms[f]) ) *
                             ( std::sqrt(domain_fmms[f]) - std::sqrt(average_fmms[f]) );
      }

      spatial_metric[i] = std::sqrt(spatial_metric[i] * 0.5);
      min_metric = std::min(min_metric, spatial_metric[i]);
      max_metric = std::max(max_metric, spatial_metric[i]);


      }


      //This threshold is user specified, and a "hyper parameter" (fudge factor)
      if(spatial_metric[i] > threshold)
      {
        //TO DO: Take whatever actions, e.g. painting all cells of this domain
        triggered = true;
      }
    }


    triggered = global_someone_agrees(triggered);

    double global_min, global_max;
    global_min = min_metric;
    global_max = max_metric;

#ifdef ASCENT_MPI_ENABLED
    MPI_Reduce(&min_metric, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0,mpi_comm);
    MPI_Reduce(&max_metric, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0,mpi_comm);
#endif
    if(rank == 0)
    {
      std::cout<<"Spatial metric min "<<global_min<<"\n";
      std::cout<<"Spatial metric max "<<global_max<<"\n";
      std::cout<<"Spatial threshold "<<threshold<<"\n";
    }

#ifdef ASCENT_MPI_ENABLED
    if (global_max == max_metric) std::cout<<"In rank "<<rank<<"\n";
#endif

    if(rank == 0 && triggered)
    {
      std::cout<<"FIRE\n";
    }

    debug<<"*** Rank ***"<<rank<<"\n";
    for(int i = 0; i < n_input->number_of_children(); ++i)
    {
      conduit::Node &dom = n_input->child(i);
      debug<<"Domain "<<i<<" "<<dom["coordsets"].to_yaml()<<"\n";
      debug<<"field size "<<field_sizes[i]<<"\n";
      debug<<"spatial metric "<< spatial_metric[i] <<"\n";
      conduit::Node &field = dom["fields/spatial_metric"];
      field["association"] = assoc;
      field["topology"] = topo;
      field["values"].set(conduit::DataType::float64(field_sizes[i]));
      conduit::float64_array array = field["values"].value();

      for(int v = 0; v < field_sizes[i]; ++v)
      {
        array[v] = spatial_metric[i];
      }
    }

    conduit::Node info;
    bool is_valid = blueprint::mesh::verify(*n_input, info);
    if(!is_valid && rank == 0)
    {
      info.print();
    }

    //  For scalability this is not necessary
    //  debugging only
    std::string result_path;
    mesh_blueprint_save(*n_input,
                        "spatial_metric",
                        "hdf5",
                        200,
                        result_path);

    // add in the spatial metric
    for(int i = 0; i < num_domains; ++i)
    {
      std::vector<Data> &domain_data = mesh_data[i];
      Data metric;
      metric.first = "spatial_metric";
      metric.second = spatial_metric[i];
      domain_data.push_back(metric);
    }

    detail::write_metric(*n_input, topo, mesh_data);

    delete[] fmms;
    debug.close();
    //higher_moments_finalize();
    //Kokkos::finalize();

    //set_output<DataObject>(d_input);
}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





