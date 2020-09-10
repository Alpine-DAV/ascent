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

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_config.h>
#include <ascent_mpi_utils.hpp>
#include <runtimes/ascent_data_object.hpp>
#include <runtimes/flow_filters/ascent_runtime_relay_filters.hpp>
#include <expressions/ascent_blueprint_architect.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

#include <limits>

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

#ifdef ASCENT_VTKM_USE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
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
namespace filters
{

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

#ifdef ASCENT_VTKM_USE_CUDA
inline void cuda_error_check(const char *file, const int line )
{
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    std::cerr<<"CUDA error reported at: "<<file<<":"<<line;
    std::cerr<<" : "<<cudaGetErrorString(err)<<"\n";
    //exit( -1 );
  }
}

#define CHECK_ERROR() cuda_error_check(__FILE__,__LINE__);

__global__ void ColumnwiseKronecker( int R,
                                     int C, // field size
                                     const double *A,
                                     double *B)
{
    int cols_per_block = C / gridDim.y;
    int col_start = cols_per_block*blockIdx.y;
    int col_end = col_start + cols_per_block - 1;
    //int vec_col_idx = 0;
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int left_vec_row_idx = row/R;
    int right_vec_row_idx = row % R;
    for(int col = col_start; col <= col_end; col++) {
       int col_offset = col * R * R;
       if (row < R*R)
           B[col_offset + row] = A[col*R + left_vec_row_idx] * A[col*R + right_vec_row_idx];
    }
    //if(blockIdx.x == 0 && threadIdx.x==0){
    //   printf("from block %d cols %d range %d--%d\n", blockIdx.y, cols_per_block, col_start, col_end);
    //}
}

void f_cokurt_vecs_cublas_wrapper(int nRows,
                                  int nCols,
                                  const double *A,
                                  double *kVecs,
                                  double *kVals)
{
    // nRows = num_fields,
    // nCols = field_size
    //              0 1 2 .. n-1
    // A = pressure
    //      temp
    //

    //Sanity checking
    //for(int i=0; i<nRows*nRows*nRows*nRows; i++) {
    //  C[i] = double(i);
    //}

    size_t bytes;

    //Allocate memory for matrices A, B and C on device
    double *d_A, *d_B, *d_C, *d_kV;

    //This is A, the raw data matrix, passed in from Fortran
    bytes = nRows * nCols * sizeof(double);
    cudaMalloc(&d_A, bytes);
    CHECK_ERROR();

    //This is the Khatri Rao product of A with itsel
    //B = KhatriRao(A,A)
    bytes = nRows * nRows * nCols * sizeof(double);
    cudaMalloc(&d_B, bytes);
    CHECK_ERROR();

    //This is the maticised cokurtosis tensor, obtained by matrix multiply of B*B'
    //C = (1/nCols)(B * B'); size(C) = (nRows*nRows) x (nRows,nRows)
    bytes = nRows * nRows * nRows * nRows * sizeof(double);
    cudaMalloc(&d_C, bytes);
    CHECK_ERROR();

    //We desire SVD(reshaped(C, nRows, nRows^3)).
    //Instead we do kV = (reshaped(C)) * (reshaped(C))'
    //Then we do kV, kVals = eigen_decomp(kV),
    //Pass kV and kVals back to Fortran as final results
    bytes = nRows * nRows * sizeof(double);
    cudaMalloc(&d_kV, bytes);
    CHECK_ERROR();


    cublasStatus_t stat;

    //Set Matrices A on device
    stat = cublasSetMatrix(nRows, //num fields
                           nCols, // field size
                           sizeof(double),
                           A,
                           nRows,
                           d_A,
                           nRows);
    CHECK_ERROR();


    //Form the Matrix Khatri Rao product i.e. B = KhatriRao(A,A)
    dim3 threadsPerBlock(64);
    dim3 numblocks(13, 32);
    ColumnwiseKronecker<<<numblocks, threadsPerBlock>>>(nRows,
                                                        nCols,
                                                        d_A,
                                                        d_B);
    CHECK_ERROR();

    // Now compute the matricised cokurt tensor in C
    // C = B * B'
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    CHECK_ERROR();

    double alpha = 1.0/double(nCols); double beta = 0.0;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       nRows*nRows, nRows*nRows, nCols,
                       &alpha,
                       d_B, nRows*nRows,
                       d_B, nRows*nRows,
                       &beta,
                       d_C, nRows*nRows);
    CHECK_ERROR();


    // Now that the matricised cokurt tensor is done, multiply C*C'
    // C is implicitly reshaped to (nRows)x(nRows^3) in the call to DGEMM itself
    double alpha2 = 1.0;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       nRows, nRows, nRows*nRows*nRows,
                       &alpha2,
                       d_C, nRows,
                       d_C, nRows,
                       &beta,
                       d_kV, nRows);
    CHECK_ERROR();



    // Now we perform the Eigen decomposition of kV
    // We use the dense-symmetric-eigenvalue-solver 'cusolverDnDsyevd'
    // The solver OVERWRITES THE INPUT MATRIX WITH THE EIGENVECTORS

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;

    // Allocate memory on device for the vector of eigenvalues, W
    double *d_W;
    bytes = nRows * sizeof(double);
    cudaMalloc(&d_W, bytes);
    CHECK_ERROR();

    // Eigenvalue step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    CHECK_ERROR();
    if(CUSOLVER_STATUS_SUCCESS != cusolver_status)
    {
      std::cout<<"Solve failed\n";
    }

    // Eigenvalue step 2: query working space of syevd
    int lwork = 0;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status =
      cusolverDnDsyevd_bufferSize(cusolverH,
                                  jobz,
                                  uplo,
                                  nRows,  // M of input matrix
                                  d_kV,   // input matrix
                                  nRows,  // leading dimension of input matrix
                                  d_W,    // vector of eigenvalues
                                  &lwork);// on return size of working array
    CHECK_ERROR();
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    if(CUSOLVER_STATUS_SUCCESS != cusolver_status)
    {
      std::cout<<"Solve failed\n";
    }

    double *d_work = NULL;
    int *devInfo = NULL;

    cudaMalloc((void**)&d_work , sizeof(double)*lwork);
    CHECK_ERROR();
    cudaMalloc ((void**)&devInfo, sizeof(int));
    CHECK_ERROR();

    //Eigenvalue step 3: compute actualy eigen decomposition

    cusolver_status = cusolverDnDsyevd(cusolverH,
                                       jobz,
                                       uplo,
                                       nRows,  // M of input matrix
                                       d_kV,   // input matrix
                                       nRows,  // leading dimension of input matrix
                                       d_W,    // vector of eigenvalues
                                       d_work,
                                       lwork,
                                       devInfo);
    CHECK_ERROR();

    cudaStat = cudaDeviceSynchronize();
    if(CUSOLVER_STATUS_SUCCESS != cusolver_status)
    {
      std::cout<<"Solve failed\n";
    }
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat);
    if(CUSOLVER_STATUS_SUCCESS != cusolver_status)
    {
      std::cout<<"Solve failed\n";
    }

    // Now copy the eigen vector vector back to host
    cudaStat = cudaMemcpy(kVals,
                          d_W,
                          sizeof(double)*nRows,
                          cudaMemcpyDeviceToHost);

    // Now get the eigenvectors matrix from device memory, to copy into kVecs, to pass back to Fortran
    stat = cublasGetMatrix( nRows,
                            nRows,
                            sizeof(double),
                            d_kV,
                            nRows,
                            kVecs,
                            nRows);

    cublasDestroy(handle);
    cusolverDnDestroy(cusolverH);

    CHECK_ERROR();

    cudaFree(d_A);
    CHECK_ERROR();
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_kV);

    CHECK_ERROR();
    cudaFree(d_work);
    cudaFree(d_W);
    cudaFree(devInfo);
    CHECK_ERROR();

}
#endif
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
//-----------------------------------------------------------------------------
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


#ifdef ASCENT_VTKM_USE_CUDA
    const int num_domains = n_input->number_of_children();
    double *kVecs = new double[num_fields*num_fields]; // TODO: need one per domain!!!!
    double *eigvals = new double[num_fields];
    double *fmms = new double[num_fields * num_domains];
    //double *anomaly_metric = new double

    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = n_input->child(i);

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
      // in column major order
      f_cokurt_vecs_cublas_wrapper(num_fields, // nrow
                                   field_sizes[i], // nCol
                                   A,
                                   kVecs,
                                   eigvals);
      delete[] A;
      //std::cout<<"kVecs "<<kVecs[0]<<" "<<kVecs[1]<<" "<<kVecs[2]<<" "<<kVecs[3]<<"\n";
      //std::cout<<"eigvals "<<eigvals[0]<<" "<<eigvals[1]<<"\n";
      debug<<"domain "<<i<<" kVecs "<<kVecs[0]<<" "<<kVecs[1]<<" "<<kVecs[2]<<" "<<kVecs[3]<<"\n";
      debug<<"domain "<<i<<" eigvals "<<eigvals[0]<<" "<<eigvals[1]<<"\n";

      debug<<"domain "<<i<<" min value "<<min_value<<"\n";
      debug<<"domain "<<i<<" max value "<<max_value<<"\n";
      double diff = max_value - min_value;
      if(diff > 10) debug<<"domain "<<i<<" diff "<<diff<<"\n";
      //Code to compute 'feature moment metrics (fmms)' from kVecs
      // offset for current domain
      double * domain_fmms = fmms + num_fields * i;
      compute_fmms(num_fields, kVecs, eigvals, domain_fmms);
      for(int f = 0; f < num_fields; ++f)
      {
        if(domain_fmms[f]  != domain_fmms[f]) domain_fmms[f] = 0;
        debug<<"domain "<<i<<" fmms "<<f<<" "<<domain_fmms[f]<<"\n";
      }
    }


    double *average_fmms = new double[num_fields];
    double *local_sum = new double[num_fields];
    for(int i = 0; i < num_fields; ++i)
    {
      local_sum[i] = 0.;
    }

    for(int i = 0; i < num_domains; ++i)
    {
      int offset = i * num_fields;
      for(int f = 0; f < num_fields; f++)
      {
        double val = fmms[offset + f];
        local_sum[f] = val;
      }
    }
    for(int f = 0; f < num_fields; f++)
    {
      debug<<"local sum field "<<f<<" sum "<<local_sum[f]<<"\n";
    }
#ifdef ASCENT_MPI_ENABLED
    //int *domains_per_rank = new int[comm_size];
    MPI_Allreduce(local_sum, average_fmms, num_fields, MPI_DOUBLE, MPI_SUM, mpi_comm);
#endif //MPI

    for(int f = 0; f< num_fields; f++)
    {
      average_fmms[f] /= double(comm_size * num_domains);
    }

    if(rank  == 0)
    {
      for(int f = 0; f < num_fields; f++)
      {
        std::cout<<field_selection[f]<<" ave "<<average_fmms[f] <<"\n";
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

    if(rank == 0 && triggered)
    {
      std::cout<<"FIRE\n";
    }

    for(int i = 0; i < n_input->number_of_children(); ++i)
    {
      conduit::Node &dom = n_input->child(i);
      debug<<"Domain "<<i<<" "<<dom["coordsets"].to_yaml()<<"\n";
      debug<<"field size "<<field_sizes[i]<<"\n";
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

    mesh_blueprint_save(*n_input,
                        "spatial_metric",
                        "hdf5",
                        -1);

    delete[] kVecs;
    delete[] eigvals;
    delete[] fmms;
    delete[] average_fmms;
    delete[] local_sum;
    delete[] spatial_metric;
#endif // cuda
  debug.close();

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





