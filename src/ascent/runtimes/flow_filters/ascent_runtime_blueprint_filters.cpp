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

#ifdef ASCENT_VTKM_USE_CUDA
#include "cublas_v2.h"
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
__global__ void ColumnwiseKronecker( int R, int C, const double *A, double *B)
{
    int cols_per_block = C/gridDim.y;
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
void f_cokurt_vecs_cublas_wrapper(int nRows, int nCols, const double *A, double *kVecs)
{
    //              0 1 2 .. n-1
    // A = pressure
    //      temp
    printf("Entered C wrapper from Fortran side\n");

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

    //This is the Khatri Rao product of A with itsel
    //B = KhatriRao(A,A)
    bytes = nRows * nRows * nCols * sizeof(double);
    cudaMalloc(&d_B, bytes);

    //This is the maticised cokurtosis tensor, obtained by matrix multiply of B*B'
    //C = (1/nCols)(B * B'); size(C) = (nRows*nRows) x (nRows,nRows)
    bytes = nRows * nRows * nRows * nRows * sizeof(double);
    cudaMalloc(&d_C, bytes);

    //We desire SVD(reshaped(C, nRows, nRows^3)).
    //Instead we do (reshaped(C)) * (reshaped(C))'
    //Pass this back to Fortran so it can do DSYEV at its end
    bytes = nRows * nRows * sizeof(double);
    cudaMalloc(&d_kV, bytes);

    cublasStatus_t stat;

    //Set Matrices A and C on device
    stat = cublasSetMatrix(nRows, nCols, sizeof(double), A, nRows, d_A, nRows);

    //stat = cublasSetMatrix(nRows*nRows, nRows*nRows, sizeof(double), C, nRows*nRows, d_C, nRows*nRows);

    //Form the Matrix Khatri Rao product i.e. B = KhatriRao(A,A)
    dim3 threadsPerBlock(64);
    dim3 numblocks(13, 32);
    ColumnwiseKronecker<<<numblocks, threadsPerBlock>>>(nRows, nCols, d_A, d_B);

    // Now compute the matricised cokurt tensor in C
    // C = B * B'
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    double alpha = 1.0/double(nCols); double beta = 0.0;
    stat = cublasDgemm(handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       nRows*nRows,
                       nRows*nRows,
                       nCols,
                       &alpha,
                       d_B,
                       nRows*nRows,
                       d_B,
                       nRows*nRows,
                       &beta,
                       d_C,
                       nRows*nRows);

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

    // Now get the matrix from device memory, to copy into C, to pass back to Fortran
    stat = cublasGetMatrix( nRows, nRows, sizeof(double), d_kV, nRows, kVecs, nRows);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_kV);
}
#endif
//-----------------------------------------------------------------------------
void compute_fmms(const int & nfields, 
                  double * kVecs, 
                  double * fmms)
{
   //Stuff to call LAPACK routine dsyev_
   //dsyev----Routine to compute eigen decomposition of real symmetric matrix
   char jobz, uplo;
   int lwork, info;
   double best_work_val;
   double *work;
   
   jobz = 'V'; //Means we want to compute both eigenvalues and eigenvectors
   uplo = 'U'; //Whether input matrix is upper or lower triangular stored

   double* eigenvalues = malloc(sizeof(double)*nfields);

   //First call dsyev to do a workspace query
   lwork = -1;
   dsyev_(&jobz, &uplo, &nfields, kvecs, &nfields, eigenvalues, &best_work_val, &lwork, &info);

   lwork = best_work_val + 0.1;

   double* work = malloc(sizeof(double)*lwork);

   //Now the actual call that does the eigen decomposition
   dsyev_(&jobz, &uplo, &nfields, kvecs, &nfields, eigenvalues, work, &lwork, &info);

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


   free(eigenvalues);
   free(work);

}
//-----------------------------------------------------------------------------
void
Learn::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("blueprint_learn input must be a DataObject");
    }

    std::string protocol = params()["protocol"].as_string();

    Node v_info;
    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_low_order_bp();
#ifdef ASCENT_VTKM_USE_CUDA
    for(int i = 0; i < n_input->number_of_children(); ++i)
    {
      const conduit::Node &dom = n_input->child(0);
      const double * e = dom["fields/energy/values"].value();
      const double * p = dom["fields/pressure/values"].value();
      const int size = dom["fields/pressure/values"].dtype().number_of_elements();
      double *A = new double[size*2];
      for(int a = 0; a < size; ++a)
      {
        int offset = a * 2;
        A[offset] = e[a];
        A[offset+1] = p[a];
      }
      double *kVecs = new double[4];
      f_cokurt_vecs_cublas_wrapper(2, size, A, kVecs);
      delete[] A;
      std::cout<<"kVecs "<<kVecs[0]<<" "<<kVecs[1]<<" "<<kVecs[2]<<" "<<kVecs[3]<<"\n";

      //Code to compute 'feature moment metrics (fmms)' from kVecs
      double *fmms = new double[2];
      compute_fmms(2, kVecs, fmms);
    }
#endif

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





