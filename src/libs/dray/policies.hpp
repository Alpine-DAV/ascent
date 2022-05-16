// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POLICIECS_HPP
#define DRAY_POLICIECS_HPP

#include <dray/exports.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{

#ifdef DRAY_CUDA_ENABLED
#define BLOCK_SIZE 128
using for_policy = RAJA::cuda_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce;
using atomic_policy = RAJA::cuda_atomic;
#elif defined(DRAY_HIP_ENABLED)
#define BLOCK_SIZE 256
using for_policy = RAJA::hip_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::hip_reduce;
using atomic_policy = RAJA::hip_atomic;
#elif defined(DRAY_OPENMP_ENABLED)
using for_policy = RAJA::omp_parallel_for_exec;
using reduce_policy = RAJA::omp_reduce;
using atomic_policy = RAJA::omp_atomic;
#else
using for_policy = RAJA::seq_exec;
using reduce_policy = RAJA::seq_reduce;
using atomic_policy = RAJA::seq_atomic;
#endif

//
// CPU only policies need when using classes
// that cannot be called on a GPU, e.g. MFEM
//
#ifdef DRAY_OPENMP_ENABLED
using for_cpu_policy = RAJA::omp_parallel_for_exec;
using reduce_cpu_policy = RAJA::omp_reduce;
using atomic_cpu_policy = RAJA::omp_atomic;
#else
using for_cpu_policy = RAJA::seq_exec;
using reduce_cpu_policy = RAJA::seq_reduce;
using atomic_cpu_policy = RAJA::seq_atomic;
#endif


} // namespace dray
#endif
