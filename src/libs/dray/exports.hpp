// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_EXPORTS_HPP
#define DRAY_EXPORTS_HPP

#include <RAJA/RAJA.hpp>

#if defined(__CUDACC__) && !defined(DEBUG_CPU_ONLY)

#define DRAY_CUDA_ENABLED
#define DRAY_EXEC RAJA_INLINE RAJA_HOST_DEVICE
#define DRAY_EXEC_ONLY RAJA_INLINE RAJA_DEVICE
#define DRAY_LAMBDA RAJA_DEVICE

#elif defined(__HIP_PLATFORM_HCC__) && !defined(DEBUG_CPU_ONLY)

#define DRAY_HIP_ENABLED
#define DRAY_EXEC RAJA_INLINE RAJA_HOST_DEVICE
#define DRAY_EXEC_ONLY RAJA_INLINE RAJA_DEVICE
#define DRAY_LAMBDA RAJA_DEVICE

#else 
#define DRAY_EXEC inline
#define DRAY_EXEC_ONLY inline
#define DRAY_LAMBDA

#endif

#define DRAY_CPU_LAMBDA

#endif
