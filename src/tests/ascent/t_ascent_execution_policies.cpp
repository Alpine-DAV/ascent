//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_execution_policies.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <expressions/ascent_execution_policies.hpp>
#include <expressions/ascent_memory_manager.hpp>

#include <cmath>
#include <iostream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;



void *device_alloc(index_t bytes)
{
#if defined(ASCENT_DEVICE_ENABLED)
    return DeviceMemory::allocate(bytes);
#else
    return HostMemory::allocate(bytes);
#endif
}


// void copy(void *dest_ptr, void *src_ptr, index_t bytes)
// {
//     MagicMemory::copy(dest_ptr,src_ptr,bytes);
// }


void device_free(void *ptr)
{
#if defined(ASCENT_DEVICE_ENABLED)
    return DeviceMemory::deallocate(ptr);
#else
    return HostMemory::deallocate(ptr);
#endif
}


void test_forall()
{
    const index_t size = 10;

    index_t host_vals[size];
    index_t *dev_vals_ptr = static_cast<index_t*>(device_alloc(sizeof(index_t) * size));

    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        dev_vals_ptr[i] = i;
    });
    ASCENT_DEVICE_ERROR_CHECK();
    
    MagicMemory::copy(&host_vals[0], dev_vals_ptr, sizeof(index_t) * size);

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(host_vals[i],i);
    }

    device_free(dev_vals_ptr);
}


void test_reductions()
{
  
    const index_t size = 4;
    index_t host_vals[size] = {0,-10,10, 5};
    index_t *dev_vals_ptr = static_cast<index_t*>(device_alloc(sizeof(index_t) * size));
    MagicMemory::copy(dev_vals_ptr, &host_vals[0], sizeof(index_t) * size);


    // sum
    ascent::ReduceSum<reduce_policy,index_t> sum_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        sum_reducer += dev_vals_ptr[i];
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(sum_reducer.get(),5);


   // min
    ascent::ReduceMin<reduce_policy,index_t> min_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        min_reducer.min(dev_vals_ptr[i]);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(min_reducer.get(),-10);

   // minloc
    ascent::ReduceMinLoc<reduce_policy,index_t> minloc_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        minloc_reducer.minloc(dev_vals_ptr[i],i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(minloc_reducer.get(),-10);
    EXPECT_EQ(minloc_reducer.getLoc(),1);


   // max
    ascent::ReduceMax<reduce_policy,index_t> max_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        max_reducer.max(dev_vals_ptr[i]);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(max_reducer.get(),10);

   // maxloc
    ascent::ReduceMaxLoc<reduce_policy,index_t> maxloc_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        maxloc_reducer.maxloc(dev_vals_ptr[i],i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(maxloc_reducer.get(),10);
    EXPECT_EQ(maxloc_reducer.getLoc(),2);

    device_free(dev_vals_ptr);
}



void test_atomics()
{
    const index_t size = 4;
    index_t host_vals[size] = {0,-1,-2,-3};
    index_t *dev_vals_ptr = static_cast<index_t*>(device_alloc(sizeof(index_t) * size));

    MagicMemory::copy(dev_vals_ptr, &host_vals[0], sizeof(index_t) * size);

    // add
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        ascent::atomic_add<atomic_policy>( dev_vals_ptr + i, i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    MagicMemory::copy(&host_vals[0], dev_vals_ptr, sizeof(index_t) * size);

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(host_vals[i],0);
    }

    // min
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        ascent::atomic_min<atomic_policy>( dev_vals_ptr + i, static_cast<index_t>(-10));
    });
    ASCENT_DEVICE_ERROR_CHECK();

    MagicMemory::copy(&host_vals[0], dev_vals_ptr, sizeof(index_t) * size);

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(host_vals[i],-10);
    }

    // max
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        ascent::atomic_max<atomic_policy>( dev_vals_ptr + i, static_cast<index_t>(10));
    });
    ASCENT_DEVICE_ERROR_CHECK();

    MagicMemory::copy(&host_vals[0], dev_vals_ptr, sizeof(index_t) * size);

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(host_vals[i],10);
    }  
    
    device_free(dev_vals_ptr);
}

TEST(ascent_execution_policies, forall)
{
    test_forall();
}

TEST(ascent_execution_policies, reductions)
{
    test_reductions();
}

TEST(ascent_execution_policies, atomics)
{
    test_atomics();
}

