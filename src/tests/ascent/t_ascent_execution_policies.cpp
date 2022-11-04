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

#include <cmath>
#include <iostream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;


TEST(ascent_execution_policies, forall)
{
    const index_t size = 10;
    index_t vals[size];
    index_t *vals_ptr =  &vals[0];
    

    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        vals_ptr[i] = i;
    });
    ASCENT_DEVICE_ERROR_CHECK();

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(vals[i],i);
    }
}


TEST(ascent_execution_policies, reductions)
{
  
    const index_t size = 4;
    index_t vals[size] = {0,-10,10, 5};
    index_t *vals_ptr =  &vals[0];
    
    // sum
    ascent::ReduceSum<reduce_policy,index_t> sum_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        sum_reducer += vals_ptr[i];
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(sum_reducer.get(),5);


   // min
    ascent::ReduceMin<reduce_policy,index_t> min_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        min_reducer.min(vals_ptr[i]);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(min_reducer.get(),-10);

   // minloc
    ascent::ReduceMinLoc<reduce_policy,index_t> minloc_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        minloc_reducer.minloc(vals_ptr[i],i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(minloc_reducer.get(),-10);
    EXPECT_EQ(minloc_reducer.getLoc(),1);



   // max
    ascent::ReduceMax<reduce_policy,index_t> max_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        max_reducer.max(vals_ptr[i]);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(max_reducer.get(),10);

   // maxloc
    ascent::ReduceMaxLoc<reduce_policy,index_t> maxloc_reducer;
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        maxloc_reducer.maxloc(vals_ptr[i],i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    EXPECT_EQ(maxloc_reducer.get(),10);
    EXPECT_EQ(maxloc_reducer.getLoc(),2);

}


TEST(ascent_execution_policies, atomics)
{
    const index_t size = 4;
    index_t vals[size] = {0,-1,-2,-3};
    index_t *vals_ptr =  &vals[0];

    // add
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        ascent::atomic_add<atomic_policy>( vals_ptr + i, i);
    });
    ASCENT_DEVICE_ERROR_CHECK();

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(vals[i],0);
    }

    // min
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        ascent::atomic_min<atomic_policy>( vals_ptr + i, static_cast<index_t>(-10));
    });
    ASCENT_DEVICE_ERROR_CHECK();

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(vals[i],-10);
    }

    // max
    ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
    {
        ascent::atomic_max<atomic_policy>( vals_ptr + i, static_cast<index_t>(10));
    });
    ASCENT_DEVICE_ERROR_CHECK();

    for(index_t i=0;i<size;i++)
    {
      EXPECT_EQ(vals[i],10);
    }  

}
