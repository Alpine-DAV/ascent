#include <dray/utils/color_buffer_utils.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>

#include <assert.h>

namespace dray
{

// dest[i] = dest[i] + add[i]
void add(Array<Vec<float32,4>> &dest,
         Array<Vec<float32,4>> &add)
{
  assert(dest.size() == add.size());
  const int32 size = dest.size();


  Vec<float32,4> *dest_ptr = dest.get_device_ptr();
  const Vec<float32,4> *add_ptr = add.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    dest_ptr[i] += add_ptr[i];
  });
  DRAY_ERROR_CHECK();
}

// dest = dest[i] / scalar
void scalar_divide(Array<Vec<float32,4>> &dest,
                   const float32 divisor)
{
  const int32 size = dest.size();

  Vec<float32,4> *dest_ptr = dest.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    dest_ptr[i] /= divisor;
  });
  DRAY_ERROR_CHECK();
}

// dest = value
void init_constant(Array<Vec<float32,4>> &dest,
                   const float32 value)
{
  const int32 size = dest.size();

  Vec<float32,4> *dest_ptr = dest.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    dest_ptr[i] = value;
  });
  DRAY_ERROR_CHECK();
}

} // namespace dray
