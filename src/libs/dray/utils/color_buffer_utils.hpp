#ifndef DRAY_COLOR_BUFFER_UTILS_HPP
#define DRAY_COLOR_BUFFER_UTILS_HPP

#include <dray/types.hpp>

#include <dray/array.hpp>
#include <dray/vec.hpp>

#include <string>

namespace dray
{

// dest[i] = dest[i] + add[i]
void add(Array<Vec<float32,4>> &dest,
         Array<Vec<float32,4>> &add);

// dest = dest[i] / scalar
void scalar_divide(Array<Vec<float32,4>> &dest,
                   const float32 divisor);

void init_constant(Array<Vec<float32,4>> &dest,
                   const float32 value);

} // namespace dray
#endif
