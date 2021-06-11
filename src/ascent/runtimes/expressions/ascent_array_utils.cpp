#include <expressions/ascent_array_utils.hpp>
#include <ascent_config.h>
#include <expressions/ascent_dispatch.hpp>

#include <RAJA/RAJA.hpp>

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
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{

template<typename T>
struct MemsetFunctor
{
  T m_value = T(0);

  template<typename Exec>
  void operator()(Array<T> &array,
                  const Exec &) const
  {
    const int size = array.size();

    const T value = m_value;

    using fp = typename Exec::for_policy;

    T *array_ptr = array.ptr(Exec::memory_space);

    RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
    {
      array_ptr[i] = value;
    });
    ASCENT_ERROR_CHECK();

  }
};

} // namespace detail

template <typename T>
void array_memset_impl(Array<T> &array, const T val)
{
  detail::MemsetFunctor<T> func;
  func.m_value = val;
  exec_dispatch_array(array, func);
}

void array_memset(Array<double> &array, const double val)
{
  array_memset_impl(array,val);
}

void array_memset(Array<int> &array, const int val)
{
  array_memset_impl(array,val);
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
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
