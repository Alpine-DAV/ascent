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

//-----------------------------------------------------------------------------
template<typename T>
struct MemsetFunctor
{
    T m_value = T(0);

    //---------------------------------------------------------------------------
    template<typename Exec>
    void operator()(Array<T> &array,
                  const Exec &) const
    {
        const int size = array.size();
        const T value = m_value;
        T *array_ptr = array.get_ptr(Exec::memory_space);
        using for_policy = typename Exec::for_policy;
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
        {
            array_ptr[i] = value;
        });
        ASCENT_DEVICE_ERROR_CHECK();
    }
};

//-----------------------------------------------------------------------------
template <typename T>
void
array_memset_impl(Array<T> &array, const T val)
{
    detail::MemsetFunctor<T> func;
    func.m_value = val;
    exec_dispatch_array(array, func);
}

} // namespace detail


//-----------------------------------------------------------------------------
void
array_memset(Array<double> &array, const double val)
{
    detail::array_memset_impl(array,val);
}

//-----------------------------------------------------------------------------
void
array_memset(Array<int> &array, const int val)
{
    detail::array_memset_impl(array,val);
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
