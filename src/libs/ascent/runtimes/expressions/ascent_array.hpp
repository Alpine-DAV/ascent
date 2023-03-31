//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_ARRAY_HPP
#define ASCENT_ARRAY_HPP

#include <memory>
#include <string>

namespace ascent
{

namespace runtime
{

// forward declaration of internals
template <typename t> class ArrayInternals;

template <typename T> class Array
{
  public:
    Array ();
    // zero copy a pointer provided by an external source
    Array(T *data, const size_t size);
    ~Array();

    // copy data from this data pointer
    void    copy(const T *data, const size_t size);
    size_t  size() const;
    void    resize(const size_t size);
    // zero copy a pointer provided by an external source
    void    set(T *data, const size_t size);

    T       *get_host_ptr();
    T       *get_device_ptr();
    const T *get_host_ptr_const() const;
    const T *get_device_ptr_const() const;

    T       *get_ptr(const std::string loc);
    const T *get_ptr_const(const std::string loc) const;

    // gets a single value and does not synch data between
    // host and device
    T        get_value(const size_t i) const;

    void     summary();
    void     status();
    void     operator=(const Array<T> &other);

    Array<T> copy();

  protected:
    std::shared_ptr<ArrayInternals<T>> m_internals;
};

} // namespace runtime
} // namespace ascent
#endif
