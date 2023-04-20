//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_data.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_DATA_HPP
#define FLOW_DATA_HPP

#include <conduit.hpp>

#include <flow_exports.h>
#include <flow_config.h>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{


//-----------------------------------------------------------------------------
/// Container that  wrappers inputs and output datasets from filters so
/// they can be managed by the registry.
///
/// Key features:
///
/// Provides easy access to specific wrapped data:
///    (so far just a Conduit Node Pointer)
///   Data can cast to conduit::Node *, or conduit::Node & .
///
///
/// Provides a release() method used by the registry to manage result lifetimes.
//
//-----------------------------------------------------------------------------

// forward declare so we can use dynamic cast in our check_type() method.
template <class T>
class DataWrapper;

//-----------------------------------------------------------------------------
class FLOW_API Data
{
public:
    Data(void *data);

    virtual ~Data();


    // creates a new container for given data
    virtual Data  *wrap(void *data)   = 0;
    // actually delete the data
    virtual void            release() = 0;

    void          *data_ptr();
    const  void   *data_ptr() const;

    // access methods
    template <class T>
    T *value()
    {
        return static_cast<T*>(data_ptr());
    }

    template <class T>
    bool check_type() const
    {
        const DataWrapper<T> *check = dynamic_cast<const DataWrapper<T>*>(this);
        return check != NULL;
    }


    template <class T>
    const T *value() const
    {
        return static_cast<T*>(data_ptr());
    }


    void        info(conduit::Node &out) const;
    std::string to_json() const;
    void        print() const;

protected:
    void    set_data_ptr(void *);

private:
    void *m_data_ptr;

};

//-----------------------------------------------------------------------------
template <class T>
class DataWrapper: public Data
{
 public:

    DataWrapper(void *data)
    : Data(data)
    {
        // empty
    }

    virtual ~DataWrapper()
    {
        // empty
    }

    Data *wrap(void *data)
    {
        return new DataWrapper<T>(data);
    }

    virtual void release()
    {
        if(data_ptr() != NULL)
        {
            T * t = static_cast<T*>(data_ptr());
            delete t;
            set_data_ptr(NULL);
        }
    }
};


// this needs to be declared here to cement proper symbol visibly
// to use runtime type checking in further libs
template class FLOW_API DataWrapper<conduit::Node>;

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------



#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


