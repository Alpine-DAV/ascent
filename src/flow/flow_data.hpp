//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_data.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_DATA_HPP
#define FLOW_DATA_HPP

#include <conduit.hpp>


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
class Data
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



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------



#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


