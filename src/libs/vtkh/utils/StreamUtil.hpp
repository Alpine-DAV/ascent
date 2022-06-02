#ifndef __STREAM_UTIL_COMM_H
#define __STREAM_UTIL_COMM_H

#include <list>
#include <vector>
#include <map>
#include <iostream>
#include <deque>
#include <vtkm/cont/DataSet.h>

namespace vtkh
{

template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::list<T> &l)
{
    os<<"{";
    for (auto it = l.begin(); it != l.end(); it++)
        os<<(*it)<<" ";
    os<<"}";
    return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::deque<T> &l)
{
    os<<"{";
    for (auto it = l.begin(); it != l.end(); it++)
        os<<(*it)<<" ";
    os<<"}";
    return os;
}

// Forward declaration so we can have pairs with vectors
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v);

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os, const std::pair<T1,T2> &p)
{
    os<<"("<<p.first<<","<<p.second<<")";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os<<"[";
    int n = v.size();
    if (n>0)
    {
        for (int i = 0; i < n-1; i++) os<<v[i]<<" ";
        os<<v[n-1];
    }
    os<<"]";
    return os;
}

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os, const std::map<T1,T2> &m)
{
    os<<"{";
    for (auto it = m.begin(); it != m.end(); it++)
        os<<"("<<it->first<<","<<it->second<<") ";
    os<<"}";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const vtkm::cont::DataSet &ds)
{
    ds.PrintSummary(os);
    return os;
}

}

#endif //__STREAM_UTIL_H
