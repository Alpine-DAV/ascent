#ifndef _COMM_DATA_H
#define _COMM_DATA_H

#include <vector>
#include <iostream>
#include <vtkh/filters/util.hpp>

class MsgCommData
{
  public:
    MsgCommData() {rank=-1;}
    MsgCommData(int r, const std::vector<int> &m) {rank=r; message = m;}
    MsgCommData(const MsgCommData &d) {rank=d.rank; message=d.message;}

    MsgCommData &operator=(const MsgCommData &d) {rank=d.rank; message=d.message; return *this; }

    int rank;
    std::vector<int> message;
};

inline std::ostream &operator<<(std::ostream &os, const MsgCommData &m)
{
    os<<"(msg: "<<m.rank<<" "<<m.message<<")";
    return os;
}

template<class T>
class ParticleCommData
{
  public:
    ParticleCommData() {rank=-1;}
    ParticleCommData(int r, const T &_p) {rank=r; p=_p;}
    ParticleCommData(const ParticleCommData &d) {rank=d.rank; p=d.p;}

    ParticleCommData &operator=(const ParticleCommData &d) {rank=d.rank; p=d.p; return *this; }

    int rank;
    T p;
};

template<class T>
inline std::ostream &operator<<(std::ostream &os, const ParticleCommData<T> &p)
{
    os<<"(pmsg: "<<p.rank<<" "<<p.p<<")";
    return os;
}

class DSCommData
{
typedef long BlockIDType;

  public:
    DSCommData() {ds=NULL;}
//    DSCommData(BlockIDType &_dom, vtkDataSet *_ds) {dom=_dom; ds=_ds;}
    DSCommData(const DSCommData &d) {ds=d.ds; dom=d.dom;}

    DSCommData &operator=(const DSCommData &d) {ds=d.ds; dom=d.dom; return *this; }

    BlockIDType dom;
    void *ds;
//    vtkDataSet *ds;
};

#endif
