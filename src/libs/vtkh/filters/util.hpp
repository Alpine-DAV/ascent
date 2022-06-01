#ifndef __UTIL_H
#define __UTIL_H

#include <vector>
#include <iostream>

template <typename T>
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

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
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

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os, const std::pair<T1,T2> &p)
{
    os<<"("<<p.first<<","<<p.second<<")";
    return os;
}

#if 0

inline std::ostream &operator<<(std::ostream &os, Bounds const &b)
{
    os<<"[("<<b.min[0]<<" "<<b.max[0]<<") ";
    os<<"("<<b.min[1]<<" "<<b.max[1]<<") ";
    os<<"("<<b.min[2]<<" "<<b.max[2]<<")]";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, diy::BlockID const &id)
{
    os<<"("<<id.gid<<","<<id.proc<<")";
    return os;
}

#include <vtkDataSet.h>
#include <vtkDataSetWriter.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>

static void dumpDS(vtkDataSet *ds, const std::string &nm, bool doBinary=false)
{
    vtkDataSetWriter *wrt = vtkDataSetWriter::New();
    wrt->SetFileName(nm.c_str());
//    if (doBinary)
  //      wrt->SetFileTypeToBinary();
   // else
        wrt->SetFileTypeToASCII();

    wrt->SetInputData(ds);
    wrt->Write();
    wrt->Delete();
}

static void
dumpStreamlines(const std::vector<std::vector<float> > &streamlines,
                const std::string &fname)
{
    vtkPolyData *pd = vtkPolyData::New();
    vtkPoints *pts = vtkPoints::New();
    vtkCellArray *cells = vtkCellArray::New();

    pd->SetPoints(pts);
    pd->SetLines(cells);

    pts->Delete();
    cells->Delete();

    int ptID = 0;
    for (int i = 0; i < streamlines.size(); i++)
    {
        int nPts = streamlines[i].size()/3;
        for (int j = 0; j < streamlines[i].size(); j+=3)
            pts->InsertNextPoint(streamlines[i][j+0],
                                 streamlines[i][j+1],
                                 streamlines[i][j+2]);
        vtkPolyLine *pline = vtkPolyLine::New();
        pline->GetPointIds()->SetNumberOfIds(nPts);
        for (int j = 0; j < nPts; j++)
            pline->GetPointIds()->SetId(j, ptID++);
        cells->InsertNextCell(pline);
        pline->Delete();
    }

    dumpDS(pd, fname);
}
#endif

#endif //__UTIL_H
