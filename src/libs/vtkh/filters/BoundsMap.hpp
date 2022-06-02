#ifndef VTK_H_BOUNDS_MAP_HPP
#define VTK_H_BOUNDS_MAP_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <deque>
#include <algorithm>
#include <map>

class BoundsMap
{
public:
    BoundsMap() {}
    BoundsMap(const BoundsMap &_bm) : bm(_bm.bm) {}

    void AddBlock(int id, const vtkm::Bounds &bounds)
    {
        if (bm.find(id) == bm.end())
            bm[id] = bounds;
        else
            throw "Duplicate block";
    }
    template <template <typename, typename> class Container,
              typename Allocator=std::allocator<Particle>>
    void FindBlockIDs(const Container<Particle, Allocator> &particles, std::vector<int> &blockIDs) const
    {
        size_t sz = particles.size();
        blockIDs.resize(sz);
        auto pit = particles.begin();
        auto oit = blockIDs.begin();
        for ( ; pit != particles.end(); pit++, oit++)
            *oit = FindBlock(pit->coords);
    }
/*
    void FindBlockIDs(const std::vector<Particle> &particles, std::vector<int> &blockIDs) const
    {
        size_t sz = particles.size();
        blockIDs.resize(sz);
        auto pit = particles.begin();
        auto oit = blockIDs.begin();
        for ( ; pit != particles.end(); pit++, oit++)
            *oit = FindBlock(pit->coords);
    }
*/
    int FindBlock(const vtkm::Vec<float,3> &pt) const
    {
        for (auto it = bm.begin(); it != bm.end(); it++)
        {
            if (pt[0] >= it->second.X.Min &&
                pt[0] < it->second.X.Max &&
                pt[1] >= it->second.Y.Min &&
                pt[1] < it->second.Y.Max &&
                pt[2] >= it->second.Z.Min &&
                pt[2] < it->second.Z.Max)
            {
                return it->first;
            }
        }
        return -1;
    }

    std::map<int, vtkm::Bounds> bm;
};

#endif
