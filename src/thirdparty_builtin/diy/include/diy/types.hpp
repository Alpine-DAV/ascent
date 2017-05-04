#ifndef DIY_TYPES_HPP
#define DIY_TYPES_HPP

#include <iostream>
#include "constants.h"
#include "point.hpp"

namespace diy
{
    struct BlockID
    {
        int gid, proc;
    };

    template<class Coordinate_>
    struct Bounds
    {
        using Coordinate = Coordinate_;

        Point<Coordinate, DIY_MAX_DIM>    min, max;
    };
    using DiscreteBounds   = Bounds<int>;
    using ContinuousBounds = Bounds<float>;

    //! Helper to create a 1-dimensional discrete domain with the specified extents
    inline
    diy::DiscreteBounds
    interval(int from, int to)            { DiscreteBounds domain; domain.min[0] = from; domain.max[0] = to; return domain; }

    struct Direction: public Point<int,DIY_MAX_DIM>
    {
              Direction()                 { for (int i = 0; i < DIY_MAX_DIM; ++i) (*this)[i] = 0; }
              Direction(int dir)
      {
          for (int i = 0; i < DIY_MAX_DIM; ++i) (*this)[i] = 0;
          if (dir & DIY_X0) (*this)[0] -= 1;
          if (dir & DIY_X1) (*this)[0] += 1;
          if (dir & DIY_Y0) (*this)[1] -= 1;
          if (dir & DIY_Y1) (*this)[1] += 1;
          if (dir & DIY_Z0) (*this)[2] -= 1;
          if (dir & DIY_Z1) (*this)[2] += 1;
          if (dir & DIY_T0) (*this)[3] -= 1;
          if (dir & DIY_T1) (*this)[3] += 1;
      }

      bool
      operator==(const diy::Direction& y) const
      {
        for (int i = 0; i < DIY_MAX_DIM; ++i)
            if ((*this)[i] != y[i]) return false;
        return true;
      }

      // lexicographic comparison
      bool
      operator<(const diy::Direction& y) const
      {
        for (int i = 0; i < DIY_MAX_DIM; ++i)
        {
            if ((*this)[i] < y[i]) return true;
            if ((*this)[i] > y[i]) return false;
        }
        return false;
      }
    };

    // Selector of bounds value type
    template<class Bounds_>
    struct BoundsValue
    {
        using type = typename Bounds_::Coordinate;
    };

    inline
    bool
    operator<(const diy::BlockID& x, const diy::BlockID& y)
    { return x.gid < y.gid; }

    inline
    bool
    operator==(const diy::BlockID& x, const diy::BlockID& y)
    { return x.gid == y.gid; }
}

#endif
