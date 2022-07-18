#ifndef DIY_LINK_HPP
#define DIY_LINK_HPP

#include <vector>
#include <map>
#include <algorithm>

#include "types.hpp"
#include "serialization.hpp"
#include "assigner.hpp"

#include "factory.hpp"

namespace apcompdiy
{
  // Local view of a distributed representation of a cover, a completely unstructured link
  class Link: public Factory<Link>
  {
    public:
      using Neighbors = std::vector<BlockID>;

                Link(Key)                           {}  // for Factory
                Link()                              = default;
                Link(const Link&)                   = default;
                Link(Link&&)                        = default;
      Link&     operator=(const Link&)              = default;
      Link&     operator=(Link&&)                   = default;
      virtual   ~Link()                             {}  // need to be able to delete derived classes

      int       size() const                        { return static_cast<int>(neighbors_.size()); }
      inline
      int       size_unique() const;
      BlockID   target(int i) const                 { return neighbors_[static_cast<size_t>(i)]; }
      BlockID&  target(int i)                       { return neighbors_[static_cast<size_t>(i)]; }
      inline
      int       find(int gid) const;

      void      add_neighbor(const BlockID& block)  { neighbors_.push_back(block); }

      void      fix(const Assigner& assigner)       { for (unsigned i = 0; i < neighbors_.size(); ++i) { neighbors_[i].proc = assigner.rank(neighbors_[i].gid); } }

      void      swap(Link& other)                   { neighbors_.swap(other.neighbors_); }

      const Neighbors&
                neighbors() const                   { return neighbors_; }
      Neighbors&
                neighbors()                         { return neighbors_; }

      virtual Link* clone() const                   { return new Link(*this); }

      virtual void  save(BinaryBuffer& bb) const    { apcompdiy::save(bb, neighbors_); }
      virtual void  load(BinaryBuffer& bb)          { apcompdiy::load(bb, neighbors_); }

    private:
      Neighbors neighbors_;
  };

  template<class Bounds_>
  class RegularLink;

  using RegularGridLink         = RegularLink<DiscreteBounds>;
  using RegularContinuousLink   = RegularLink<ContinuousBounds>;

  // for a regular decomposition, it makes sense to address the neighbors by direction
  // and store local and neighbor bounds
  template<class Bounds_>
  class RegularLink: public Link::Registrar<RegularLink<Bounds_>>
  {
    public:
      typedef   Bounds_                             Bounds;

      typedef   std::map<Direction, int>            DirMap;
      typedef   std::vector<Direction>              DirVec;

    public:
                RegularLink():
                  dim_(0), core_(0), bounds_(0)               {}        // for Factory
                RegularLink(int dim, const Bounds& core__, const Bounds& bounds__):
                  dim_(dim), core_(core__), bounds_(bounds__) {}

      // dimension
      int       dimension() const                       { return dim_; }

      // direction
      int       direction(Direction dir) const;         // convert direction to a neighbor (-1 if no neighbor)
      Direction direction(int i) const                  { return dir_vec_[i]; }
      void      add_direction(Direction dir)            { int c = dir_map_.size(); dir_map_[dir] = c; dir_vec_.push_back(dir); }

      // wrap
      void       add_wrap(Direction dir)                { wrap_.push_back(dir); }
      Direction  wrap(int i) const                      { return wrap_[i]; }
      Direction& wrap(int i)                            { return wrap_[i]; }

      // bounds
      const Bounds& core() const                        { return core_; }
      Bounds&       core()                              { return core_; }
      const Bounds& bounds() const                      { return bounds_; }
      Bounds&       bounds()                            { return bounds_; }
      const Bounds& core(int i) const                   { return nbr_cores_[i]; }
      const Bounds& bounds(int i) const                 { return nbr_bounds_[i]; }
      void          add_core(const Bounds& core__)      { nbr_cores_.push_back(core__); }
      void          add_bounds(const Bounds& bounds__)  { nbr_bounds_.push_back(bounds__); }

      void      swap(RegularLink& other)                { Link::swap(other); dir_map_.swap(other.dir_map_); dir_vec_.swap(other.dir_vec_); nbr_bounds_.swap(other.nbr_bounds_); std::swap(dim_, other.dim_); wrap_.swap(other.wrap_); std::swap(core_, other.core_); std::swap(bounds_, other.bounds_); }

      Link*     clone() const override                  { return new RegularLink(*this); }

      void      save(BinaryBuffer& bb) const override
      {
          Link::save(bb);
          apcompdiy::save(bb, dim_);
          apcompdiy::save(bb, dir_map_);
          apcompdiy::save(bb, dir_vec_);
          apcompdiy::save(bb, core_);
          apcompdiy::save(bb, bounds_);
          apcompdiy::save(bb, nbr_cores_);
          apcompdiy::save(bb, nbr_bounds_);
          apcompdiy::save(bb, wrap_);
      }

      void      load(BinaryBuffer& bb) override
      {
          Link::load(bb);
          apcompdiy::load(bb, dim_);
          apcompdiy::load(bb, dir_map_);
          apcompdiy::load(bb, dir_vec_);
          apcompdiy::load(bb, core_);
          apcompdiy::load(bb, bounds_);
          apcompdiy::load(bb, nbr_cores_);
          apcompdiy::load(bb, nbr_bounds_);
          apcompdiy::load(bb, wrap_);
      }

    private:
      int       dim_;

      DirMap    dir_map_;
      DirVec    dir_vec_;

      Bounds                    core_;
      Bounds                    bounds_;
      std::vector<Bounds>       nbr_cores_;
      std::vector<Bounds>       nbr_bounds_;
      std::vector<Direction>    wrap_;
  };

  struct AMRLink: public Link::Registrar<AMRLink>
  {
    public:
      using Bounds      = DiscreteBounds;
      using Directions  = std::vector<Direction>;
      using Point       = Bounds::Point;

      struct Description
      {
          int       level       { -1 };
          Point     refinement  { 0 };      // refinement of this level w.r.t. level 0
          Bounds    core        { 0 };
          Bounds    bounds      { 0 };      // with ghosts

                    Description() = default;
                    Description(int level_, Point refinement_, Bounds core_, Bounds bounds_):
                        level(level_), refinement(refinement_), core(core_), bounds(bounds_)    {}
      };
      using Descriptions = std::vector<Description>;

    public:
                    AMRLink(int dim, int level, Point refinement, const Bounds& core, const Bounds& bounds):
                        dim_(dim), local_ { level, refinement, core, bounds }               {}
                    AMRLink(int dim, int level, int refinement, const Bounds& core, const Bounds& bounds):
                        AMRLink(dim, level, refinement * Point::one(dim), core, bounds)     {}
                    AMRLink(): AMRLink(0, -1, 0, Bounds(0), Bounds(0))                      {}        // for Factory

      // dimension
      int           dimension() const                       { return dim_; }

      // local information
      int           level() const                           { return local_.level; }
      int           level(int i) const                      { return nbr_descriptions_[i].level; }
      Point         refinement() const                      { return local_.refinement; }
      Point         refinement(int i) const                 { return nbr_descriptions_[i].refinement; }

      // wrap
      void          add_wrap(Direction dir)                 { wrap_.push_back(dir); }
      const Directions&
                    wrap() const                            { return wrap_; }

      // bounds
      const Bounds& core() const                            { return local_.core; }
      Bounds&       core()                                  { return local_.core; }
      const Bounds& bounds() const                          { return local_.bounds; }
      Bounds&       bounds()                                { return local_.bounds; }
      const Bounds& core(int i) const                       { return nbr_descriptions_[i].core; }
      const Bounds& bounds(int i) const                     { return nbr_descriptions_[i].bounds; }
      void          add_bounds(int level_,
                               Point refinement_,
                               const Bounds& core_,
                               const Bounds& bounds_)       { nbr_descriptions_.emplace_back(Description {level_, refinement_, core_, bounds_}); }
      void          add_bounds(int level_,
                               int refinement_,
                               const Bounds& core_,
                               const Bounds& bounds_)       { add_bounds(level_, refinement_ * Point::one(dim_), core_, bounds_); }

      Link*         clone() const override                  { return new AMRLink(*this); }

      void          save(BinaryBuffer& bb) const override
      {
          Link::save(bb);
          apcompdiy::save(bb, dim_);
          apcompdiy::save(bb, local_);
          apcompdiy::save(bb, nbr_descriptions_);
          apcompdiy::save(bb, wrap_);
      }

      void          load(BinaryBuffer& bb) override
      {
          Link::load(bb);
          apcompdiy::load(bb, dim_);
          apcompdiy::load(bb, local_);
          apcompdiy::load(bb, nbr_descriptions_);
          apcompdiy::load(bb, wrap_);
      }

    private:
        int                         dim_;

        Description                 local_;
        Descriptions                nbr_descriptions_;
        Directions                  wrap_;
  };

  struct LinkFactory
  {
    public:
      static Link*          create(std::string name)
      {
          return Link::make(name);
      }

      inline static void    save(BinaryBuffer& bb, const Link* l);
      inline static Link*   load(BinaryBuffer& bb);
  };

  namespace detail
  {
      inline void instantiate_common_regular_links()
      {
          // Instantiate the common types to register them
          RegularLink<Bounds<int>>      rl_int;
          RegularLink<Bounds<float>>    rl_float;
          RegularLink<Bounds<double>>   rl_double;
          RegularLink<Bounds<long>>     rl_long;
      }
  }

    template<>
    struct Serialization<apcompdiy::AMRLink::Description>
    {
        static void         save(apcompdiy::BinaryBuffer& bb, const apcompdiy::AMRLink::Description& x)
        {
            apcompdiy::save(bb, x.level);
            apcompdiy::save(bb, x.refinement);
            apcompdiy::save(bb, x.core);
            apcompdiy::save(bb, x.bounds);
        }

        static void         load(apcompdiy::BinaryBuffer& bb, apcompdiy::AMRLink::Description& x)
        {
            apcompdiy::load(bb, x.level);
            apcompdiy::load(bb, x.refinement);
            apcompdiy::load(bb, x.core);
            apcompdiy::load(bb, x.bounds);
        }
    };
}


void
apcompdiy::LinkFactory::
save(BinaryBuffer& bb, const Link* l)
{
    apcompdiy::save(bb, l->id());
    l->save(bb);
}

apcompdiy::Link*
apcompdiy::LinkFactory::
load(BinaryBuffer& bb)
{
    std::string id;
    apcompdiy::load(bb, id);
    Link* l = create(id);
    l->load(bb);
    return l;
}

int
apcompdiy::Link::
find(int gid) const
{
    for (int i = 0; i < size(); ++i)
  {
    if (target(i).gid == gid)
      return i;
  }
  return -1;
}
int
apcompdiy::Link::
size_unique() const
{
    std::vector<BlockID> tmp(neighbors_.begin(), neighbors_.end());
    std::sort(tmp.begin(), tmp.end());
    return static_cast<int>(std::unique(tmp.begin(), tmp.end()) - tmp.begin());
}

template<class Bounds>
int
apcompdiy::RegularLink<Bounds>::
direction(Direction dir) const
{
  DirMap::const_iterator it = dir_map_.find(dir);
  if (it == dir_map_.end())
    return -1;
  else
    return it->second;
}

#endif      // DIY_LINK_HPP
