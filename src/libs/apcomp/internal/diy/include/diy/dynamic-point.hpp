#ifndef DIY_DYNAMIC_POINT_HPP
#define DIY_DYNAMIC_POINT_HPP

#include <string>
#include <sstream>
#include <vector>

#include "constants.h"
#include "chobo/small_vector.hpp"

namespace apcompdiy
{

template<class Coordinate_, size_t static_size = DIY_MAX_DIM>
class DynamicPoint: public chobo::small_vector<Coordinate_, static_size>
{
    public:
        using Coordinate    = Coordinate_;
        using Parent        = chobo::small_vector<Coordinate_, static_size>;

        template<class U>
        struct rebind       { typedef DynamicPoint<U> type; };

    public:
                            DynamicPoint(int dim, Coordinate x = 0):
                                Parent(dim, x)                      {}
        template<class T>   DynamicPoint(const DynamicPoint<T>& p)  { for (size_t i = 0; i < dimension(); ++i) (*this)[i] = p[i]; }
        template<class T>   DynamicPoint(const T* a, int dim)       { for (size_t i = 0; i < static_cast<size_t>(dim); ++i) (*this)[i] = a[i]; }
        template<class T>   DynamicPoint(const std::vector<T>& a):
                                Parent(a.begin(), a.end())          {}
                            DynamicPoint(std::initializer_list<Coordinate> lst):
                                Parent(lst.size())                  { size_t i = 0; for (Coordinate x : lst) (*this)[i++] = x; }

                            DynamicPoint(DynamicPoint&&)            =default;
                            DynamicPoint(const DynamicPoint&)       =default;
        DynamicPoint&       operator=(const DynamicPoint&)          =default;

        unsigned            dimension() const                       { return Parent::size(); }

        static DynamicPoint zero(int dim)                           { return DynamicPoint(dim, 0); }
        static DynamicPoint one(int dim)                            { return DynamicPoint(dim, 1); }

        DynamicPoint        drop(int dim) const                     { DynamicPoint p(dimension() - 1); size_t c = 0; for (size_t i = 0; i < dimension();   ++i) { if (i == dim) continue; p[c++] = (*this)[i]; } return p; }
        DynamicPoint        lift(int dim, Coordinate x) const       { DynamicPoint p(dimension() + 1); for (size_t i = 0; i < dimension()+1; ++i) { if (i < dim) p[i] = (*this)[i]; else if (i == dim) p[i] = x; else if (i > dim) p[i] = (*this)[i-1]; } return p; }

        using Parent::operator[];

        DynamicPoint&       operator+=(const DynamicPoint& y)       { for (size_t i = 0; i < dimension(); ++i) (*this)[i] += y[i];  return *this; }
        DynamicPoint&       operator-=(const DynamicPoint& y)       { for (size_t i = 0; i < dimension(); ++i) (*this)[i] -= y[i];  return *this; }
        DynamicPoint&       operator*=(Coordinate a)                { for (size_t i = 0; i < dimension(); ++i) (*this)[i] *= a;     return *this; }
        DynamicPoint&       operator/=(Coordinate a)                { for (size_t i = 0; i < dimension(); ++i) (*this)[i] /= a;     return *this; }

        DEPRECATED("Use norm2 instead")
        Coordinate          norm() const                            { return norm2(); }
        Coordinate          norm2() const                           { return (*this)*(*this); }

        std::ostream&       operator<<(std::ostream& out) const     { out << (*this)[0]; for (size_t i = 1; i < dimension(); ++i) out << " " << (*this)[i]; return out; }
        std::istream&       operator>>(std::istream& in);

        friend
        DynamicPoint        operator+(DynamicPoint x, const DynamicPoint& y)    { x += y; return x; }

        friend
        DynamicPoint        operator-(DynamicPoint x, const DynamicPoint& y)    { x -= y; return x; }

        friend
        DynamicPoint        operator/(DynamicPoint x, Coordinate y)             { x /= y; return x; }

        friend
        DynamicPoint        operator*(DynamicPoint x, Coordinate y)             { x *= y; return x; }

        friend
        DynamicPoint        operator*(Coordinate y, DynamicPoint x)             { x *= y; return x; }

        friend
        Coordinate          operator*(const DynamicPoint& x, const DynamicPoint& y)         { Coordinate n = 0; for (size_t i = 0; i < x.dimension(); ++i) n += x[i] * y[i]; return n; }

        template<class T, size_t s_>
        friend
        Coordinate          operator*(const DynamicPoint<T,s_>& x, const DynamicPoint& y)   { Coordinate n = 0; for (size_t i = 0; i < x.dimension(); ++i) n += x[i] * y[i]; return n; }
};

template<class C, size_t s_>
std::istream&
DynamicPoint<C,s_>::
operator>>(std::istream& in)
{
    std::string point_str;
    in >> point_str;        // read until ' '
    std::stringstream ps(point_str);

    char x;
    for (unsigned i = 0; i < dimension(); ++i)
    {
        ps >> (*this)[i];
        ps >> x;
    }

    return in;
}


template<class Coordinate, size_t s_>
Coordinate norm2(const DynamicPoint<Coordinate,s_>& p)
{ Coordinate res = 0; for (unsigned i = 0; i < p.dimension(); ++i) res += p[i]*p[i]; return res; }

template<class C, size_t s_>
std::ostream&
operator<<(std::ostream& out, const DynamicPoint<C,s_>& p)
{ return p.operator<<(out); }

template<class C, size_t s_>
std::istream&
operator>>(std::istream& in, DynamicPoint<C,s_>& p)
{ return p.operator>>(in); }

// Serialization
template<class T>
struct Serialization;
struct BinaryBuffer;
template<class T> void save(BinaryBuffer&, const T&);
template<class T> void load(BinaryBuffer&, T&);
template<class T> void save(BinaryBuffer&, const T*, size_t);
template<class T> void load(BinaryBuffer&, T*, size_t);

template<class C, size_t s_>
struct Serialization<DynamicPoint<C, s_>>
{
    using Point = DynamicPoint<C,s_>;

    static void         save(BinaryBuffer& bb, const Point& p)
    {
      size_t s = p.size();
      apcompdiy::save(bb, s);
      if (s > 0)
        apcompdiy::save(bb, &p[0], p.size());
    }

    static void         load(BinaryBuffer& bb, Point& p)
    {
      size_t s;
      apcompdiy::load(bb, s);
      p.resize(s);
      if (s > 0)
        apcompdiy::load(bb, &p[0], s);
    }
};

}

#endif // DIY_POINT_HPP
