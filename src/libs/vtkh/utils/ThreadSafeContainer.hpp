#ifndef VTK_H_THREAD_SAFE_CONTAINER_HPP
#define VTK_H_THREAD_SAFE_CONTAINER_HPP

#include <vtkh/vtkh_exports.h>
#include <algorithm>
#include <vtkh/utils/Mutex.hpp>

namespace vtkh
{

template <typename T,
          template <typename, typename> class Container>
class VTKH_API ThreadSafeContainer
{
public:
 ThreadSafeContainer()
 {
 }

 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 ThreadSafeContainer(const C<T,Allocator> &c)
 {
     Assign(c);
 }

 ThreadSafeContainer(const ThreadSafeContainer<T, Container> &c)
 {
     if (this == &c)
         throw "ERROR: attempting to assign thread identical thread safe containers.";

     Container<T, std::allocator<T>> tmp;
     c.Get(tmp);
     Assign(tmp);
 }

 ~ThreadSafeContainer()
 {
     Clear();
 }

 bool Empty()
 {
     lock.Lock();
     bool val = data.empty();
     lock.Unlock();
     return val;
 }
 size_t Size()
 {
     lock.Lock();
     size_t sz = data.size();
     lock.Unlock();
     return sz;
 }
 void Clear()
 {
     lock.Lock();
     data.clear();
     lock.Unlock();
 }

 //Add/set elements
 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 void Insert(const C<T, Allocator> &c)
 {
     if (c.empty())
         return;

     lock.Lock();
     data.insert(data.end(), c.begin(), c.end());
     lock.Unlock();
 }

 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 void Assign(const C<T, Allocator> &c)
 {
     lock.Lock();
     data.clear();
     data.insert(data.end(), c.begin(), c.end());
     lock.Unlock();
 }

 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 Container<T, std::allocator<T>>& operator=(const C<T, Allocator> &c)
 {
     lock.Lock();
     data.clear();
     data.insert(data.end(), c.begin(), c.end());
     lock.Unlock();
     return *this;
 }

 //Get elements
 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 bool Get(C<T, Allocator> &c)
 {
     lock.Lock();
     c.insert(c.end(), data.begin(), data.end());
     data.clear();
     lock.Unlock();

     return !c.empty();
 }

 //Get elements
 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 bool Get(C<T, Allocator> &c, std::size_t N)
 {
     lock.Lock();
     std::size_t n = std::min(N, data.size());
     c.insert(c.end(), data.begin(), data.begin()+n);
     data.erase(data.begin(), data.begin()+n);
     lock.Unlock();

     return !c.empty();
 }

 template <template <typename, typename> class C,
           typename Allocator=std::allocator<T>>
 void Put(C<T, Allocator> &c)
 {
     if (c.empty())
         return;

     lock.Lock();
     data.insert(data.end(), c.begin(), c.end());
     lock.Unlock();
 }

 friend std::ostream &
 operator<<(std::ostream &os, ThreadSafeContainer<T, Container> &c)
 {
    c.lock.Lock();
    os<<"ts_(("<<c.data<<"))";
    c.lock.Unlock();
    return os;
 }

protected:
 Container<T, std::allocator<T>> data;
 vtkh::Mutex lock;
};

};

#endif //VTK_H_THREAD_SAFE_CONTAINER_HPP
