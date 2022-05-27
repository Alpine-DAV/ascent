#include <vtkh/utils/Mutex.hpp>

#ifdef VTKH_USE_OPENMP
#include <omp.h>
#else
#include <thread>
#include <mutex>
#endif

namespace vtkh
{
struct Mutex::InternalsType
{
#ifdef VTKH_USE_OPENMP
  omp_lock_t lock;
#else
  std::mutex lock;
#endif
};

//Mutex class for both openmp and std::mutex

//openMP version
#ifdef VTKH_USE_OPENMP
Mutex::Mutex()
  : m_internals(new InternalsType)
{
  omp_init_lock(&(m_internals->lock));
}

Mutex::~Mutex()
{
  omp_destroy_lock(&(m_internals->lock));
}

void Mutex::Lock()
{
  omp_set_lock(&(m_internals->lock));
}

void Mutex::Unlock()
{
  omp_unset_lock(&(m_internals->lock));
}

//std::mutex version
#else
Mutex::Mutex()
  : m_internals(new InternalsType)
{
}

Mutex::~Mutex()
{
}

void Mutex::Lock()
{
  m_internals->lock.lock();
}

void Mutex::Unlock()
{
  m_internals->lock.unlock();
}
#endif

} //namespace vtkh
