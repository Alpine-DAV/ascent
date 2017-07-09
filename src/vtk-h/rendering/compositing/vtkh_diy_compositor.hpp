#ifndef VTKH_DIY_COMPOSITOR_HPP
#define VTKH_DIY_COMPOSITOR_HPP

#include <rendering/vtkh_image.hpp>
#include "vtkh_compositor.hpp"
#include <diy/mpi.hpp>
#include <iostream>

namespace vtkh 
{

class DIYCompositor : public Compositor
{
public:
     DIYCompositor();
    ~DIYCompositor();
    
    void Cleanup();
    
private:
    virtual void CompositeZBufferSurface() override;
    virtual void CompositeZBufferBlend() override;
    virtual void CompositeVisOrder() override;
    diy::mpi::communicator   m_diy_comm;
    int                      m_rank;
};

}; // namespace vtkh

#endif


