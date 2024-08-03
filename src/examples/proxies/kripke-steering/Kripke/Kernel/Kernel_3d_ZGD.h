#ifndef KRIPKE_KERNEL_3D_ZGD_H__
#define KRIPKE_KERNEL_3D_ZGD_H__

#include<Kripke/Kernel.h>

class Kernel_3d_ZGD : public Kernel {
  public:
    virtual Nesting_Order nestingPsi(void) const;
    virtual Nesting_Order nestingPhi(void) const;
    virtual Nesting_Order nestingSigt(void) const;
    virtual Nesting_Order nestingEll(void) const;
    virtual Nesting_Order nestingEllPlus(void) const;
    virtual Nesting_Order nestingSigs(void) const;

    virtual void LTimes(Grid_Data *grid_data);
    virtual void LPlusTimes(Grid_Data *grid_data);
    virtual void scattering(Grid_Data *grid_data);
    virtual void source(Grid_Data *grid_data);
    virtual void sweep(Subdomain *ga_set);
};

#endif
