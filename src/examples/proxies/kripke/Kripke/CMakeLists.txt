
set(KRIPKE_SOURCE ${KRIPKE_SOURCE}
  Kripke/Directions.cpp  
  Kripke/Grid.cpp  
  Kripke/Kernel.cpp  
  Kripke/Layout.cpp
  Kripke/Subdomain.cpp  
  Kripke/Sweep_Solver.cpp
  Kripke/ParallelComm.cpp  
  Kripke/Timing.cpp  
    
  Kripke/Kernel/Kernel_3d_GDZ.cpp
  Kripke/Kernel/Kernel_3d_DGZ.cpp
  Kripke/Kernel/Kernel_3d_ZDG.cpp
  Kripke/Kernel/Kernel_3d_DZG.cpp
  Kripke/Kernel/Kernel_3d_ZGD.cpp
  Kripke/Kernel/Kernel_3d_GZD.cpp
  
  Kripke/ParallelComm/BlockJacobiComm.cpp
  Kripke/ParallelComm/SweepComm.cpp 
)
                                                                                                          
set(KRIPKE_SOURCE ${KRIPKE_SOURCE} PARENT_SCOPE) 
