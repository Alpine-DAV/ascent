!Crown Copyright 2012 AWE.
!
! This file is part of CloverLeaf.
!
! CloverLeaf is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! CloverLeaf is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! CloverLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Fortran timestep kernel
!>  @author Wayne Gaudin
!>  @details Calculates the minimum timestep on the mesh chunk based on the CFL
!>  condition, the velocity gradient and the velocity divergence. A safety
!>  factor is used to ensure numerical stability.

MODULE calc_dt_kernel_module

CONTAINS

SUBROUTINE calc_dt_kernel(x_min,x_max,y_min,y_max,z_min,z_max, &
                          g_small,g_big,dtmin,                 &
                          dtc_safe,                            &
                          dtu_safe,                            &
                          dtv_safe,                            &
                          dtw_safe,                            &
                          dtdiv_safe,                          &
                          xarea,                               &
                          yarea,                               &
                          zarea,                               &
                          cellx,                               &
                          celly,                               &
                          cellz,                               &
                          celldx,                              &
                          celldy,                              &
                          celldz,                              &
                          volume,                              &
                          density0,                            &
                          energy0,                             &
                          pressure,                            &
                          viscosity_a,                         &
                          soundspeed,                          &
                          xvel0,yvel0,zvel0,                   &
                          dt_min,                              &
                          dt_min_val,                          &
                          dtl_control,                         &
                          xl_pos,                              &
                          yl_pos,                              &
                          zl_pos,                              &
                          jldt,                                &
                          kldt,                                &
                          lldt,                                &
                          small)

  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  REAL(KIND=8)  :: g_small,g_big,dtmin,dt_min_val
  REAL(KIND=8)  :: dtc_safe,dtu_safe,dtv_safe,dtw_safe,dtdiv_safe
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: xarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: yarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: zarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2)             :: cellx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2)             :: celly
  REAL(KIND=8), DIMENSION(z_min-2:z_max+2)             :: cellz
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2)             :: celldx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2)             :: celldy
  REAL(KIND=8), DIMENSION(z_min-2:z_max+2)             :: celldz
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: volume
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: energy0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: pressure
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: viscosity_a
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: soundspeed
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel0,yvel0,zvel0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: dt_min

  INTEGER          :: dtl_control
  REAL(KIND=8)     :: xl_pos,yl_pos,zl_pos
  INTEGER          :: jldt,kldt,lldt
  INTEGER          :: small

  INTEGER          :: j,k,l

  REAL(KIND=8)     :: ds,div,dtut,dtvt,dtwt,dtct,dtdivt,cc,jkl_control
  REAL(KIND=8)     :: du1,du2,dv1,dv2,dw1,dw2

  small=0
  dt_min_val = g_big
  jkl_control=1.1

!$OMP PARALLEL

!$OMP DO PRIVATE(ds,cc,du1,du2,dv1,dv2,dw1,dw2,div,dtct,dtut,dtvt,dtwt,dtdivt)
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max

        ds=1.0_8/MIN(celldx(j),celldy(k),celldz(l))**2.0_8

        cc=soundspeed(j,k,l)*soundspeed(j,k,l)
        cc=cc+2.0_8*viscosity_a(j,k,l)/density0(j,k,l)

        dtct=ds*cc
        dtct=dtc_safe*1.0_8/MAX(SQRT(dtct),g_small)

        du1=(xvel0(j  ,k  ,l  )+xvel0(j  ,k+1,l  )+xvel0(j  ,k  ,l+1)+xvel0(j  ,k+1,l+1))*xarea(j,k,l)
        du2=(xvel0(j+1,k  ,l  )+xvel0(j+1,k+1,l  )+xvel0(j+1,k  ,l+1)+xvel0(j+1,k+1,l+1))*xarea(j,k,l)

        dtut=dtu_safe*4.0_8*volume(j,k,l  )/MAX(ABS(du1),ABS(du2),1.0e-5_8*volume(j,k,l))

        dv1=(yvel0(j  ,k  ,l  )+yvel0(j+1,k  ,l  )+yvel0(j  ,k  ,l+1)+yvel0(j+1,k  ,l+1))*yarea(j,k,l)
        dv2=(yvel0(j  ,k+1,l  )+yvel0(j+1,k+1,l  )+yvel0(j  ,k+1,l+1)+yvel0(j+1,k+1,l+1))*yarea(j,k,l)

        dtvt=dtv_safe*4.0_8*volume(j,k,l)/MAX(ABS(dv1),ABS(dv2),1.0e-5_8*volume(j,k,l))

        dw1=(zvel0(j  ,k  ,l  )+zvel0(j  ,k+1,l  )+zvel0(j+1,k  ,l  )+zvel0(j+1,k+1,l  ))*zarea(j  ,k  ,l  )
        dw2=(zvel0(j  ,k  ,l+1)+zvel0(j  ,k+1,l+1)+zvel0(j+1,k  ,l+1)+zvel0(j+1,k+1,l+1))*zarea(j  ,k  ,l  )


        dtwt=dtw_safe*4.0_8*volume(j,k,l)/MAX(ABS(dw1),ABS(dw2),1.0e-5_8*volume(j,k,l))

        div=du2-du1+dv2-dv1+dw2-dw1

        dtdivt=dtdiv_safe*4.0_8*(volume(j,k,l))/MAX(volume(j,k,l)*1.0e-05_8,ABS(div))

        dt_min(j,k,l)=MIN(dtct,dtut,dtvt,dtwt,dtdivt)

      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO REDUCTION(MIN : dt_min_val)
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        IF(dt_min(j,k,l).LT.dt_min_val) dt_min_val=dt_min(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP END PARALLEL

  ! Extract the mimimum timestep information
  dtl_control=10.01*(jkl_control-INT(jkl_control))
  jkl_control=jkl_control-(jkl_control-INT(jkl_control))
  jldt=MOD(INT(jkl_control),x_max)
  kldt=1+(jkl_control/x_max)
  lldt=1+(jkl_control/x_max)
  xl_pos=cellx(jldt)
  yl_pos=celly(kldt)
  zl_pos=cellz(lldt)

  IF(dt_min_val.LT.dtmin) small=1

  IF(small.NE.0)THEN
    WRITE(0,*) 'Timestep information:'
    WRITE(0,*) 'j, k                 : ',jldt,kldt
    WRITE(0,*) 'x, y                 : ',cellx(jldt),celly(kldt)
    WRITE(0,*) 'timestep : ',dt_min_val
    WRITE(0,*) 'Cell velocities;'
    WRITE(0,*) xvel0(jldt  ,kldt  ,lldt  ),yvel0(jldt  ,kldt  ,lldt  ),zvel0(jldt  ,kldt  ,lldt  )
    WRITE(0,*) xvel0(jldt+1,kldt  ,lldt  ),yvel0(jldt+1,kldt  ,lldt  ),zvel0(jldt  ,kldt  ,lldt  )
    WRITE(0,*) xvel0(jldt+1,kldt+1,lldt  ),yvel0(jldt+1,kldt+1,lldt  ),zvel0(jldt  ,kldt  ,lldt  )
    WRITE(0,*) xvel0(jldt  ,kldt+1,lldt  ),yvel0(jldt  ,kldt+1,lldt  ),zvel0(jldt  ,kldt  ,lldt  )
    WRITE(0,*) xvel0(jldt  ,kldt  ,lldt+1),yvel0(jldt  ,kldt  ,lldt+1),zvel0(jldt  ,kldt  ,lldt+1)
    WRITE(0,*) xvel0(jldt+1,kldt  ,lldt+1),yvel0(jldt+1,kldt  ,lldt+1),zvel0(jldt  ,kldt  ,lldt+1)
    WRITE(0,*) xvel0(jldt+1,kldt+1,lldt+1),yvel0(jldt+1,kldt+1,lldt+1),zvel0(jldt  ,kldt  ,lldt+1)
    WRITE(0,*) xvel0(jldt  ,kldt+1,lldt+1),yvel0(jldt  ,kldt+1,lldt+1),zvel0(jldt  ,kldt  ,lldt+1)
    WRITE(0,*) 'density, energy, pressure, soundspeed '
    WRITE(0,*) density0(jldt,kldt,lldt),energy0(jldt,kldt,lldt),pressure(jldt,kldt,lldt),soundspeed(jldt,kldt,lldt)
  ENDIF

END SUBROUTINE calc_dt_kernel

END MODULE calc_dt_kernel_module

