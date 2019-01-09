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

!>  @brief Fortran PdV kernel.
!>  @author Wayne Gaudin
!>  @details Calculates the change in energy and density in a cell using the
!>  change on cell volume due to the velocity gradients in a cell. The time
!>  level of the velocity data depends on whether it is invoked as the
!>  predictor or corrector.

! Notes
! Again, fluxes need updating for 3d

MODULE PdV_kernel_module

CONTAINS

SUBROUTINE PdV_kernel(predict,                                          &
                      x_min,x_max,y_min,y_max,z_min,z_max,dt,           &
                      xarea,yarea,zarea,volume,                         &
                      density0,                                         &
                      density1,                                         &
                      energy0,                                          &
                      energy1,                                          &
                      pressure,                                         &
                      viscosity,                                        &
                      xvel0,                                            &
                      xvel1,                                            &
                      yvel0,                                            &
                      yvel1,                                            &
                      zvel0,                                            &
                      zvel1,                                            &
                      volume_change                                     )

  IMPLICIT NONE

  LOGICAL :: predict

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  REAL(KIND=8)  :: dt
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: xarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: yarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: zarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: volume
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density0,energy0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: pressure
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density1,energy1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: viscosity
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel0,yvel0,zvel0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel1,yvel1,zvel1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: volume_change

  INTEGER :: j,k,l

  REAL(KIND=8)  :: recip_volume,energy_change,min_cell_volume
  REAL(KIND=8)  :: right_flux,left_flux,top_flux,bottom_flux,back_flux,front_flux,total_flux

!$OMP PARALLEL

  IF(predict)THEN

!$OMP DO PRIVATE(right_flux,left_flux,top_flux,bottom_flux,back_flux,front_flux,total_flux,min_cell_volume, &
!$OMP            energy_change,recip_volume)
    DO l=z_min,z_max
      DO k=y_min,y_max
        DO j=x_min,x_max

          left_flux=  (xarea(j  ,k  ,l  )*(xvel0(j  ,k  ,l  )+xvel0(j  ,k+1,l  )+xvel0(j  ,k  ,l+1)+xvel0(j  ,k+1,l+1)   &
                                          +xvel0(j  ,k  ,l  )+xvel0(j  ,k+1,l  )+xvel0(j  ,k  ,l+1)+xvel0(j  ,k+1,l+1))) &
                      *0.125_8*dt*0.5
          right_flux= (xarea(j+1,k  ,l  )*(xvel0(j+1,k  ,l  )+xvel0(j+1,k+1,l  )+xvel0(j+1,k  ,l+1)+xvel0(j+1,k+1,l+1)   &
                                          +xvel0(j+1,k  ,l  )+xvel0(j+1,k+1,l  )+xvel0(j+1,k  ,l+1)+xvel0(j+1,k+1,l+1))) &
                      *0.125_8*dt*0.5
          bottom_flux=(yarea(j  ,k  ,l  )*(yvel0(j  ,k  ,l  )+yvel0(j+1,k  ,l  )+yvel0(j  ,k  ,l+1)+yvel0(j+1,k  ,l+1)   &
                                          +yvel0(j  ,k  ,l  )+yvel0(j+1,k  ,l  )+yvel0(j  ,k  ,l+1)+yvel0(j+1,k  ,l+1))) &
                      *0.125_8*dt*0.5
          top_flux=   (yarea(j  ,k+1,l  )*(yvel0(j  ,k+1,l  )+yvel0(j+1,k+1,l  )+yvel0(j  ,k+1,l+1)+yvel0(j+1,k+1,l+1)   &
                                          +yvel0(j  ,k+1,l  )+yvel0(j+1,k+1,l  )+yvel0(j  ,k+1,l+1)+yvel0(j+1,k+1,l+1))) &
                      *0.125_8*dt*0.5
          back_flux=  (zarea(j  ,k  ,l  )*(zvel0(j  ,k  ,l  )+zvel0(j+1,k  ,l  )+zvel0(j  ,k+1,l  )+zvel0(j+1,k+1,l  )   &
                                          +zvel0(j  ,k  ,l  )+zvel0(j+1,k  ,l  )+zvel0(j  ,k+1,l  )+zvel0(j+1,k+1,l  ))) &
                      *0.125_8*dt*0.5
          front_flux= (zarea(j  ,k  ,l+1)*(zvel0(j  ,k  ,l+1)+zvel0(j+1,k  ,l+1)+zvel0(j  ,k+1,l+1)+zvel0(j+1,k+1,l+1)   &
                                          +zvel0(j  ,k  ,l+1)+zvel0(j+1,k  ,l+1)+zvel0(j  ,k+1,l+1)+zvel0(j+1,k+1,l+1))) &
                      *0.125_8*dt*0.5
          total_flux=right_flux-left_flux+top_flux-bottom_flux+front_flux-back_flux

          volume_change(j,k,l)=volume(j,k,l)/(volume(j,k,l)+total_flux)

          min_cell_volume=MIN(volume(j,k,l)+right_flux-left_flux+top_flux-bottom_flux+front_flux-back_flux  &
                             ,volume(j,k,l)+right_flux-left_flux+top_flux-bottom_flux                       &
                             ,volume(j,k,l)+right_flux-left_flux                                            &
                             ,volume(j,k,l)+top_flux-bottom_flux)

          recip_volume=1.0/volume(j,k,l)

          energy_change=(pressure(j,k,l)/density0(j,k,l)+viscosity(j,k,l)/density0(j,k,l))*total_flux*recip_volume

          energy1(j,k,l)=energy0(j,k,l)-energy_change

          density1(j,k,l)=density0(j,k,l)*volume_change(j,k,l)

        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

  ELSE

!$OMP DO PRIVATE(right_flux,left_flux,top_flux,bottom_flux,back_flux,front_flux,total_flux,min_cell_volume, &
!$OMP            energy_change,recip_volume)
    DO l=z_min,z_max
      DO k=y_min,y_max
        DO j=x_min,x_max

          left_flux=  (xarea(j  ,k  ,l  )*(xvel0(j  ,k  ,l  )+xvel0(j  ,k+1,l  )+xvel0(j  ,k  ,l+1)+xvel0(j  ,k+1,l+1)   &
                                          +xvel1(j  ,k  ,l  )+xvel1(j  ,k+1,l  )+xvel1(j  ,k  ,l+1)+xvel1(j  ,k+1,l+1))) &
                      *0.125_8*dt
          right_flux= (xarea(j+1,k  ,l  )*(xvel0(j+1,k  ,l  )+xvel0(j+1,k+1,l  )+xvel0(j+1,k  ,l+1)+xvel0(j+1,k+1,l+1)   &
                                          +xvel1(j+1,k  ,l  )+xvel1(j+1,k+1,l  )+xvel1(j+1,k  ,l+1)+xvel1(j+1,k+1,l+1))) &
                      *0.125_8*dt
          bottom_flux=(yarea(j  ,k  ,l  )*(yvel0(j  ,k  ,l  )+yvel0(j+1,k  ,l  )+yvel0(j  ,k  ,l+1)+yvel0(j+1,k  ,l+1)   &
                                          +yvel1(j  ,k  ,l  )+yvel1(j+1,k  ,l  )+yvel1(j  ,k  ,l+1)+yvel1(j+1,k  ,l+1))) &
                      *0.125_8*dt
          top_flux=   (yarea(j  ,k+1,l  )*(yvel0(j  ,k+1,l  )+yvel0(j+1,k+1,l  )+yvel0(j  ,k+1,l+1)+yvel0(j+1,k+1,l+1)   &
                                          +yvel1(j  ,k+1,l  )+yvel1(j+1,k+1,l  )+yvel1(j  ,k+1,l+1)+yvel1(j+1,k+1,l+1))) &
                      *0.125_8*dt
          back_flux=  (zarea(j  ,k  ,l  )*(zvel0(j  ,k  ,l  )+zvel0(j+1,k  ,l  )+zvel0(j  ,k+1,l  )+zvel0(j+1,k+1,l  )   &
                                          +zvel1(j  ,k  ,l  )+zvel1(j+1,k  ,l  )+zvel1(j  ,k+1,l  )+zvel1(j+1,k+1,l  ))) &
                      *0.125_8*dt
          front_flux= (zarea(j  ,k  ,l+1)*(zvel0(j  ,k  ,l+1)+zvel0(j+1,k  ,l+1)+zvel0(j  ,k+1,l+1)+zvel0(j+1,k+1,l+1)   &
                                          +zvel1(j  ,k  ,l+1)+zvel1(j+1,k  ,l+1)+zvel1(j  ,k+1,l+1)+zvel1(j+1,k+1,l+1))) &
                      *0.125_8*dt
          total_flux=right_flux-left_flux+top_flux-bottom_flux+front_flux-back_flux

          volume_change(j,k,l)=volume(j,k,l)/(volume(j,k,l)+total_flux)

          min_cell_volume=MIN(volume(j,k,l)+right_flux-left_flux+top_flux-bottom_flux+front_flux-back_flux  &
                             ,volume(j,k,l)+right_flux-left_flux+top_flux-bottom_flux                       &
                             ,volume(j,k,l)+right_flux-left_flux                                            &
                             ,volume(j,k,l)+top_flux-bottom_flux)

          recip_volume=1.0/volume(j,k,l)

          energy_change=(pressure(j,k,l)/density0(j,k,l)+viscosity(j,k,l)/density0(j,k,l))*total_flux*recip_volume

          energy1(j,k,l)=energy0(j,k,l)-energy_change

          density1(j,k,l)=density0(j,k,l)*volume_change(j,k,l)

        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

  ENDIF

!$OMP END PARALLEL

END SUBROUTINE PdV_kernel

END MODULE PdV_kernel_module

