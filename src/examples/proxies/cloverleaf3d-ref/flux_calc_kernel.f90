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

!>  @brief Fortran flux kernel.
!>  @author Wayne Gaudin
!>  @details The edge volume fluxes are calculated based on the velocity fields.

MODULE flux_calc_kernel_module

CONTAINS

SUBROUTINE flux_calc_kernel(x_min,x_max,y_min,y_max,z_min,z_max,dt, &
                            xarea,                                  &
                            yarea,                                  &
                            zarea,                                  &
                            xvel0,                                  &
                            yvel0,                                  &
                            zvel0,                                  &
                            xvel1,                                  &
                            yvel1,                                  &
                            zvel1,                                  &
                            vol_flux_x,                             &
                            vol_flux_y,                             &
                            vol_flux_z                              )

  IMPLICIT NONE

  INTEGER       :: x_min, x_max, y_min, y_max, z_min, z_max
  REAL(KIND=8) :: dt
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: xarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: yarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: zarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel0,yvel0,zvel0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel1,yvel1,zvel1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: vol_flux_x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: vol_flux_y
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: vol_flux_z

  INTEGER :: j,k,l

!$OMP PARALLEL

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max+1 
        vol_flux_x(j,k,l)=0.125_8*dt*xarea(j,k,l)                  &
                         *(xvel0(j,k,l)+xvel0(j,k+1,l)+xvel0(j,k,l+1)+xvel0(j,k+1,l+1) &
                          +xvel1(j,k,l)+xvel1(j,k+1,l)+xvel1(j,k,l+1)+xvel1(j,k+1,l+1))
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max+1
      DO j=x_min,x_max
        vol_flux_y(j,k,l)=0.125_8*dt*yarea(j,k,l)                  &
                         *(yvel0(j,k,l)+yvel0(j+1,k,l)+yvel0(j,k,l+1)+yvel0(j+1,k,l+1) &
                          +yvel1(j,k,l)+yvel1(j+1,k,l)+yvel1(j,k,l+1)+yvel1(j+1,k,l+1))
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max+1
    DO k=y_min,y_max
      DO j=x_min,x_max
        vol_flux_z(j,k,l)=0.125_8*dt*zarea(j,k,l)                  &
                         *(zvel0(j,k,l)+zvel0(j+1,k,l)+zvel0(j+1,k,l)+zvel0(j+1,k+1,l) &
                          +zvel1(j,k,l)+zvel1(j+1,k,l)+zvel1(j,k+1,l)+zvel1(j+1,k+1,l))
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP END PARALLEL

END SUBROUTINE flux_calc_kernel

END MODULE flux_calc_kernel_module
