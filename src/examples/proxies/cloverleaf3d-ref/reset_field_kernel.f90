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

!>  @brief Fortran reset field kernel.
!>  @author Wayne Gaudin
!>  @details Copies all of the final end of step filed data to the begining of
!>  step data, ready for the next timestep.

MODULE reset_field_kernel_module

CONTAINS

SUBROUTINE reset_field_kernel(x_min,x_max,y_min,y_max,z_min,z_max,    &
                              density0,           &
                              density1,           &
                              energy0,            &
                              energy1,            &
                              xvel0,              &
                              xvel1,              &
                              yvel0,              &
                              yvel1,              &
                              zvel0,              &
                              zvel1)

  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density0,energy0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density1,energy1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel0,yvel0,zvel0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel1,yvel1,zvel1

  INTEGER :: j,k,l

!$OMP PARALLEL
!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
       DO j=x_min,x_max
         density0(j,k,l)=density1(j,k,l)
       ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
       DO j=x_min,x_max
         energy0(j,k,l)=energy1(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max+1
    DO k=y_min,y_max+1
      DO j=x_min,x_max+1
        xvel0(j,k,l)=xvel1(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max+1
    DO k=y_min,y_max+1
      DO j=x_min,x_max+1
        yvel0(j,k,l)=yvel1(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max+1
    DO k=y_min,y_max+1
      DO j=x_min,x_max+1
        zvel0(j,k,l)=zvel1(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE reset_field_kernel

END MODULE reset_field_kernel_module
