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

!>  @brief Fortran revert kernel.
!>  @author Wayne Gaudin
!>  @details Takes the half step field data used in the predictor and reverts
!>  it to the start of step data, ready for the corrector.
!>  Note that this does not seem necessary in this proxy-app but should be
!>  left in to remain relevant to the full method.

MODULE revert_kernel_module

CONTAINS

SUBROUTINE revert_kernel(x_min,x_max,y_min,y_max,z_min,z_max,density0,density1,energy0,energy1)

  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)    :: density0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)    :: density1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)    :: energy0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)    :: energy1

  INTEGER :: j,k,l

!$OMP PARALLEL

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        density1(j,k,l)=density0(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        energy1(j,k,l)=energy0(j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP END PARALLEL

END SUBROUTINE revert_kernel

END MODULE revert_kernel_module
