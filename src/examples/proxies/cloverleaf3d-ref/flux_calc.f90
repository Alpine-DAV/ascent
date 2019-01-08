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

!>  @brief Driver for the flux kernels
!>  @author Wayne Gaudin
!>  @details Invokes the used specified flux kernel

MODULE flux_calc_module

CONTAINS

SUBROUTINE flux_calc()

  USE clover_module
  USE flux_calc_kernel_module

  IMPLICIT NONE

  INTEGER :: c

  REAL(KIND=8) :: kernel_time,timer

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task

    IF(chunks(c)%task.EQ.parallel%task) THEN

      IF(use_fortran_kernels)THEN
        CALL flux_calc_kernel(chunks(c)%field%x_min,         &
                            chunks(c)%field%x_max,           &
                            chunks(c)%field%y_min,           &
                            chunks(c)%field%y_max,           &
                            chunks(c)%field%z_min,           &
                            chunks(c)%field%z_max,           &
                            dt,                              &
                            chunks(c)%field%xarea,           &
                            chunks(c)%field%yarea,           &
                            chunks(c)%field%zarea,           &
                            chunks(c)%field%xvel0,           &
                            chunks(c)%field%yvel0,           &
                            chunks(c)%field%zvel0,           &
                            chunks(c)%field%xvel1,           &
                            chunks(c)%field%yvel1,           &
                            chunks(c)%field%zvel1,           &
                            chunks(c)%field%vol_flux_x,      &
                            chunks(c)%field%vol_flux_y,      &
                            chunks(c)%field%vol_flux_z       )
      ENDIF

    ENDIF

  ENDDO
  IF(profiler_on) profiler%flux=profiler%flux+(timer()-kernel_time)

END SUBROUTINE flux_calc

END MODULE flux_calc_module
