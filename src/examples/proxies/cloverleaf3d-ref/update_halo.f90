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

!>  @brief Driver for the halo updates
!>  @author Wayne Gaudin
!>  @details Invokes the kernels for the internal and external halo cells for
!>  the fields specified.

MODULE update_halo_module

CONTAINS

SUBROUTINE update_halo(fields,depth)

  USE clover_module
  USE update_halo_kernel_module

  IMPLICIT NONE

  INTEGER :: c,fields(NUM_FIELDS),depth

  CALL clover_exchange(fields,depth)

  DO c=1,chunks_per_task
    
    IF(chunks(c)%task.EQ.parallel%task) THEN

      IF(use_fortran_kernels)THEN
        CALL update_halo_kernel(chunks(c)%field%x_min,          &
                                chunks(c)%field%x_max,          &
                                chunks(c)%field%y_min,          &
                                chunks(c)%field%y_max,          &
                                chunks(c)%field%z_min,          &
                                chunks(c)%field%z_max,          &
                                chunks(c)%chunk_neighbours,     &
                                chunks(c)%field%density0,       &
                                chunks(c)%field%energy0,        &
                                chunks(c)%field%pressure,       &
                                chunks(c)%field%viscosity,      &
                                chunks(c)%field%soundspeed,     &
                                chunks(c)%field%density1,       &
                                chunks(c)%field%energy1,        &
                                chunks(c)%field%xvel0,          &
                                chunks(c)%field%yvel0,          &
                                chunks(c)%field%zvel0,          &
                                chunks(c)%field%xvel1,          &
                                chunks(c)%field%yvel1,          &
                                chunks(c)%field%zvel1,          &
                                chunks(c)%field%vol_flux_x,     &
                                chunks(c)%field%vol_flux_y,     &
                                chunks(c)%field%vol_flux_z,     &
                                chunks(c)%field%mass_flux_x,    &
                                chunks(c)%field%mass_flux_y,    &
                                chunks(c)%field%mass_flux_z,    &
                                fields,                         &
                                depth                           )
      ENDIF
    ENDIF

  ENDDO

END SUBROUTINE update_halo

END MODULE update_halo_module
