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

!>  @brief Driver for the PdV update.
!>  @author Wayne Gaudin
!>  @details Invokes the user specified kernel for the PdV update.

MODULE PdV_module

CONTAINS

SUBROUTINE PdV(predict)

  USE clover_module
  USE report_module
  USE PdV_kernel_module
  USE revert_module
  USE update_halo_module
  USE ideal_gas_module

  IMPLICIT NONE

  LOGICAL :: predict

  INTEGER :: prdct

  INTEGER :: c
  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

  error_condition=0 ! Not used yet due to issue with OpenA reduction

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task

    IF(chunks(c)%task.EQ.parallel%task) THEN

      IF(use_fortran_kernels)THEN
        CALL PdV_kernel(predict,                  &
                      chunks(c)%field%x_min,      &
                      chunks(c)%field%x_max,      &
                      chunks(c)%field%y_min,      &
                      chunks(c)%field%y_max,      &
                      chunks(c)%field%z_min,      &
                      chunks(c)%field%z_max,      &
                      dt,                         &
                      chunks(c)%field%xarea,      &
                      chunks(c)%field%yarea,      &
                      chunks(c)%field%zarea,      &
                      chunks(c)%field%volume ,    &
                      chunks(c)%field%density0,   &
                      chunks(c)%field%density1,   &
                      chunks(c)%field%energy0,    &
                      chunks(c)%field%energy1,    &
                      chunks(c)%field%pressure,   &
                      chunks(c)%field%viscosity,  &
                      chunks(c)%field%xvel0,      &
                      chunks(c)%field%xvel1,      &
                      chunks(c)%field%yvel0,      &
                      chunks(c)%field%yvel1,      &
                      chunks(c)%field%zvel0,      &
                      chunks(c)%field%zvel1,      &
                      chunks(c)%field%work_array1 )
      ENDIF
    ENDIF

  ENDDO

  CALL clover_check_error(error_condition)
  IF(profiler_on) profiler%PdV=profiler%PdV+(timer()-kernel_time)

  IF(error_condition.EQ.1) THEN
    CALL report_error('PdV','error in PdV')
  ENDIF

  IF(predict)THEN
    IF(profiler_on) kernel_time=timer()
    DO c=1,chunks_per_task
      CALL ideal_gas(c,.TRUE.)
    ENDDO
    IF(profiler_on) profiler%ideal_gas=profiler%ideal_gas+(timer()-kernel_time)
    fields=0
    fields(FIELD_PRESSURE)=1
    IF(profiler_on) kernel_time=timer()
    CALL update_halo(fields,1)
    IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)
  ENDIF

  IF ( predict ) THEN
    IF(profiler_on) kernel_time=timer()
    CALL revert()
    IF(profiler_on) profiler%revert=profiler%revert+(timer()-kernel_time)
  ENDIF

END SUBROUTINE PdV

END MODULE PdV_module
