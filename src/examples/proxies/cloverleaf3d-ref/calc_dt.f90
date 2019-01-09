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

!>  @brief Driver for the timestep kernels
!>  @author Wayne Gaudin
!>  @details Invokes the user specified timestep kernel.

MODULE calc_dt_module

CONTAINS

SUBROUTINE calc_dt(chunk,local_dt,local_control,xl_pos,yl_pos,zl_pos,jldt,kldt,lldt)

  USE clover_module
  USE calc_dt_kernel_module

  IMPLICIT NONE

  INTEGER          :: chunk
  REAL(KIND=8)     :: local_dt
  CHARACTER(LEN=8) :: local_control
  REAL(KIND=8)     :: xl_pos,yl_pos,zl_pos
  INTEGER          :: jldt,kldt,lldt

  INTEGER          :: l_control
  INTEGER          :: small

  local_dt=g_big

  IF(chunks(chunk)%task.NE.parallel%task) RETURN

  small = 0

  IF(use_fortran_kernels)THEN

    CALL calc_dt_kernel(chunks(chunk)%field%x_min,     &
                        chunks(chunk)%field%x_max,     &
                        chunks(chunk)%field%y_min,     &
                        chunks(chunk)%field%y_max,     &
                        chunks(chunk)%field%z_min,     &
                        chunks(chunk)%field%z_max,     &
                        g_small,                       &
                        g_big,                         &
                        dtmin,                         &
                        dtc_safe,                      &
                        dtu_safe,                      &
                        dtv_safe,                      &
                        dtw_safe,                      &
                        dtdiv_safe,                    &
                        chunks(chunk)%field%xarea,     &
                        chunks(chunk)%field%yarea,     &
                        chunks(chunk)%field%zarea,     &
                        chunks(chunk)%field%cellx,     &
                        chunks(chunk)%field%celly,     &
                        chunks(chunk)%field%cellz,     &
                        chunks(chunk)%field%celldx,    &
                        chunks(chunk)%field%celldy,    &
                        chunks(chunk)%field%celldz,    &
                        chunks(chunk)%field%volume,    &
                        chunks(chunk)%field%density0,  &
                        chunks(chunk)%field%energy0,   &
                        chunks(chunk)%field%pressure,  &
                        chunks(chunk)%field%viscosity, &
                        chunks(chunk)%field%soundspeed,&
                        chunks(chunk)%field%xvel0,     &
                        chunks(chunk)%field%yvel0,     &
                        chunks(chunk)%field%zvel0,     &
                        chunks(chunk)%field%work_array1,&
                        local_dt,                      &
                        l_control,                     &
                        xl_pos,                        &
                        yl_pos,                        &
                        zl_pos,                        &
                        jldt,                          &
                        kldt,                          &
                        lldt,                          &
                        small                          )

  ENDIF

  IF(l_control.EQ.1) local_control='sound'
  IF(l_control.EQ.2) local_control='xvel'
  IF(l_control.EQ.3) local_control='yvel'
  IF(l_control.EQ.4) local_control='zvel'
  IF(l_control.EQ.5) local_control='div'

END SUBROUTINE calc_dt

END MODULE calc_dt_module
