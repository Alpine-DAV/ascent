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

!>  @brief Top level advection driver
!>  @author Wayne Gaudin
!>  @details Controls the advection step and invokes required communications.

MODULE advection_module

CONTAINS

SUBROUTINE advection()

  USE clover_module
  USE advec_cell_driver_module
  USE advec_mom_driver_module
  USE update_halo_module

  IMPLICIT NONE

  INTEGER :: sweep_number,direction,c

  INTEGER :: xvel,yvel,zvel

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

  sweep_number=1
  IF(advect_x)      direction=g_xdir
  IF(.not.advect_x) direction=g_zdir
  xvel=g_xdir
  yvel=g_ydir
  zvel=g_zdir

  fields=0
  fields(FIELD_ENERGY1)=1
  fields(FIELD_DENSITY1)=1
  fields(FIELD_VOL_FLUX_X)=1
  fields(FIELD_VOL_FLUX_Y)=1
  fields(FIELD_VOL_FLUX_Z)=1
  IF(profiler_on) kernel_time=timer()
  CALL update_halo(fields,2)
  IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL advec_cell_driver(c,sweep_number,direction)
  ENDDO
  IF(profiler_on) profiler%cell_advection=profiler%cell_advection+(timer()-kernel_time)

  fields=0
  fields(FIELD_DENSITY1)=1
  fields(FIELD_ENERGY1)=1
  fields(FIELD_XVEL1)=1
  fields(FIELD_YVEL1)=1
  fields(FIELD_ZVEL1)=1
  fields(FIELD_MASS_FLUX_X)=1
  fields(FIELD_MASS_FLUX_Y)=1
  fields(FIELD_MASS_FLUX_Z)=1
  IF(profiler_on) kernel_time=timer()
  CALL update_halo(fields,2)
  IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,xvel,direction,sweep_number)
  ENDDO
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,yvel,direction,sweep_number)
  ENDDO
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,zvel,direction,sweep_number)
  ENDDO
  IF(profiler_on) profiler%mom_advection=profiler%mom_advection+(timer()-kernel_time)

  sweep_number=2
  direction=g_ydir

  fields=0
  fields(FIELD_ENERGY1)=1
  fields(FIELD_DENSITY1)=1
  fields(FIELD_VOL_FLUX_X)=1
  fields(FIELD_VOL_FLUX_Y)=1
  fields(FIELD_VOL_FLUX_Z)=1
  IF(profiler_on) kernel_time=timer()
  CALL update_halo(fields,2)
  IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL advec_cell_driver(c,sweep_number,direction)
  ENDDO
  IF(profiler_on) profiler%cell_advection=profiler%cell_advection+(timer()-kernel_time)

  fields=0
  fields(FIELD_DENSITY1)=1
  fields(FIELD_ENERGY1)=1
  fields(FIELD_XVEL1)=1
  fields(FIELD_YVEL1)=1
  fields(FIELD_ZVEL1)=1
  fields(FIELD_MASS_FLUX_X)=1
  fields(FIELD_MASS_FLUX_Y)=1
  fields(FIELD_MASS_FLUX_Z)=1
  IF(profiler_on) kernel_time=timer()
  CALL update_halo(fields,2)
  IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,xvel,direction,sweep_number)
  ENDDO
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,yvel,direction,sweep_number)
  ENDDO
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,zvel,direction,sweep_number)
  ENDDO
  IF(profiler_on) profiler%mom_advection=profiler%mom_advection+(timer()-kernel_time)

  sweep_number=3
  IF(advect_x)      direction=g_zdir
  IF(.not.advect_x) direction=g_xdir

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL advec_cell_driver(c,sweep_number,direction)
  ENDDO
  IF(profiler_on) profiler%cell_advection=profiler%cell_advection+(timer()-kernel_time)

  fields=0
  fields(FIELD_DENSITY1)=1
  fields(FIELD_ENERGY1)=1
  fields(FIELD_XVEL1)=1
  fields(FIELD_YVEL1)=1
  fields(FIELD_ZVEL1)=1
  fields(FIELD_MASS_FLUX_X)=1
  fields(FIELD_MASS_FLUX_Y)=1
  fields(FIELD_MASS_FLUX_Z)=1
  IF(profiler_on) kernel_time=timer()
  CALL update_halo(fields,2)
  IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

  IF(profiler_on) kernel_time=timer()
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,xvel,direction,sweep_number)
  ENDDO
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,yvel,direction,sweep_number)
  ENDDO
  DO c=1,chunks_per_task
    CALL advec_mom_driver(c,zvel,direction,sweep_number)
  ENDDO
  IF(profiler_on) profiler%mom_advection=profiler%mom_advection+(timer()-kernel_time)

END SUBROUTINE advection

END MODULE advection_module
