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

!>  @brief Mesh chunk generation driver
!>  @author Wayne Gaudin
!>  @details Invoked the users specified chunk generator.

SUBROUTINE generate_chunk(chunk)

  USE clover_module
  USE generate_chunk_kernel_module

  IMPLICIT NONE

  INTEGER         :: chunk

  INTEGER         :: state
  REAL(KIND=8), DIMENSION(number_of_states) :: state_density,state_energy,state_xvel,state_yvel,state_zvel
  REAL(KIND=8), DIMENSION(number_of_states) :: state_xmin,state_xmax,state_ymin,state_ymax,state_zmin,state_zmax,state_radius
  INTEGER,      DIMENSION(number_of_states) :: state_geometry

  DO state=1,number_of_states 
   state_density(state)=states(state)%density
   state_energy(state)=states(state)%energy
   state_xvel(state)=states(state)%xvel
   state_yvel(state)=states(state)%yvel
   state_zvel(state)=states(state)%zvel
   state_xmin(state)=states(state)%xmin
   state_xmax(state)=states(state)%xmax
   state_ymin(state)=states(state)%ymin
   state_ymax(state)=states(state)%ymax
   state_zmin(state)=states(state)%zmin
   state_zmax(state)=states(state)%zmax
   state_radius(state)=states(state)%radius
   state_geometry(state)=states(state)%geometry
  ENDDO

  IF(use_fortran_kernels) THEN
    CALL generate_chunk_kernel(chunks(chunk)%field%x_min,             &
                               chunks(chunk)%field%x_max,             &
                               chunks(chunk)%field%y_min,             &
                               chunks(chunk)%field%y_max,             &
                               chunks(chunk)%field%z_min,             &
                               chunks(chunk)%field%z_max,             &
                               chunks(chunk)%field%vertexx,           &
                               chunks(chunk)%field%vertexy,           &
                               chunks(chunk)%field%vertexz,           &
                               chunks(chunk)%field%cellx,             &
                               chunks(chunk)%field%celly,             &
                               chunks(chunk)%field%cellz,             &
                               chunks(chunk)%field%density0,          &
                               chunks(chunk)%field%energy0,           &
                               chunks(chunk)%field%xvel0,             &
                               chunks(chunk)%field%yvel0,             &
                               chunks(chunk)%field%zvel0,             &
                               number_of_states,                      &
                               state_density,                         &
                               state_energy,                          &
                               state_xvel,                            &
                               state_yvel,                            &
                               state_zvel,                            &
                               state_xmin,                            &
                               state_xmax,                            &
                               state_ymin,                            &
                               state_ymax,                            &
                               state_zmin,                            &
                               state_zmax,                            &
                               state_radius,                          &
                               state_geometry,                        &
                               g_rect,                                &
                               g_circ,                                &
                               g_point)
      ENDIF

END SUBROUTINE generate_chunk
