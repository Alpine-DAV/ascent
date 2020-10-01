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

!>  @brief Reads the user input
!>  @author Wayne Gaudin
!>  @details Reads and parses the user input from the processed file and sets
!>  the variables used in the generation phase. Default values are also set
!>  here.

SUBROUTINE read_input()

  USE clover_module
  USE parse_module
  USE report_module

  IMPLICIT NONE

  INTEGER            :: state,stat,state_max,n

  REAL(KIND=8) :: dx,dy,dz

  CHARACTER(LEN=500) :: word

  test_problem=0
  
  state_max=0
  
  grid%xmin=  0.0_8
  grid%ymin=  0.0_8
  grid%zmin=  0.0_8
  grid%xmax=100.0_8
  grid%ymax=100.0_8
  grid%zmax=100.0_8
  
  grid%x_cells=10
  grid%y_cells=10
  grid%z_cells=10
  
  end_time=10.0_8
  end_step=g_ibig
  complete=.FALSE.
  
  visit_frequency=0
  summary_frequency=10
  
  visit_sim_time=0.0_8
  dtinit=0.1_8
  dtmax=1.0_8
  dtmin=0.0000001_8
  dtrise=1.05_8
  dtc_safe=0.75_8
  dtu_safe=0.5_8
  dtv_safe=0.5_8
  dtw_safe=0.5_8
  dtdiv_safe=0.5_8
  
  use_fortran_kernels=.TRUE.
  sim_nodes_only=.FALSE.
  profiler_on=.FALSE.
  profiler%timestep=0.0
  profiler%acceleration=0.0
  profiler%PdV=0.0
  profiler%cell_advection=0.0
  profiler%mom_advection=0.0
  profiler%viscosity=0.0
  profiler%ideal_gas=0.0
  profiler%visit=0.0
  profiler%summary=0.0
  profiler%reset=0.0
  profiler%revert=0.0
  profiler%flux=0.0
  profiler%halo_exchange=0.0

  IF(parallel%boss)WRITE(g_out,*) 'Reading input file'
  IF(parallel%boss)WRITE(g_out,*)

  stat=parse_init(g_in,'*clover')

  DO
    stat=parse_getline(dummy)
    IF (stat.ne.0) exit
    DO
      word=parse_getword(.FALSE.)
      IF(word.EQ.'')EXIT
      IF (word.EQ.'state') THEN
        state_max=MAX(state_max,parse_getival(parse_getword(.TRUE.)))
        EXIT
      ENDIF
    ENDDO
  ENDDO

  number_of_states=state_max

  IF(number_of_states.LT.1) CALL report_error('read_input','No states defined.')

  stat=parse_init(g_in,'*clover')

  ALLOCATE(states(number_of_states))
  states(:)%defined=.FALSE.
  states(:)%energy=0.0
  states(:)%density=0.0
  states(:)%xvel=0.0
  states(:)%yvel=0.0
  states(:)%zvel=0.0

  DO
    stat=parse_getline(dummy)

    IF(stat.NE.0)EXIT

    DO
      word=parse_getword(.FALSE.)

      IF(word.EQ.'')EXIT
      SELECT CASE(word)
      CASE('initial_timestep')
        dtinit=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'initial_timestep ',dtinit
      CASE('max_timestep')
        dtmax=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'max_timestep',dtinit
      CASE('timestep_rise')
        dtrise=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'timestep_rise',dtrise
      CASE('end_time')
        end_time=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'end_time',end_time
      CASE('end_step')
        end_step=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'end_step',end_step
      CASE('xmin')
        grid%xmin=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'xmin',grid%xmin
      CASE('xmax')
        grid%xmax=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'xmax',grid%xmax
      CASE('ymin')
        grid%ymin=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'ymin',grid%ymin
      CASE('ymax')
        grid%ymax=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'ymax',grid%ymax
      CASE('zmin')
        grid%zmin=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'zmin',grid%zmin
      CASE('zmax')
        grid%zmax=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'zmax',grid%zmax
      CASE('x_cells')
        grid%x_cells=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'x_cells',grid%x_cells
      CASE('y_cells')
        grid%y_cells=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'y_cells',grid%y_cells
      CASE('z_cells')
        grid%z_cells=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'z_cells',grid%z_cells
      CASE('visit_frequency')
        visit_frequency=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'visit_frequency',visit_frequency
      CASE('visit_sim_time')
        visit_sim_time=parse_getrval(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'visit_sim_time',visit_sim_time
      CASE('summary_frequency')
        summary_frequency=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'summary_frequency',summary_frequency
      CASE('use_fortran_kernels')
        use_fortran_kernels=.TRUE.
      CASE('profiler_on')
        profiler_on=.TRUE.
        IF(parallel%boss)WRITE(g_out,"(1x,a25)")'Profiler on'
      CASE('test_problem')
        test_problem=parse_getival(parse_getword(.TRUE.))
        IF(parallel%boss)WRITE(g_out,"(1x,a25,i12)")'test_problem',test_problem
      CASE('sim_nodes_only')
        sim_nodes_only=.TRUE.
      CASE('state')

        state=parse_getival(parse_getword(.TRUE.))

        IF(parallel%boss)WRITE(g_out,*)'Reading specification for state ',state
        IF (states(state)%defined) CALL report_error('read_input','State defined twice.')
        IF(parallel%boss) WRITE(g_out,*)

        states(state)%defined=.TRUE.
        DO
          word=parse_getword(.FALSE.)
          IF(word.EQ.'') EXIT

          SELECT CASE(word)

          CASE('xvel')
            states(state)%xvel=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'xvel ',states(state)%xvel
          CASE('yvel')
            states(state)%yvel=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'yvel ',states(state)%yvel
          CASE('zvel')
            states(state)%zvel=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'zvel ',states(state)%zvel
          CASE('xmin')
            states(state)%xmin=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state xmin ',states(state)%xmin
          CASE('ymin')
            states(state)%ymin=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state ymin ',states(state)%ymin
          CASE('zmin')
            states(state)%zmin=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state zmin ',states(state)%zmin
          CASE('xmax')
            states(state)%xmax=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state xmax ',states(state)%xmax
          CASE('ymax')
            states(state)%ymax=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state ymax ',states(state)%ymax
          CASE('zmax')
            states(state)%zmax=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state zmax ',states(state)%zmax
          CASE('radius')
            states(state)%radius=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state radius ',states(state)%radius
          CASE('density')
            states(state)%density=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state density ',states(state)%density
          CASE('energy')
            states(state)%energy=parse_getrval(parse_getword(.TRUE.))
            IF(parallel%boss)WRITE(g_out,"(1x,a25,e12.4)")'state energy ',states(state)%energy
          CASE('geometry')
            word=TRIM(parse_getword(.TRUE.))
            SELECT CASE(word)
            CASE("cuboid")
              states(state)%geometry=g_rect
              IF(parallel%boss)WRITE(g_out,"(1x,a26)")'state geometry cuboid'
            CASE("sphere")
              states(state)%geometry=g_circ
              IF(parallel%boss)WRITE(g_out,"(1x,a25)")'state geometry sphere'
            CASE("point")
              states(state)%geometry=g_point
              IF(parallel%boss)WRITE(g_out,"(1x,a25)")'state geometry point'
            END SELECT
          END SELECT
        ENDDO
        IF(parallel%boss) WRITE(g_out,*)
      END SELECT
    ENDDO
  ENDDO

  IF(parallel%boss) THEN
    WRITE(g_out,*)
    IF(use_fortran_kernels) THEN
      WRITE(g_out,"(1x,a25)")'Using Fortran Kernels'
    ENDIF
    WRITE(g_out,*)
    WRITE(g_out,*) 'Input read finished.'
    WRITE(g_out,*)
  ENDIF

  ! If a state boundary falls exactly on a cell boundary then round off can
  ! cause the state to be put one cell further that expected. This is compiler
  ! /system dependent. To avoid this, a state boundary is reduced/increased by a 100th
  ! of a cell width so it lies well with in the intended cell.
  ! Because a cell is either full or empty of a specified state, this small
  ! modification to the state extents does not change the answers.
  dx=(grid%xmax-grid%xmin)/float(grid%x_cells)
  dy=(grid%ymax-grid%ymin)/float(grid%y_cells)
  dz=(grid%zmax-grid%zmin)/float(grid%z_cells)
  DO n=2,number_of_states
    states(n)%xmin=states(n)%xmin+(dx/100.0_8)
    states(n)%ymin=states(n)%ymin+(dy/100.0_8)
    states(n)%zmin=states(n)%zmin+(dz/100.0_8)
    states(n)%xmax=states(n)%xmax-(dx/100.0_8)
    states(n)%ymax=states(n)%ymax-(dy/100.0_8)
    states(n)%zmax=states(n)%zmax-(dz/100.0_8)
  ENDDO

END SUBROUTINE read_input
