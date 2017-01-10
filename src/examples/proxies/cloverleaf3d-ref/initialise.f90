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

!>  @brief Top level initialisation routine
!>  @author Wayne Gaudin
!>  @details Checks for the user input and either invokes the input reader or
!>  switches to the internal test problem. It processes the input and strips
!>  comments before writing a final input file.
!>  It then calls the start routine.

SUBROUTINE initialise

  USE clover_module
  USE parse_module
  USE report_module

  IMPLICIT NONE

  INTEGER :: ios
  INTEGER :: get_unit,stat,uin,out_unit
!$ INTEGER :: OMP_GET_THREAD_NUM,OMP_GET_NUM_THREADS
  CHARACTER(LEN=g_len_max) :: ltmp

  IF(parallel%boss)THEN
    g_out=get_unit(dummy)

    OPEN(FILE='clover.out',ACTION='WRITE',UNIT=g_out,IOSTAT=ios)
    IF(ios.NE.0) CALL report_error('initialise','Error opening clover.out file.')

  ELSE
    g_out=6
  ENDIF

!$OMP PARALLEL
  IF(parallel%boss)THEN
!$  IF(OMP_GET_THREAD_NUM().EQ.0) THEN
      WRITE(g_out,*)
      WRITE(g_out,'(a15,f8.3)') 'Clover Version ',g_version
      WRITE(g_out,'(a18)') 'MPI Version'
!$    WRITE(g_out,'(a18)') 'OpenMP Version'
      WRITE(g_out,'(a14,i6)') 'Task Count ',parallel%max_task !MPI
!$    WRITE(g_out,'(a15,i5)') 'Thread Count: ',OMP_GET_NUM_THREADS()
      WRITE(g_out,*)
      WRITE(*,*)'Output file clover.out opened. All output will go there.'
!$  ENDIF
  ENDIF
!$OMP END PARALLEL

  CALL clover_barrier

  IF(parallel%boss)THEN
    WRITE(g_out,*) 'Clover will run from the following input:-'
    WRITE(g_out,*)
  ENDIF

  IF(parallel%boss)THEN
    uin=get_unit(dummy)

    OPEN(FILE='clover.in',ACTION='READ',STATUS='OLD',UNIT=uin,IOSTAT=ios)
    IF(ios.NE.0) THEN
      out_unit=get_unit(dummy)
      OPEN(FILE='clover.in',UNIT=out_unit,STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      WRITE(out_unit,'(A)')'*clover'
      WRITE(out_unit,'(A)')' state 1 density=0.2 energy=1.0'
      WRITE(out_unit,'(A)')' state 2 density=1.0 energy=2.5 geometry=cuboid xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0 zmin=0.0 zmax=2.0'
      WRITE(out_unit,'(A)')' x_cells=10'
      WRITE(out_unit,'(A)')' y_cells=2'
      WRITE(out_unit,'(A)')' z_cells=2'
      WRITE(out_unit,'(A)')' xmin=0.0'
      WRITE(out_unit,'(A)')' ymin=0.0'
      WRITE(out_unit,'(A)')' zmin=0.0'
      WRITE(out_unit,'(A)')' xmax=10.0'
      WRITE(out_unit,'(A)')' ymax=2.0'
      WRITE(out_unit,'(A)')' zmax=2.0'
      WRITE(out_unit,'(A)')' initial_timestep=0.04'
      WRITE(out_unit,'(A)')' max_timestep=0.04'
      WRITE(out_unit,'(A)')' end_step=75'
      WRITE(out_unit,'(A)')' test_problem 1'
      WRITE(out_unit,'(A)')'*endclover'
      CLOSE(out_unit)
      uin=get_unit(dummy)
      OPEN(FILE='clover.in',ACTION='READ',STATUS='OLD',UNIT=uin,IOSTAT=ios)
    ENDIF

    out_unit=get_unit(dummy)
    OPEN(FILE='clover.in.tmp',UNIT=out_unit,STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
    IF(ios.NE.0) CALL  report_error('initialise','Error opening clover.in.tmp file')
    stat=parse_init(uin,'')
    DO
       stat=parse_getline(-1_4)
       IF(stat.NE.0)EXIT
       WRITE(out_unit,'(A)') line
    ENDDO
    CLOSE(out_unit)
  ENDIF

  CALL clover_barrier

  g_in=get_unit(dummy)
  OPEN(FILE='clover.in.tmp',ACTION='READ',STATUS='OLD',UNIT=g_in,IOSTAT=ios)

  IF(ios.NE.0) CALL report_error('initialise','Error opening clover.in.tmp file')

  CALL clover_barrier

  IF(parallel%boss)THEN
     REWIND(uin)
     DO 
        READ(UNIT=uin,IOSTAT=ios,FMT='(a150)') ltmp ! Read in next line.
        IF(ios.NE.0)EXIT
        WRITE(g_out,FMT='(a150)') ltmp
     ENDDO
  ENDIF

  IF(parallel%boss)THEN
     WRITE(g_out,*)
     WRITE(g_out,*) 'Initialising and generating'
     WRITE(g_out,*)
  ENDIF

  CALL read_input()

  CALL clover_barrier

  step=0

  CALL start

  CALL clover_barrier

  IF(parallel%boss)THEN
     WRITE(g_out,*) 'Starting the calculation'
  ENDIF

  CLOSE(g_in)

END SUBROUTINE initialise

FUNCTION get_unit(dummy)
  INTEGER :: get_unit,dummy

  INTEGER :: u
  LOGICAL :: used

  DO u=7,99
     INQUIRE(UNIT=u,OPENED=used)
     IF(.NOT.used)THEN
        EXIT
     ENDIF
  ENDDO

  get_unit=u

END FUNCTION get_unit
