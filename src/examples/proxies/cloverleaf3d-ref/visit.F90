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

!>  @brief Generates graphics output files.
!>  @author Wayne Gaudin
!>  @details The field data over all mesh chunks is written to a .vtk files and
!>  the .visit file is written that defines the time for each set of vtk files.
!>  The ideal gas and viscosity routines are invoked to make sure this data is
!>  up to data with the current energy, density and velocity.

SUBROUTINE visit(my_ascent, sim_time)

  USE clover_module
  USE update_halo_module
  USE viscosity_module
  USE ideal_gas_module

#if defined(USE_MOD)
  use mpi
#endif

  USE iso_c_binding
  USE conduit
  USE conduit_blueprint
  USE ascent

  IMPLICIT NONE

  INTEGER :: j,k,l,c,err,get_unit,u,dummy,jmin,kmin,lmin
  INTEGER :: nxc,nyc,nzc,nxv,nyv,nzv,nblocks
  INTEGER :: gnxc,gnyc,gnzc,gnxv,gnyv,gnzv
  INTEGER :: ghost_flag
  REAL(KIND=8)    :: temp_var

  CHARACTER(len=80)           :: name
  CHARACTER(len=10)           :: chunk_name,step_name
  CHARACTER(len=90)           :: filename

  LOGICAL, SAVE :: first_call=.TRUE.

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer,timerstart

  !
  ! Conduit variables
  !
  CHARACTER(len=80) :: savename

  TYPE(C_PTR) ascent_opts
  TYPE(C_PTR) my_ascent
  TYPE(C_PTR) sim_data
  TYPE(C_PTR) verify_info

  TYPE(C_PTR) sim_actions
  TYPE(C_PTR) add_scene_act
  TYPE(C_PTR) scenes
  TYPE(C_PTR) execute_act
  TYPE(C_PTR) reset_act

  INTEGER(8) :: nnodes, ncells
  REAL(8), ALLOCATABLE :: ghost_flags(:,:,:)
  REAL(8), DIMENSION(1) :: array
  REAL(8) :: sim_time
  INTEGER(C_SIZE_T) :: num_elements

  name = 'clover'
  
  ! skip for vis nodes
  ! IF(parallel%task.LT.parallel%max_task)THEN
  IF(MPI_COMM_NULL.NE.parallel%sim_comm)THEN
    IF(profiler_on) kernel_time=timer()
    DO c=1,chunks_per_task
      CALL ideal_gas(c,.FALSE.)
    ENDDO
    IF(profiler_on) profiler%ideal_gas=profiler%ideal_gas+(timer()-kernel_time)

    fields=0
    fields(FIELD_PRESSURE)=1
    fields(FIELD_XVEL0)=1
    fields(FIELD_YVEL0)=1
    fields(FIELD_ZVEL0)=1
    IF(profiler_on) kernel_time=timer()
    CALL update_halo(fields,1)
    
    IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)
    
    IF(profiler_on) kernel_time=timer()
    CALL viscosity()
    IF(profiler_on) profiler%viscosity=profiler%viscosity+(timer()-kernel_time)
  ENDIF

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! Begin Ascent Integration
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  IF(profiler_on) kernel_time=timer()

  sim_data = conduit_node_create()

  ! vis nodes: send 'empty' data set
  IF(parallel%task.GE.parallel%max_task)THEN     
    CALL conduit_node_set_path_float64(sim_data,"state/time", time)
    CALL conduit_node_set_path_float64(sim_data,"state/sim_time", sim_time*visit_frequency)
    CALL conduit_node_set_path_int32(sim_data,"state/domain_id", parallel%task)
    CALL conduit_node_set_path_int32(sim_data,"state/cycle", step)
    CALL conduit_node_set_path_char8_str(sim_data,"coordsets/coords/type", "rectilinear")
    array(1) = 0.0
    num_elements = 1
    CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/x", array, num_elements)
    CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/y", array, num_elements)
    CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/z", array, num_elements)
    CALL conduit_node_set_path_char8_str(sim_data,"topologies/mesh/type", "rectilinear")
    CALL conduit_node_set_path_char8_str(sim_data,"topologies/mesh/coordset", "coords")

    sim_actions = conduit_node_create()
    add_scene_act = conduit_node_append(sim_actions)
    CALL conduit_node_set_path_char8_str(add_scene_act,"action", "add_scenes")

    scenes = conduit_node_fetch(add_scene_act,"scenes")
    CALL conduit_node_set_path_char8_str(scenes,"s1/plots/p1/type", "volume")
    CALL conduit_node_set_path_char8_str(scenes,"s1/plots/p1/field", "energy")

    CALL ascent_publish(my_ascent, sim_data)
    CALL ascent_execute(my_ascent, sim_actions)

    CALL conduit_node_destroy(sim_actions)
    CALL conduit_node_destroy(sim_data)

  ELSE  ! sim nodes
    DO c = 1, chunks_per_task
      ! skip this stuff for vis nodes
      IF(chunks(c)%task.EQ.parallel%task) THEN
        nxc=chunks(c)%field%x_max-chunks(c)%field%x_min+1
        nyc=chunks(c)%field%y_max-chunks(c)%field%y_min+1
        nzc=chunks(c)%field%z_max-chunks(c)%field%z_min+1
        nxv=nxc+1
        nyv=nyc+1
        nzv=nzc+1

        ! dimensions with ghosts
        gnxc=nxc+4
        gnyc=nyc+4
        gnzc=nzc+4
        gnxv=gnxc+1
        gnyv=gnyc+1
        gnzv=gnzc+1
        ncells = gnxc * gnyc * gnzc
        nnodes = gnxv * gnyv * gnzv

        !
        ! Ascent in situ visualization
        !
        ! CALL ascent_timer_start(C_CHAR_"COPY_DATA"//C_NULL_CHAR)

        ALLOCATE(ghost_flags(0:gnxc-1,0:gnyc-1,0:gnzc-1))
        DO l=0,gnzc-1
          DO k=0, gnyc-1
            DO j=0, gnxc-1
              ghost_flag=0
              IF(l < 2 .OR. l > gnzc - 3) THEN
                ghost_flag = 1
              END IF
              IF(k < 2 .OR. k > gnyc - 3) THEN
                ghost_flag = 1
              END IF
              IF(j < 2 .OR. j > gnxc - 3) THEN
                ghost_flag = 1
              END IF
              ghost_flags(j,k,l)=ghost_flag
            ENDDO
          ENDDO
        ENDDO
        

        CALL conduit_node_set_path_float64(sim_data,"state/time", time)
        CALL conduit_node_set_path_float64(sim_data,"state/sim_time", sim_time*visit_frequency)
        CALL conduit_node_set_path_int32(sim_data,"state/domain_id", parallel%task)
        CALL conduit_node_set_path_int32(sim_data,"state/cycle", step)
        CALL conduit_node_set_path_char8_str(sim_data,"coordsets/coords/type", "rectilinear")

        CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/x", chunks(c)%field%vertexx, gnxv*1_8)
        CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/y", chunks(c)%field%vertexy, gnyv*1_8)
        CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/z", chunks(c)%field%vertexz, gnzv*1_8)
        CALL conduit_node_set_path_char8_str(sim_data,"topologies/mesh/type", "rectilinear")
        CALL conduit_node_set_path_char8_str(sim_data,"topologies/mesh/coordset", "coords")
        ! ghost zone flags
        CALL conduit_node_set_path_char8_str(sim_data,"fields/ascent_ghosts/association", "element")
        CALL conduit_node_set_path_char8_str(sim_data,"fields/ascent_ghosts/topology", "mesh")
        CALL conduit_node_set_path_char8_str(sim_data,"fields/ascent_ghosts/type", "scalar")
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/ascent_ghosts/values", ghost_flags, ncells)
        ! density
        CALL conduit_node_set_path_char8_str(sim_data,"fields/density/association", "element")
        CALL conduit_node_set_path_char8_str(sim_data,"fields/density/topology", "mesh")
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/density/values", chunks(c)%field%density0, ncells)
        ! energy
        CALL conduit_node_set_path_char8_str(sim_data,"fields/energy/association", "element")
        CALL conduit_node_set_path_char8_str(sim_data,"fields/energy/topology", "mesh")
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/energy/values", chunks(c)%field%energy0, ncells)
        ! pressure
        CALL conduit_node_set_path_char8_str(sim_data,"fields/pressure/association", "element")
        CALL conduit_node_set_path_char8_str(sim_data,"fields/pressure/topology", "mesh")
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/pressure/values", chunks(c)%field%pressure, ncells)
        ! velocity x,y,z
        CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity/association", "vertex")
        CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity/topology", "mesh")
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/velocity/values/u", chunks(c)%field%xvel0, nnodes)
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/velocity/values/v", chunks(c)%field%yvel0, nnodes)
        CALL conduit_node_set_path_float64_ptr(sim_data,"fields/velocity/values/w", chunks(c)%field%zvel0, nnodes)
        ! CALL sim_data%print_detailed()
        
        WRITE(chunk_name, '(i6)') parallel%task+100001
        chunk_name(1:1) = "."
        WRITE(step_name, '(i6)') step+100000
        step_name(1:1) = "."
        savename = trim(trim(name) //trim(chunk_name)//trim(step_name))
        
        ! CALL ascent_timer_stop(C_CHAR_"COPY_DATA"//C_NULL_CHAR)
        
        sim_actions = conduit_node_create()
        add_scene_act = conduit_node_append(sim_actions)
        CALL conduit_node_set_path_char8_str(add_scene_act,"action", "add_scenes")
        
        scenes = conduit_node_fetch(add_scene_act,"scenes")
        CALL conduit_node_set_path_char8_str(scenes,"s1/plots/p1/type", "volume")
        CALL conduit_node_set_path_char8_str(scenes,"s1/plots/p1/field", "energy")
        
        CALL ascent_publish(my_ascent, sim_data)   
        CALL ascent_execute(my_ascent, sim_actions)
        
        CALL conduit_node_destroy(sim_actions)
        CALL conduit_node_destroy(sim_data)
        
        DEALLOCATE(ghost_flags)

      ENDIF

      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! End Ascent Integration
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    IF( .FALSE. ) THEN
        WRITE(chunk_name, '(i6)') parallel%task+100001
        chunk_name(1:1) = "."
        WRITE(step_name, '(i6)') step+100000
        step_name(1:1) = "."
        filename = trim(trim(name) //trim(chunk_name)//trim(step_name))//".vtk"
        u=get_unit(dummy)
        OPEN(UNIT=u,FILE=filename,STATUS='UNKNOWN',IOSTAT=err)
        WRITE(u,'(a)')'# vtk DataFile Version 3.0'
        WRITE(u,'(a)')'vtk output'
        WRITE(u,'(a)')'ASCII'
        WRITE(u,'(a)')'DATASET RECTILINEAR_GRID'
        WRITE(u,'(a,3i12)')'DIMENSIONS',nxv,nyv,nzv
        WRITE(u,'(a,i5,a)')'X_COORDINATES ',nxv,' double'
        DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
          WRITE(u,'(e12.4)')chunks(c)%field%vertexx(j)
        ENDDO
        WRITE(u,'(a,i5,a)')'Y_COORDINATES ',nyv,' double'
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
          WRITE(u,'(e12.4)')chunks(c)%field%vertexy(k)
        ENDDO
        WRITE(u,'(a,i5,a)')'Z_COORDINATES ',nzv,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
          WRITE(u,'(e12.4)')chunks(c)%field%vertexz(l)
        ENDDO
        WRITE(u,'(a,i20)')'CELL_DATA ',nxc*nyc*nzc
        WRITE(u,'(a)')'FIELD FieldData 4'
        WRITE(u,'(a,i20,a)')'density 1 ',nxc*nyc*nzc,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
            WRITE(u,'(e12.4)')(chunks(c)%field%density0(j,k,l),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
          ENDDO
        ENDDO
        WRITE(u,'(a,i20,a)')'energy 1 ',nxc*nyc*nzc,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
            WRITE(u,'(e12.4)')(chunks(c)%field%energy0(j,k,l),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
          ENDDO
        ENDDO
        WRITE(u,'(a,i20,a)')'pressure 1 ',nxc*nyc*nzc,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
            WRITE(u,'(e12.4)')(chunks(c)%field%pressure(j,k,l),j=chunks(c)%field%x_min,chunks(c)%field%x_max)
          ENDDO
        ENDDO
        WRITE(u,'(a,i20,a)')'viscosity 1 ',nxc*nyc*nzc,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
            DO j=chunks(c)%field%x_min,chunks(c)%field%x_max
              temp_var=0.0
              IF(chunks(c)%field%viscosity(j,k,l).GT.0.00000001) temp_var=chunks(c)%field%viscosity(j,k,l)
              WRITE(u,'(e12.4)') temp_var
            ENDDO
          ENDDO
        ENDDO
        WRITE(u,'(a,i20)')'POINT_DATA ',nxv*nyv*nzv
        WRITE(u,'(a)')'FIELD FieldData 3'
        WRITE(u,'(a,i20,a)')'x_vel 1 ',nxv*nyv*nzv,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
            DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
              temp_var=0.0
              IF(ABS(chunks(c)%field%xvel0(j,k,l)).GT.0.00000001) temp_var=chunks(c)%field%xvel0(j,k,l)
              WRITE(u,'(e12.4)') temp_var
            ENDDO
          ENDDO
        ENDDO
        WRITE(u,'(a,i20,a)')'y_vel 1 ',nxv*nyv*nzv,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
            DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
              temp_var=0.0
              IF(ABS(chunks(c)%field%yvel0(j,k,l)).GT.0.00000001) temp_var=chunks(c)%field%yvel0(j,k,l)
              WRITE(u,'(e12.4)') temp_var
            ENDDO
          ENDDO
        ENDDO
        WRITE(u,'(a,i20,a)')'z_vel 1 ',nxv*nyv*nzv,' double'
        DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
          DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
            DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
              temp_var=0.0
              IF(ABS(chunks(c)%field%zvel0(j,k,l)).GT.0.00000001) temp_var=chunks(c)%field%zvel0(j,k,l)
              WRITE(u,'(e12.4)') temp_var
            ENDDO
          ENDDO
        ENDDO
        CLOSE(u)
      ENDIF

    ENDDO

  ENDIF ! sim nodes

  IF(profiler_on) profiler%visit=profiler%visit+(timer()-kernel_time)

END SUBROUTINE visit
