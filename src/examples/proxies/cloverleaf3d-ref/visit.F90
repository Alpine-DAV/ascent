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

SUBROUTINE visit(my_ascent)

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
  REAL(KIND=8)    :: temp_var

  CHARACTER(len=80)           :: name
  CHARACTER(len=10)           :: chunk_name,step_name
  CHARACTER(len=90)           :: filename

  LOGICAL, SAVE :: first_call=.TRUE.

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

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
  REAL(8), ALLOCATABLE :: xcoords(:), ycoords(:), zcoords(:)
  REAL(8), ALLOCATABLE :: density(:,:,:), energy(:,:,:), pressure(:,:,:)
  REAL(8), ALLOCATABLE :: xvel(:,:,:), yvel(:,:,:), zvel(:,:,:)

  name = 'clover'
  IF ( parallel%boss ) THEN
    IF(first_call) THEN

      nblocks=number_of_chunks
      filename = "clover.visit"
      u=get_unit(dummy)
      OPEN(UNIT=u,FILE=filename,STATUS='UNKNOWN',IOSTAT=err)
      WRITE(u,'(a,i5)')'!NBLOCKS ',nblocks
      CLOSE(u)

      first_call=.FALSE.

    ENDIF
  ENDIF

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
  IF ( parallel%boss ) THEN

    filename = "clover.visit"
    u=get_unit(dummy)
    OPEN(UNIT=u,FILE=filename,STATUS='UNKNOWN',POSITION='APPEND',IOSTAT=err)

    DO c = 1, number_of_chunks
      WRITE(chunk_name, '(i6)') c+100000
      chunk_name(1:1) = "."
      WRITE(step_name, '(i6)') step+100000
      step_name(1:1) = "."
      filename = trim(trim(name) //trim(chunk_name)//trim(step_name))//".vtk"
      WRITE(u,'(a)')TRIM(filename)
    ENDDO
    CLOSE(u)

  ENDIF
  
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! Begin Ascent Integration
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  IF(profiler_on) kernel_time=timer()
  DO c = 1, chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task) THEN
      nxc=chunks(c)%field%x_max-chunks(c)%field%x_min+1
      nyc=chunks(c)%field%y_max-chunks(c)%field%y_min+1
      nzc=chunks(c)%field%z_max-chunks(c)%field%z_min+1
      nxv=nxc+1
      nyv=nyc+1
      nzv=nzc+1
      ncells = nxc * nyc * nzc
      nnodes = nxv * nyv * nzv

      !
      ! Ascent in situ visualization
      !
      CALL ascent_timer_start(C_CHAR_"COPY_DATA"//C_NULL_CHAR)
      ALLOCATE(xcoords(0:nxv-1), ycoords(0:nyv-1), zcoords(0:nzv-1))
      jmin=chunks(c)%field%x_min
      DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
        xcoords(j-jmin)=chunks(c)%field%vertexx(j)
      ENDDO
      kmin=chunks(c)%field%y_min
      DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
        ycoords(k-kmin)=chunks(c)%field%vertexy(k)
      ENDDO
      lmin=chunks(c)%field%z_min
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
        zcoords(l-lmin)=chunks(c)%field%vertexz(l)
      ENDDO

      ALLOCATE(density(0:nxc-1,0:nyc-1,0:nzc-1))
      ALLOCATE(energy(0:nxc-1,0:nyc-1,0:nzc-1))
      ALLOCATE(pressure(0:nxc-1,0:nyc-1,0:nzc-1))
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
          DO j=chunks(c)%field%x_min,chunks(c)%field%x_max
            density(j-jmin,k-kmin,l-lmin)=chunks(c)%field%density0(j,k,l)
          ENDDO
        ENDDO
      ENDDO
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
          DO j=chunks(c)%field%x_min,chunks(c)%field%x_max
            energy(j-jmin,k-kmin,l-lmin)=chunks(c)%field%energy0(j,k,l)
          ENDDO
        ENDDO
      ENDDO
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max
          DO j=chunks(c)%field%x_min,chunks(c)%field%x_max
            pressure(j-jmin,k-kmin,l-lmin)=chunks(c)%field%pressure(j,k,l)
          ENDDO
        ENDDO
      ENDDO

      ALLOCATE(xvel(0:nxv-1,0:nyv-1,0:nzv-1))
      ALLOCATE(yvel(0:nxv-1,0:nyv-1,0:nzv-1))
      ALLOCATE(zvel(0:nxv-1,0:nyv-1,0:nzv-1))
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
          DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
            xvel(j-jmin,k-kmin,l-lmin)=chunks(c)%field%xvel0(j,k,l)
          ENDDO
        ENDDO
      ENDDO
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
          DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
            yvel(j-jmin,k-kmin,l-lmin)=chunks(c)%field%yvel0(j,k,l)
          ENDDO
        ENDDO
      ENDDO
      DO l=chunks(c)%field%z_min,chunks(c)%field%z_max+1
        DO k=chunks(c)%field%y_min,chunks(c)%field%y_max+1
          DO j=chunks(c)%field%x_min,chunks(c)%field%x_max+1
            zvel(j-jmin,k-kmin,l-lmin)=chunks(c)%field%zvel0(j,k,l)
          ENDDO
        ENDDO
      ENDDO


      sim_data = conduit_node_create()
      CALL conduit_node_set_path_float64(sim_data,"state/time", time)
      CALL conduit_node_set_path_int32(sim_data,"state/domain", parallel%task)
      CALL conduit_node_set_path_int32(sim_data,"state/cycle", step)
      CALL conduit_node_set_path_char8_str(sim_data,"coordsets/coords/type", "rectilinear")
      CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/x", xcoords, nxv*1_8)
      CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/y", ycoords, nyv*1_8)
      CALL conduit_node_set_path_float64_ptr(sim_data,"coordsets/coords/values/z", zcoords, nzv*1_8)
      CALL conduit_node_set_path_char8_str(sim_data,"topologies/mesh/type", "rectilinear")
      CALL conduit_node_set_path_char8_str(sim_data,"topologies/mesh/coordset", "coords")
      ! density 
      CALL conduit_node_set_path_char8_str(sim_data,"fields/density/association", "element")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/density/topology", "mesh")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/density/type", "scalar")
      CALL conduit_node_set_path_float64_ptr(sim_data,"fields/density/values", density, ncells)
      ! energy
      CALL conduit_node_set_path_char8_str(sim_data,"fields/energy/association", "element")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/energy/topology", "mesh")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/energy/type", "scalar")
      CALL conduit_node_set_path_float64_ptr(sim_data,"fields/energy/values", energy, ncells)
      ! pressure
      CALL conduit_node_set_path_char8_str(sim_data,"fields/pressure/association", "element")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/pressure/topology", "mesh")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/pressure/type", "scalar")
      CALL conduit_node_set_path_float64_ptr(sim_data,"fields/pressure/values", pressure, ncells)
      ! velocity x 
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_x/association", "vertex")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_x/topology", "mesh")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_x/type", "scalar")
      CALL conduit_node_set_path_float64_ptr(sim_data,"fields/velocity_x/values", xvel, nnodes)
      ! velocity y
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_y/association", "vertex")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_y/topology", "mesh")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_y/type", "scalar")
      CALL conduit_node_set_path_float64_ptr(sim_data,"fields/velocity_y/values", yvel, nnodes)
      ! velocity z
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_z/association", "vertex")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_z/topology", "mesh")
      CALL conduit_node_set_path_char8_str(sim_data,"fields/velocity_z/type", "scalar")
      CALL conduit_node_set_path_float64_ptr(sim_data,"fields/velocity_z/values", zvel, nnodes)
      ! CALL sim_data%print_detailed()

      WRITE(chunk_name, '(i6)') parallel%task+100001
      chunk_name(1:1) = "."
      WRITE(step_name, '(i6)') step+100000
      step_name(1:1) = "."
      savename = trim(trim(name) //trim(chunk_name)//trim(step_name))

      sim_actions = conduit_node_create()
      add_scene_act = conduit_node_append(sim_actions)
      CALL conduit_node_set_path_char8_str(add_scene_act,"action", "add_scenes")

      scenes = conduit_node_fetch(add_scene_act,"scenes")
      CALL conduit_node_set_path_char8_str(scenes,"s1/plots/p1/type", "volume")      
      CALL conduit_node_set_path_char8_str(scenes,"s1/plots/p1/params/field", "velocity_y")

      execute_act = conduit_node_append(sim_actions)
      CALL conduit_node_set_path_char8_str(execute_act,"action", "execute")

      reset_act = conduit_node_append(sim_actions)
      CALL conduit_node_set_path_char8_str(reset_act,"action", "reset")

      ! ---- old actions -- 
!       add_plot = conduit_node_append(sim_actions)
!       CALL conduit_node_set_path_char8_str(add_plot,"action", "add_plot")
!       !CALL conduit_node_set_path_char8_str(add_plot,"field_name", "pressure")
!       CALL conduit_node_set_path_char8_str(add_plot,"field_name", "velocity_y")
!       CALL conduit_node_set_path_char8_str(add_plot,"render_options/file_name", savename)
!       CALL conduit_node_set_path_char8_str(add_plot,"render_options/renderer","volume")
!       CALL conduit_node_set_path_int32(add_plot,"render_options/width", 1024)
!       CALL conduit_node_set_path_int32(add_plot,"render_options/height", 1024)
!       draw_plots = conduit_node_append(sim_actions)
!       CALL conduit_node_set_path_char8_str(draw_plots,"action", "draw_plots")

      ! CALL sim_actions%print_detailed()

      CALL ascent_timer_stop(C_CHAR_"COPY_DATA"//C_NULL_CHAR)
      CALL ascent_publish(my_ascent, sim_data)
      CALL ascent_execute(my_ascent, sim_actions)
      CALL conduit_node_destroy(sim_actions)
      CALL conduit_node_destroy(sim_data)

      DEALLOCATE(xvel, yvel, zvel)
      DEALLOCATE(density, energy, pressure)
      DEALLOCATE(xcoords, ycoords, zcoords)
      
      
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
ENDIF
  ENDDO

  IF(profiler_on) profiler%visit=profiler%visit+(timer()-kernel_time)

END SUBROUTINE visit
