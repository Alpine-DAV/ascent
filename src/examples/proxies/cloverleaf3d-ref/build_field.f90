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

!>  @brief  Allocates the data for each mesh chunk
!>  @author Wayne Gaudin
!>  @details The data fields for the mesh chunk are allocated based on the mesh
!>  size.

MODULE build_field_module

CONTAINS

SUBROUTINE build_field(chunk,x_cells,y_cells,z_cells)

   USE clover_module

   IMPLICIT NONE

   INTEGER :: chunk,x_cells,y_cells,z_cells

   chunks(chunk)%field%x_min=1
   chunks(chunk)%field%y_min=1
   chunks(chunk)%field%z_min=1

   chunks(chunk)%field%x_max=x_cells
   chunks(chunk)%field%y_max=y_cells
   chunks(chunk)%field%z_max=z_cells

   ALLOCATE(chunks(chunk)%field%density0  (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%density1  (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%energy0   (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%energy1   (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%pressure  (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%viscosity (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%soundspeed(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                   chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2,                         &
                   chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))

   ALLOCATE(chunks(chunk)%field%xvel0(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                      chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                      chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%xvel1(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                      chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                      chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%yvel0(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                      chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                      chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%yvel1(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                      chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                      chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%zvel0(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                      chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                      chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%zvel1(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                      chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                      chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))


   ALLOCATE(chunks(chunk)%field%vol_flux_x (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%mass_flux_x(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%vol_flux_y (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%mass_flux_y(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%vol_flux_z (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%mass_flux_z(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))

   ALLOCATE(chunks(chunk)%field%work_array1(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%work_array2(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%work_array3(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%work_array4(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%work_array5(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%work_array6(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%work_array7(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                            chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                            chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))

   ALLOCATE(chunks(chunk)%field%cellx   (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2))
   ALLOCATE(chunks(chunk)%field%celly   (chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2))
   ALLOCATE(chunks(chunk)%field%cellz   (chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%vertexx (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3))
   ALLOCATE(chunks(chunk)%field%vertexy (chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3))
   ALLOCATE(chunks(chunk)%field%vertexz (chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%celldx  (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2))
   ALLOCATE(chunks(chunk)%field%celldy  (chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2))
   ALLOCATE(chunks(chunk)%field%celldz  (chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%vertexdx(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3))
   ALLOCATE(chunks(chunk)%field%vertexdy(chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3))
   ALLOCATE(chunks(chunk)%field%vertexdz(chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))
   ALLOCATE(chunks(chunk)%field%volume  (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                         chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                         chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%xarea   (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+3, &
                                         chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                         chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%yarea   (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                         chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+3, &
                                         chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+2))
   ALLOCATE(chunks(chunk)%field%zarea   (chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2, &
                                         chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2, &
                                         chunks(chunk)%field%z_min-2:chunks(chunk)%field%z_max+3))

   ! Zeroing isn't strictly neccessary but it ensures physical pages
   ! are allocated. This prevents first touch overheads in the main code
   ! cycle which can skew timings in the first step
!$OMP PARALLEL
   chunks(chunk)%field%work_array1=0.0
   chunks(chunk)%field%work_array2=0.0
   chunks(chunk)%field%work_array3=0.0
   chunks(chunk)%field%work_array4=0.0
   chunks(chunk)%field%work_array5=0.0
   chunks(chunk)%field%work_array6=0.0
   chunks(chunk)%field%work_array7=0.0

   chunks(chunk)%field%density0=0.0
   chunks(chunk)%field%density1=0.0
   chunks(chunk)%field%energy0=0.0
   chunks(chunk)%field%energy1=0.0
   chunks(chunk)%field%pressure=0.0
   chunks(chunk)%field%viscosity=0.0
   chunks(chunk)%field%soundspeed=0.0
   
   chunks(chunk)%field%xvel0=0.0
   chunks(chunk)%field%xvel1=0.0
   chunks(chunk)%field%yvel0=0.0
   chunks(chunk)%field%yvel1=0.0
   chunks(chunk)%field%zvel0=0.0
   chunks(chunk)%field%zvel1=0.0
   
   chunks(chunk)%field%vol_flux_x=0.0
   chunks(chunk)%field%mass_flux_x=0.0
   chunks(chunk)%field%vol_flux_y=0.0
   chunks(chunk)%field%mass_flux_y=0.0
   chunks(chunk)%field%vol_flux_z=0.0
   chunks(chunk)%field%mass_flux_z=0.0

   chunks(chunk)%field%cellx=0.0
   chunks(chunk)%field%celly=0.0
   chunks(chunk)%field%cellz=0.0
   chunks(chunk)%field%vertexx=0.0
   chunks(chunk)%field%vertexy=0.0
   chunks(chunk)%field%vertexz=0.0
   chunks(chunk)%field%celldx=0.0
   chunks(chunk)%field%celldy=0.0
   chunks(chunk)%field%vertexdx=0.0
   chunks(chunk)%field%vertexdy=0.0
   chunks(chunk)%field%vertexdz=0.0
   chunks(chunk)%field%volume=0.0
   chunks(chunk)%field%xarea=0.0
   chunks(chunk)%field%yarea=0.0
   chunks(chunk)%field%zarea=0.0
!$OMP END PARALLEL
  
END SUBROUTINE build_field

END MODULE build_field_module
