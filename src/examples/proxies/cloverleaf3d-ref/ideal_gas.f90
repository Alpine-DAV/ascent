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

!>  @brief Ideal gas kernel driver
!>  @author Wayne Gaudin
!>  @details Invokes the user specified kernel for the ideal gas equation of
!>  state using the specified time level data.

MODULE ideal_gas_module

CONTAINS

SUBROUTINE ideal_gas(chunk,predict)

  USE clover_module
  USE ideal_gas_kernel_module

  IMPLICIT NONE

  INTEGER :: chunk

  LOGICAl :: predict

  IF(chunks(chunk)%task .EQ. parallel%task) THEN

    IF(.NOT.predict) THEN
      IF(use_fortran_kernels)THEN
        CALL ideal_gas_kernel(chunks(chunk)%field%x_min,    &
                            chunks(chunk)%field%x_max,      &
                            chunks(chunk)%field%y_min,      &
                            chunks(chunk)%field%y_max,      &
                            chunks(chunk)%field%z_min,      &
                            chunks(chunk)%field%z_max,      &
                            chunks(chunk)%field%density0,   &
                            chunks(chunk)%field%energy0,    &
                            chunks(chunk)%field%pressure,   &
                            chunks(chunk)%field%soundspeed  )
      ENDIF
    ELSE
      IF(use_fortran_kernels)THEN
        CALL ideal_gas_kernel(chunks(chunk)%field%x_min,    &
                            chunks(chunk)%field%x_max,      &
                            chunks(chunk)%field%y_min,      &
                            chunks(chunk)%field%y_max,      &
                            chunks(chunk)%field%z_min,      &
                            chunks(chunk)%field%z_max,      &
                            chunks(chunk)%field%density1,   &
                            chunks(chunk)%field%energy1,    &
                            chunks(chunk)%field%pressure,   &
                            chunks(chunk)%field%soundspeed  )
      ENDIF
    ENDIF

  ENDIF

END SUBROUTINE ideal_gas

END MODULE ideal_gas_module
