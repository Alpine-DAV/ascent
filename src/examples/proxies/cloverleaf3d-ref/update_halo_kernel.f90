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

!>  @brief Fortran kernel to update the external halo cells in a chunk.
!>  @author Wayne Gaudin
!>  @details Updates halo cells for the required fields at the required depth
!>  for any halo cells that lie on an external boundary. The location and type
!>  of data governs how this is carried out. External boundaries are always
!>  reflective.

! Notes
! More fields to add and corrections to be made

MODULE update_halo_kernel_module

CONTAINS

  SUBROUTINE update_halo_kernel(x_min,x_max,y_min,y_max,z_min,z_max,                &
                        chunk_neighbours,                                           &
                        density0,                                                   &
                        energy0,                                                    &
                        pressure,                                                   &
                        viscosity,                                                  &
                        soundspeed,                                                 &
                        density1,                                                   &
                        energy1,                                                    &
                        xvel0,                                                      &
                        yvel0,                                                      &
                        zvel0,                                                      &
                        xvel1,                                                      &
                        yvel1,                                                      &
                        zvel1,                                                      &
                        vol_flux_x,                                                 &
                        vol_flux_y,                                                 &
                        vol_flux_z,                                                 &
                        mass_flux_x,                                                &
                        mass_flux_y,                                                &
                        mass_flux_z,                                                &
                        fields,                                                     &
                        depth                                                       )
  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER, DIMENSION(6) :: chunk_neighbours
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density0,energy0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: pressure,viscosity,soundspeed
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density1,energy1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel0,yvel0,zvel0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel1,yvel1,zvel1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: vol_flux_x,mass_flux_x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: vol_flux_y,mass_flux_y
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: vol_flux_z,mass_flux_z
  INTEGER :: fields(:),depth

  ! These need to be kept consistent with the data module to avoid use statement
  INTEGER,      PARAMETER :: CHUNK_LEFT   =1    &
                            ,CHUNK_RIGHT  =2    &
                            ,CHUNK_BOTTOM =3    &
                            ,CHUNK_TOP    =4    &
                            ,CHUNK_BACK   =5    &
                            ,CHUNK_FRONT  =6    &
                            ,EXTERNAL_FACE=-1

  INTEGER,      PARAMETER :: FIELD_DENSITY0   = 1         &
                            ,FIELD_DENSITY1   = 2         &
                            ,FIELD_ENERGY0    = 3         &
                            ,FIELD_ENERGY1    = 4         &
                            ,FIELD_PRESSURE   = 5         &
                            ,FIELD_VISCOSITY  = 6         &
                            ,FIELD_SOUNDSPEED = 7         &
                            ,FIELD_XVEL0      = 8         &
                            ,FIELD_XVEL1      = 9         &
                            ,FIELD_YVEL0      =10         &
                            ,FIELD_YVEL1      =11         &
                            ,FIELD_ZVEL0      =12         &
                            ,FIELD_ZVEL1      =13         &
                            ,FIELD_VOL_FLUX_X =14         &
                            ,FIELD_VOL_FLUX_Y =15         &
                            ,FIELD_VOL_FLUX_Z =16         &
                            ,FIELD_MASS_FLUX_X=17         &
                            ,FIELD_MASS_FLUX_Y=18         &
                            ,FIELD_MASS_FLUX_Z=19

  INTEGER :: j,k,l

!$OMP PARALLEL PRIVATE(j,k)

  ! Update values in external halo cells based on depth and fields requested
  ! Even though half of these loops look the wrong way around, it should be noted
  ! that depth is either 1 or 2 so that it is more efficient to always thread
  ! loop along the mesh edge.
  IF(fields(FIELD_DENSITY0).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            density0(j,1-k,l)=density0(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            density0(j,y_max+k,l)=density0(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            density0(1-j,k,l)=density0(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            density0(x_max+j,k,l)=density0(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            density0(j,k,1-l)=density0(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            density0(j,k,z_max+l)=density0(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_DENSITY1).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            density1(j,1-k,l)=density1(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            density1(j,y_max+k,l)=density1(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            density1(1-j,k,l)=density1(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            density1(x_max+j,k,l)=density1(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            density1(j,k,1-l)=density1(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            density1(j,k,z_max+l)=density1(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_ENERGY0).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            energy0(j,1-k,l)=energy0(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            energy0(j,y_max+k,l)=energy0(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            energy0(1-j,k,l)=energy0(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            energy0(x_max+j,k,l)=energy0(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            energy0(j,k,1-l)=energy0(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            energy0(j,k,z_max+l)=energy0(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_ENERGY1).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            energy1(j,1-k,l)=energy1(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            energy1(j,y_max+k,l)=energy1(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            energy1(1-j,k,l)=energy1(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            energy1(x_max+j,k,l)=energy1(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            energy1(j,k,1-l)=energy1(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            energy1(j,k,z_max+l)=energy1(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_PRESSURE).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            pressure(j,1-k,l)=pressure(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            pressure(j,y_max+k,l)=pressure(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            pressure(1-j,k,l)=pressure(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            pressure(x_max+j,k,l)=pressure(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            pressure(j,k,1-l)=pressure(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            pressure(j,k,z_max+l)=pressure(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_VISCOSITY).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            viscosity(j,1-k,l)=viscosity(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            viscosity(j,y_max+k,l)=viscosity(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            viscosity(1-j,k,l)=viscosity(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            viscosity(x_max+j,k,l)=viscosity(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            viscosity(j,k,1-l)=viscosity(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            viscosity(j,k,z_max+l)=viscosity(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_SOUNDSPEED).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            soundspeed(j,1-k,l)=soundspeed(j,0+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            soundspeed(j,y_max+k,l)=soundspeed(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            soundspeed(1-j,k,l)=soundspeed(0+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            soundspeed(x_max+j,k,l)=soundspeed(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            soundspeed(j,k,1-l)=soundspeed(j,k,0+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            soundspeed(j,k,z_max+l)=soundspeed(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_XVEL0).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            xvel0(j,1-k,l)=xvel0(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            xvel0(j,y_max+1+k,l)=xvel0(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            xvel0(1-j,k,l)=-xvel0(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            xvel0(x_max+1+j,k,l)=-xvel0(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            xvel0(j,k,1-l)=xvel0(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            xvel0(j,k,z_max+1+l)=xvel0(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_XVEL1).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            xvel1(j,1-k,l)=xvel1(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            xvel1(j,y_max+1+k,l)=xvel1(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            xvel1(1-j,k,l)=-xvel1(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            xvel1(x_max+1+j,k,l)=-xvel1(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            xvel1(j,k,1-l)=xvel1(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            xvel1(j,k,z_max+1+l)=xvel1(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_YVEL0).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            yvel0(j,1-k,l)=-yvel0(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            yvel0(j,y_max+1+k,l)=-yvel0(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            yvel0(1-j,k,l)=yvel0(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            yvel0(x_max+1+j,k,l)=yvel0(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            yvel0(j,k,1-l)=yvel0(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            yvel0(j,k,z_max+1+l)=yvel0(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_YVEL1).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            yvel1(j,1-k,l)=-yvel1(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            yvel1(j,y_max+1+k,l)=-yvel1(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            yvel1(1-j,k,l)=yvel1(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            yvel1(x_max+1+j,k,l)=yvel1(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            yvel1(j,k,1-l)=yvel1(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            yvel1(j,k,z_max+1+l)=yvel1(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_ZVEL0).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            zvel0(j,1-k,l)=zvel0(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            zvel0(j,y_max+1+k,l)=zvel0(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            zvel0(1-j,k,l)=zvel0(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            zvel0(x_max+1+j,k,l)=zvel0(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            zvel0(j,k,1-l)=-zvel0(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            zvel0(j,k,z_max+1+l)=-zvel0(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_ZVEL1).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            zvel1(j,1-k,l)=zvel1(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            zvel1(j,y_max+1+k,l)=zvel1(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            zvel1(1-j,k,l)=zvel1(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            zvel1(x_max+1+j,k,l)=zvel1(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            zvel1(j,k,1-l)=-zvel1(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+1+depth
        DO j=x_min-depth,x_max+1+depth
          DO l=1,depth
            zvel1(j,k,z_max+1+l)=-zvel1(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_VOL_FLUX_X).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            vol_flux_x(j,1-k,l)=vol_flux_x(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            vol_flux_x(j,y_max+k,l)=vol_flux_x(j,y_max-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            vol_flux_x(1-j,k,l)=-vol_flux_x(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            vol_flux_x(x_max+j+1,k,l)=-vol_flux_x(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            vol_flux_x(j,k,1-l)=vol_flux_x(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            vol_flux_x(j,k,z_max+l)=vol_flux_x(j,k,z_max-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_MASS_FLUX_X).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            mass_flux_x(j,1-k,l)=mass_flux_x(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+1+depth
          DO k=1,depth
            mass_flux_x(j,y_max+k,l)=mass_flux_x(j,y_max-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            mass_flux_x(1-j,k,l)=-mass_flux_x(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            mass_flux_x(x_max+j+1,k,l)=-mass_flux_x(x_max+1-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            mass_flux_x(j,k,1-l)=mass_flux_x(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            mass_flux_x(j,k,z_max+l)=mass_flux_x(j,k,z_max-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_VOL_FLUX_Y).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            vol_flux_y(j,1-k,l)=-vol_flux_y(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            vol_flux_y(j,y_max+k+1,l)=-vol_flux_y(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            vol_flux_y(1-j,k,l)=vol_flux_y(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            vol_flux_y(x_max+j,k,l)=vol_flux_y(x_max-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            vol_flux_y(j,k,1-l)=vol_flux_y(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            vol_flux_y(j,k,z_max+l)=vol_flux_y(j,k,z_max-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_MASS_FLUX_Y).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            mass_flux_y(j,1-k,l)=-mass_flux_y(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            mass_flux_y(j,y_max+k+1,l)=-mass_flux_y(j,y_max+1-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            mass_flux_y(1-j,k,l)=mass_flux_y(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO k=y_min-depth,y_max+1+depth
          DO j=1,depth
            mass_flux_y(x_max+j,k,l)=mass_flux_y(x_max-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            mass_flux_y(j,k,1-l)=mass_flux_y(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            mass_flux_y(j,k,z_max+l)=mass_flux_y(j,k,z_max-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

  IF(fields(FIELD_VOL_FLUX_Z).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            vol_flux_z(j,1-k,l)=vol_flux_z(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            vol_flux_z(j,y_max+k,l)=vol_flux_z(j,y_max-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            vol_flux_z(1-j,k,l)=vol_flux_z(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            vol_flux_z(x_max+j,k,l)=vol_flux_z(x_max-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            vol_flux_z(j,k,1-l)=-vol_flux_z(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            vol_flux_z(j,k,z_max+l+1)=-vol_flux_z(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF
  IF(fields(FIELD_MASS_FLUX_Z).EQ.1) THEN
    IF(chunk_neighbours(CHUNK_BOTTOM).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            mass_flux_z(j,1-k,l)=mass_flux_z(j,1+k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_TOP).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO j=x_min-depth,x_max+depth
          DO k=1,depth
            mass_flux_z(j,y_max+k,l)=mass_flux_z(j,y_max-k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_LEFT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            mass_flux_z(1-j,k,l)=mass_flux_z(1+j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_RIGHT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO l=z_min-depth,z_max+1+depth
        DO k=y_min-depth,y_max+depth
          DO j=1,depth
            mass_flux_z(x_max+j,k,l)=mass_flux_z(x_max-j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_BACK).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            mass_flux_z(j,k,1-l)=-mass_flux_z(j,k,1+l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
    IF(chunk_neighbours(CHUNK_FRONT).EQ.EXTERNAL_FACE) THEN
!$OMP DO
      DO k=y_min-depth,y_max+depth
        DO j=x_min-depth,x_max+depth
          DO l=1,depth
            mass_flux_z(j,k,z_max+1+l)=-mass_flux_z(j,k,z_max+1-l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
  ENDIF

!$OMP END PARALLEL

END SUBROUTINE update_halo_kernel

END  MODULE update_halo_kernel_module
