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

!>  @brief Fortran cell advection kernel.
!>  @author Wayne Gaudin
!>  @details Performs a second order advective remap using van-Leer limiting
!>  with directional splitting.

! Notes
! All the sweep numbers need to be update and intermediate volumes correctly calculated

MODULE advec_cell_kernel_module

CONTAINS

SUBROUTINE advec_cell_kernel(x_min,       &
                             x_max,       &
                             y_min,       &
                             y_max,       &
                             z_min,       &
                             z_max,       &
                             advect_x,    &
                             dir,         &
                             sweep_number,&
                             vertexdx,    &
                             vertexdy,    &
                             vertexdz,    &
                             volume,      &
                             density1,    &
                             energy1,     &
                             mass_flux_x, &
                             vol_flux_x,  &
                             mass_flux_y, &
                             vol_flux_y,  &
                             mass_flux_z, &
                             vol_flux_z,  &
                             pre_vol,     &
                             post_vol,    &
                             pre_mass,    &
                             post_mass,   &
                             advec_vol,   &
                             post_ener,   &
                             ener_flux    )

  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER :: sweep_number,dir
  LOGICAL :: advect_x
  INTEGER :: g_xdir=1,g_ydir=2,g_zdir=3

  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: volume
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: energy1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: vol_flux_x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: vol_flux_y
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: vol_flux_z
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: mass_flux_x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: mass_flux_y
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: mass_flux_z
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: pre_vol
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: post_vol
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: pre_mass
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: post_mass
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: advec_vol
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+2) :: post_ener
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+2) :: ener_flux

  REAL(KIND=8), DIMENSION(x_min-2:x_max+3) :: vertexdx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+3) :: vertexdy
  REAL(KIND=8), DIMENSION(z_min-2:z_max+3) :: vertexdz

  INTEGER :: j,k,l,upwind,donor,downwind,dif

  REAL(KIND=8) :: wind,sigma,sigmat,sigmav,sigmam,sigma3,sigma4
  REAL(KIND=8) :: diffuw,diffdw,limiter
  REAL(KIND=8) :: one_by_six=1.0_8/6.0_8

!$OMP PARALLEL

  IF(dir.EQ.g_xdir) THEN

    IF(sweep_number.EQ.1)THEN
!$OMP DO
      DO l=z_min-2,z_max+2
        DO k=y_min-2,y_max+2
          DO j=x_min-2,x_max+2
            pre_vol(j,k,l)=volume(j,k,l)  +(vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l) &
                                           +vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l) &
                                           +vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l))
            post_vol(j,k,l)=pre_vol(j,k,l)-(vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ELSEIF(sweep_number.EQ.3) THEN
!$OMP DO
      DO l=z_min-2,z_max+2
        DO k=y_min-2,y_max+2
          DO j=x_min-2,x_max+2
            pre_vol(j,k,l) =volume(j,k,l)+vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l)
            post_vol(j,k,l)=volume(j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF

!$OMP DO PRIVATE(upwind,donor,downwind,dif,sigmat,sigma3,sigma4,sigmav,sigma,sigmam, &
!$OMP            diffuw,diffdw,limiter)
    DO l=z_min,z_max
      DO k=y_min,y_max
        DO j=x_min,x_max+2

          IF(vol_flux_x(j,k,l).GT.0.0)THEN
            upwind   =j-2
            donor    =j-1
            downwind =j
            dif      =donor
          ELSE
            upwind   =MIN(j+1,x_max+2)
            donor    =j
            downwind =j-1
            dif      =upwind
          ENDIF

          sigmat=ABS(vol_flux_x(j,k,l))/pre_vol(donor,k,l)
          sigma3=(1.0_8+sigmat)*(vertexdx(j)/vertexdx(dif))
          sigma4=2.0_8-sigmat

          sigma=sigmat
          sigmav=sigmat

          diffuw=density1(donor,k,l)-density1(upwind,k,l)
          diffdw=density1(downwind,k,l)-density1(donor,k,l)
          IF(diffuw*diffdw.GT.0.0)THEN
            limiter=(1.0_8-sigmav)*SIGN(1.0_8,diffdw)*MIN(ABS(diffuw),ABS(diffdw)&
                ,one_by_six*(sigma3*ABS(diffuw)+sigma4*ABS(diffdw)))
          ELSE
            limiter=0.0
          ENDIF
          mass_flux_x(j,k,l)=vol_flux_x(j,k,l)*(density1(donor,k,l)+limiter)

          sigmam=ABS(mass_flux_x(j,k,l))/(density1(donor,k,l)*pre_vol(donor,k,l))
          diffuw=energy1(donor,k,l)-energy1(upwind,k,l)
          diffdw=energy1(downwind,k,l)-energy1(donor,k,l)
          IF(diffuw*diffdw.GT.0.0)THEN
            limiter=(1.0_8-sigmam)*SIGN(1.0_8,diffdw)*MIN(ABS(diffuw),ABS(diffdw)&
                ,one_by_six*(sigma3*ABS(diffuw)+sigma4*ABS(diffdw)))
          ELSE
            limiter=0.0
          ENDIF
          ener_flux(j,k,l)=mass_flux_x(j,k,l)*(energy1(donor,k,l)+limiter)

        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO
    DO l=z_min,z_max
      DO k=y_min,y_max
        DO j=x_min,x_max
          pre_mass(j,k,l)=density1(j,k,l)*pre_vol(j,k,l)
          post_mass(j,k,l)=pre_mass(j,k,l)+mass_flux_x(j,k,l)-mass_flux_x(j+1,k,l)
          post_ener(j,k,l)=(energy1(j,k,l)*pre_mass(j,k,l)+ener_flux(j,k,l)-ener_flux(j+1,k,l))/post_mass(j,k,l)
          advec_vol(j,k,l)=pre_vol(j,k,l)+vol_flux_x(j,k,l)-vol_flux_x(j+1,k,l)
          density1(j,k,l)=post_mass(j,k,l)/advec_vol(j,k,l)
          energy1(j,k,l)=post_ener(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

  ELSEIF(dir.EQ.g_ydir) THEN
    IF(sweep_number.EQ.2) THEN
      IF(advect_x) THEN
!$OMP DO
        DO l=z_min-2,z_max+2
          DO k=y_min-2,y_max+2
            DO j=x_min-2,x_max+2
              pre_vol(j,k,l) =volume(j,k,l)  +vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l) &
                                             +vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l)
              post_vol(j,k,l)=pre_vol(j,k,l)-(vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l))
            ENDDO
          ENDDO
        ENDDO
!$OMP END DO
      ELSE
!$OMP DO
        DO l=z_min-2,z_max+2
          DO k=y_min-2,y_max+2
            DO j=x_min-2,x_max+2
              pre_vol(j,k,l) =volume(j,k,l)  +vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l) &
                                             +vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l)
              post_vol(j,k,l)=pre_vol(j,k,l)-(vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l))
            ENDDO
          ENDDO
        ENDDO
!$OMP END DO
      ENDIF
    ENDIF

!$OMP DO PRIVATE(upwind,donor,downwind,dif,sigmat,sigma3,sigma4,sigmav,sigma,sigmam, &
!$OMP            diffuw,diffdw,limiter)
    DO l=z_min,z_max+2
      DO k=y_min,y_max+2
        DO j=x_min,x_max

          IF(vol_flux_y(j,k,l).GT.0.0)THEN
            upwind   =k-2
            donor    =k-1
            downwind =k
            dif      =donor
          ELSE
            upwind   =MIN(k+1,y_max+2)
            donor    =k
            downwind =k-1
            dif      =upwind
          ENDIF

          sigmat=ABS(vol_flux_y(j,k,l))/pre_vol(j,donor,l)
          sigma3=(1.0_8+sigmat)*(vertexdy(k)/vertexdy(dif))
          sigma4=2.0_8-sigmat

          sigma=sigmat
          sigmav=sigmat

          diffuw=density1(j,donor,l)-density1(j,upwind,l)
          diffdw=density1(j,downwind,l)-density1(j,donor,l)
          IF(diffuw*diffdw.GT.0.0)THEN
            limiter=(1.0_8-sigmav)*SIGN(1.0_8,diffdw)*MIN(ABS(diffuw),ABS(diffdw)&
                ,one_by_six*(sigma3*ABS(diffuw)+sigma4*ABS(diffdw)))
          ELSE
            limiter=0.0
          ENDIF
          mass_flux_y(j,k,l)=vol_flux_y(j,k,l)*(density1(j,donor,l)+limiter)

          sigmam=ABS(mass_flux_y(j,k,l))/(density1(j,donor,l)*pre_vol(j,donor,l))
          diffuw=energy1(j,donor,l)-energy1(j,upwind,l)
          diffdw=energy1(j,downwind,l)-energy1(j,donor,l)
          IF(diffuw*diffdw.GT.0.0)THEN
            limiter=(1.0_8-sigmam)*SIGN(1.0_8,diffdw)*MIN(ABS(diffuw),ABS(diffdw)&
                ,one_by_six*(sigma3*ABS(diffuw)+sigma4*ABS(diffdw)))
          ELSE
            limiter=0.0
          ENDIF
          ener_flux(j,k,l)=mass_flux_y(j,k,l)*(energy1(j,donor,l)+limiter)

        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO
    DO l=z_min,z_max
      DO k=y_min,y_max
        DO j=x_min,x_max
          pre_mass(j,k,l)=density1(j,k,l)*pre_vol(j,k,l)
          post_mass(j,k,l)=pre_mass(j,k,l)+mass_flux_y(j,k,l)-mass_flux_y(j,k+1,l)
          post_ener(j,k,l)=(energy1(j,k,l)*pre_mass(j,k,l)+ener_flux(j,k,l)-ener_flux(j,k+1,l))/post_mass(j,k,l)
          advec_vol(j,k,l)=pre_vol(j,k,l)+vol_flux_y(j,k,l)-vol_flux_y(j,k+1,l)
          density1(j,k,l)=post_mass(j,k,l)/advec_vol(j,k,l)
          energy1(j,k,l)=post_ener(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO


  ELSEIF(dir.EQ.g_zdir) THEN

    IF(sweep_number.EQ.1)THEN
!$OMP DO
      DO l=z_min-2,z_max+2
        DO k=y_min-2,y_max+2
          DO j=x_min-2,x_max+2
            pre_vol(j,k,l)=  volume(j,k,l)+(vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l) &
                                           +vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l) &
                                           +vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l))
            post_vol(j,k,l)=pre_vol(j,k,l)-(vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ELSEIF(sweep_number.EQ.3) THEN
!$OMP DO
      DO l=z_min-2,z_max+2
        DO k=y_min-2,y_max+2
          DO j=x_min-2,x_max+2
            pre_vol(j,k,l)= volume(j,k,l)+vol_flux_z(j  ,k,l+1)-vol_flux_z(j,k,l)
            post_vol(j,k,l)=volume(j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF

!$OMP DO PRIVATE(upwind,donor,downwind,dif,sigmat,sigma3,sigma4,sigmav,sigma,sigmam, &
!$OMP            diffuw,diffdw,limiter)
    DO l=z_min,z_max+2
      DO k=y_min,y_max
        DO j=x_min,x_max

          IF(vol_flux_z(j,k,l).GT.0.0)THEN
            upwind   =l-2
            donor    =l-1
            downwind =l
            dif      =donor
          ELSE
            upwind   =MIN(l+1,z_max+2)
            donor    =l
            downwind =l-1
            dif      =upwind
          ENDIF

          sigmat=ABS(vol_flux_z(j,k,l))/pre_vol(j,k,donor)
          sigma3=(1.0_8+sigmat)*(vertexdz(l)/vertexdz(dif))
          sigma4=2.0_8-sigmat

          sigma=sigmat
          sigmav=sigmat

          diffuw=density1(j,k,donor)-density1(j,k,upwind)
          diffdw=density1(j,k,downwind)-density1(j,k,donor)
          IF(diffuw*diffdw.GT.0.0)THEN
            limiter=(1.0_8-sigmav)*SIGN(1.0_8,diffdw)*MIN(ABS(diffuw),ABS(diffdw)&
                ,one_by_six*(sigma3*ABS(diffuw)+sigma4*ABS(diffdw)))
          ELSE
            limiter=0.0
          ENDIF
          mass_flux_z(j,k,l)=vol_flux_z(j,k,l)*(density1(j,k,donor)+limiter)

          sigmam=ABS(mass_flux_z(j,k,l))/(density1(j,k,donor)*pre_vol(j,k,donor))
          diffuw=energy1(j,k,donor)-energy1(j,k,upwind)
          diffdw=energy1(j,k,downwind)-energy1(j,k,donor)
          IF(diffuw*diffdw.GT.0.0)THEN
            limiter=(1.0_8-sigmam)*SIGN(1.0_8,diffdw)*MIN(ABS(diffuw),ABS(diffdw)&
                ,one_by_six*(sigma3*ABS(diffuw)+sigma4*ABS(diffdw)))
          ELSE
            limiter=0.0
          ENDIF
          ener_flux(j,k,l)=mass_flux_z(j,k,l)*(energy1(j,k,donor)+limiter)

        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO
    DO l=z_min,z_max
      DO k=y_min,y_max
        DO j=x_min,x_max
          pre_mass(j,k,l)=density1(j,k,l)*pre_vol(j,k,l)
          post_mass(j,k,l)=pre_mass(j,k,l)+mass_flux_z(j,k,l)-mass_flux_z(j,k,l+1)
          post_ener(j,k,l)=(energy1(j,k,l)*pre_mass(j,k,l)+ener_flux(j,k,l)-ener_flux(j,k,l+1))/post_mass(j,k,l)
          advec_vol(j,k,l)=pre_vol(j,k,l)+vol_flux_z(j,k,l)-vol_flux_z(j,k,l+1)
          density1(j,k,l)=post_mass(j,k,l)/advec_vol(j,k,l)
          energy1(j,k,l)=post_ener(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

  ENDIF

!$OMP END PARALLEL

END SUBROUTINE advec_cell_kernel

END MODULE advec_cell_kernel_module

