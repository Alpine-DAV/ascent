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

!>  @brief Fortran momentum advection kernel
!>  @author Wayne Gaudin
!>  @details Performs a second order advective remap on the vertex momentum
!>  using van-Leer limiting and directional splitting.
!>  Note that although pre_vol is only set and not used in the update, please
!>  leave it in the method.

MODULE advec_mom_kernel_mod

CONTAINS

SUBROUTINE advec_mom_kernel(x_min,x_max,y_min,y_max,z_min,z_max, &
                            xvel1,                               &
                            yvel1,                               &
                            zvel1,                               &
                            mass_flux_x,                         &
                            vol_flux_x,                          &
                            mass_flux_y,                         &
                            vol_flux_y,                          &
                            mass_flux_z,                         &
                            vol_flux_z,                          &
                            volume,                              &
                            density1,                            &
                            node_flux,                           &
                            node_mass_post,                      &
                            node_mass_pre,                       &
                            advec_vel,                           &
                            mom_flux,                            &
                            pre_vol,                             &
                            post_vol,                            &
                            celldx,                              &
                            celldy,                              &
                            celldz,                              &
                            advect_x,                            &
                            which_vel,                           &
                            sweep_number,                        &
                            direction                            )

  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER :: which_vel,sweep_number,direction
  LOGICAL :: advect_x

  REAL(KIND=8), TARGET,DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: xvel1,yvel1,zvel1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: mass_flux_x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2,z_min-2:z_max+2) :: vol_flux_x
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: mass_flux_y
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3,z_min-2:z_max+2) :: vol_flux_y
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: mass_flux_z
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+3) :: vol_flux_z
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: volume
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: density1
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: node_flux
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: node_mass_post
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: node_mass_pre
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: advec_vel
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: mom_flux
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: pre_vol
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3) :: post_vol

  REAL(KIND=8), DIMENSION(x_min-2:x_max+2) :: celldx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2) :: celldy
  REAL(KIND=8), DIMENSION(z_min-2:z_max+2) :: celldz

  INTEGER :: j,k,l
  INTEGER :: upwind,donor,downwind,dif
  REAL(KIND=8) :: sigma,wind,width
  REAL(KIND=8) :: vdiffuw,vdiffdw,auw,adw,limiter
  REAL(KIND=8) :: vdiffuw2,vdiffdw2,auw2,limiter2
  REAL(KIND=8), POINTER, DIMENSION(:,:,:) :: vel1

  ! Choose the correct velocity, ideally, remove this pointer
  !  if it affects performance.
  ! Leave this one in as a test of performance
  IF(which_vel.EQ.1)THEN
    vel1=>xvel1
  ELSEIF(which_vel.EQ.2)THEN
    vel1=>yvel1
  ELSEIF(which_vel.EQ.3)THEN
    vel1=>zvel1
  ENDIF

!$OMP PARALLEL

! I think these only have to be done once per cell advection sweep. So put in some logic so they are just done the first time

  IF(sweep_number.EQ.1.AND.direction.EQ.1)THEN ! x first
!$OMP DO
    DO l=z_min-2,z_max+2
      DO k=y_min-2,y_max+2
        DO j=x_min-2,x_max+2
          post_vol(j,k,l)= volume(j,k,l)+vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l) &
                                        +vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l)
          pre_vol(j,k,l)=post_vol(j,k,l)+vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(sweep_number.EQ.1.AND.direction.EQ.3)THEN ! z first
!$OMP DO
    DO l=z_min-2,z_max+2
      DO k=y_min-2,y_max+2
        DO j=x_min-2,x_max+2
          post_vol(j,k,l)= volume(j,k,l)+vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l) &
                                        +vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l)
          pre_vol(j,k,l)=post_vol(j,k,l)+vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(sweep_number.EQ.2.AND.advect_x)THEN ! x first
!$OMP DO
    DO l=z_min-2,z_max+2
      DO k=y_min-2,y_max+2
        DO j=x_min-2,x_max+2
          post_vol(j,k,l)=volume(j,k,l) +vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l)
          pre_vol(j,k,l)=post_vol(j,k,l)+vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(sweep_number.EQ.2.AND..NOT.advect_x)THEN ! Z first
!$OMP DO
    DO l=z_min-2,z_max+2
      DO k=y_min-2,y_max+2
        DO j=x_min-2,x_max+2
          post_vol(j,k,l)=volume(j,k,l) +vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l)
          pre_vol(j,k,l)=post_vol(j,k,l)+vol_flux_y(j  ,k+1,l  )-vol_flux_y(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(sweep_number.EQ.3.AND.direction.EQ.1)THEN ! z first
!$OMP DO
    DO l=z_min-2,z_max+2
      DO k=y_min-2,y_max+2
        DO j=x_min-2,x_max+2
          post_vol(j,k,l)=volume(j,k,l)
          pre_vol(j,k,l)=post_vol(j,k,l)+vol_flux_x(j+1,k  ,l  )-vol_flux_x(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(sweep_number.EQ.3.AND.direction.EQ.3)THEN ! x first
!$OMP DO
    DO l=z_min-2,z_max+2
      DO k=y_min-2,y_max+2
        DO j=x_min-2,x_max+2
          post_vol(j,k,l)=volume(j,k,l)
          pre_vol(j,k,l)=post_vol(j,k,l)+vol_flux_z(j  ,k  ,l+1)-vol_flux_z(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ENDIF

  IF(direction.EQ.1)THEN
    IF(which_vel.EQ.1) THEN
!$OMP DO
      DO l=z_min,z_max+1
        DO k=y_min,y_max+1
          DO j=x_min-2,x_max+2
            ! Find staggered mesh mass fluxes, nodal masses and volumes.
            node_flux(j,k,l)=0.125_8*(mass_flux_x(j  ,k-1,l  )+mass_flux_x(j  ,k,l  )  &
                                     +mass_flux_x(j+1,k-1,l  )+mass_flux_x(j+1,k,l  )  &
                                     +mass_flux_x(j  ,k-1,l-1)+mass_flux_x(j  ,k,l-1)  &
                                     +mass_flux_x(j+1,k-1,l-1)+mass_flux_x(j+1,k,l-1))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
!And do I need to calc the node mass for all 3 directions, or just once?
!$OMP DO
      DO l=z_min,z_max+1
        DO k=y_min,y_max+1
          DO j=x_min-1,x_max+2
            ! Staggered cell mass post advection
            node_mass_post(j,k,l)=0.125_8*(density1(j  ,k-1,l  )*post_vol(j  ,k-1,l  )                   &
                                          +density1(j  ,k  ,l  )*post_vol(j  ,k  ,l  )                   &
                                          +density1(j-1,k-1,l  )*post_vol(j-1,k-1,l  )                   &
                                          +density1(j-1,k  ,l  )*post_vol(j-1,k  ,l  )                   &
                                          +density1(j  ,k-1,l-1)*post_vol(j  ,k-1,l-1)                   &
                                          +density1(j  ,k  ,l-1)*post_vol(j  ,k  ,l-1)                   &
                                          +density1(j-1,k-1,l-1)*post_vol(j-1,k-1,l-1)                   &
                                          +density1(j-1,k  ,l-1)*post_vol(j-1,k  ,l-1))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
!$OMP DO
      DO l=z_min,z_max+1
        DO k=y_min,y_max+1
          DO j=x_min-1,x_max+2
            ! Staggered cell mass pre advection
            node_mass_pre(j,k,l)=node_mass_post(j,k,l)-node_flux(j-1,k,l)+node_flux(j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF

!$OMP DO PRIVATE(upwind,downwind,donor,dif,sigma,width,limiter,vdiffuw,vdiffdw,auw,adw,wind)
    DO l=z_min,z_max+1
      DO k=y_min,y_max+1
        DO j=x_min-1,x_max+1
          IF(node_flux(j,k,l).LT.0.0)THEN
            upwind=j+2
            donor=j+1
            downwind=j
            dif=donor
          ELSE
            upwind=j-1
            donor=j
            downwind=j+1
            dif=upwind
          ENDIF
          sigma=ABS(node_flux(j,k,l))/(node_mass_pre(donor,k,l))
          width=celldx(j)
          vdiffuw=vel1(donor,k,l)-vel1(upwind,k,l)
          vdiffdw=vel1(downwind,k,l)-vel1(donor,k,l)
          limiter=0.0
          IF(vdiffuw*vdiffdw.GT.0.0)THEN
            auw=ABS(vdiffuw)
            adw=ABS(vdiffdw)
            wind=1.0_8
            IF(vdiffdw.LE.0.0) wind=-1.0_8
            limiter=wind*MIN(width*((2.0_8-sigma)*adw/width+(1.0_8+sigma)*auw/celldx(dif))/6.0_8,auw,adw)
          ENDIF
          advec_vel(j,k,l)=vel1(donor,k,l)+(1.0-sigma)*limiter
          mom_flux(j,k,l)=advec_vel(j,k,l)*node_flux(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO
    DO l=z_min,z_max+1
      DO k=y_min,y_max+1
        DO j=x_min,x_max+1
          vel1 (j,k,l)=(vel1 (j,k,l)*node_mass_pre(j,k,l)+mom_flux(j-1,k,l)-mom_flux(j,k,l))/node_mass_post(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(direction.EQ.2)THEN
    IF(which_vel.EQ.1)THEN
!$OMP DO
      DO l=z_min,z_max+1
        DO k=y_min-2,y_max+2
          DO j=x_min,x_max+1
            ! Find staggered mesh mass fluxes and nodal masses and volumes.
            node_flux(j,k,l)=0.125_8*(mass_flux_y(j-1,k  ,l  )+mass_flux_y(j  ,k  ,l  ) &
                                     +mass_flux_y(j-1,k+1,l  )+mass_flux_y(j  ,k+1,l  ) &
                                     +mass_flux_y(j-1,k  ,l-1)+mass_flux_y(j  ,k  ,l-1) &
                                     +mass_flux_y(j-1,k+1,l-1)+mass_flux_y(j  ,k+1,l-1))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
!$OMP DO
      DO l=z_min,z_max+1
        DO k=y_min-1,y_max+2
          DO j=x_min,x_max+1
            node_mass_post(j,k,l)=0.125_8*(density1(j  ,k-1,l  )*post_vol(j  ,k-1,l  )                     &
                                          +density1(j  ,k  ,l  )*post_vol(j  ,k  ,l  )                     &
                                          +density1(j-1,k-1,l  )*post_vol(j-1,k-1,l  )                     &
                                          +density1(j-1,k  ,l  )*post_vol(j-1,k  ,l  )                     &
                                          +density1(j  ,k-1,l-1)*post_vol(j  ,k-1,l-1)                     &
                                          +density1(j  ,k  ,l-1)*post_vol(j  ,k  ,l-1)                     &
                                          +density1(j-1,k-1,l-1)*post_vol(j-1,k-1,l-1)                     &
                                          +density1(j-1,k  ,l-1)*post_vol(j-1,k  ,l-1))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
!$OMP DO
      DO l=z_min,z_max+1
        DO k=y_min-1,y_max+2
          DO j=x_min,x_max+1
            node_mass_pre(j,k,l)=node_mass_post(j,k,l)-node_flux(j,k-1,l)+node_flux(j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
!$OMP DO PRIVATE(upwind,donor,downwind,dif,sigma,width,limiter,vdiffuw,vdiffdw,auw,adw,wind)
    DO l=z_min,z_max+1
      DO k=y_min-1,y_max+1
        DO j=x_min,x_max+1
          IF(node_flux(j,k,l).LT.0.0)THEN
            upwind=k+2
            donor=k+1
            downwind=k
            dif=donor
          ELSE
            upwind=k-1
            donor=k
            downwind=k+1
            dif=upwind
          ENDIF

          sigma=ABS(node_flux(j,k,l))/(node_mass_pre(j,donor,l))
          width=celldy(k)
          vdiffuw=vel1(j,donor,l)-vel1(j,upwind,l)
          vdiffdw=vel1(j,downwind,l)-vel1(j,donor,l)
          limiter=0.0
          IF(vdiffuw*vdiffdw.GT.0.0)THEN
            auw=ABS(vdiffuw)
            adw=ABS(vdiffdw)
            wind=1.0_8
            IF(vdiffdw.LE.0.0) wind=-1.0_8
            limiter=wind*MIN(width*((2.0_8-sigma)*adw/width+(1.0_8+sigma)*auw/celldy(dif))/6.0_8,auw,adw)
          ENDIF
          advec_vel(j,k,l)=vel1(j,donor,l)+(1.0_8-sigma)*limiter
          mom_flux(j,k,l)=advec_vel(j,k,l)*node_flux(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO
    DO l=z_min,z_max+1
      DO k=y_min,y_max+1
        DO j=x_min,x_max+1
          vel1 (j,k,l)=(vel1(j,k,l)*node_mass_pre(j,k,l)+mom_flux(j,k-1,l)-mom_flux(j,k,l))/node_mass_post(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ELSEIF(direction.EQ.3)THEN
    IF(which_vel.EQ.1) THEN
!$OMP DO
      DO l=z_min-2,z_max+2
        DO k=y_min,y_max+1
          DO j=x_min,x_max+1
            ! Find staggered mesh mass fluxes and nodal masses and volumes.
            node_flux(j,k,l)=0.125_8*(mass_flux_z(j-1,k  ,l  )+mass_flux_z(j  ,k  ,l  ) &
                                     +mass_flux_z(j-1,k  ,l+1)+mass_flux_z(j  ,k  ,l+1) &
                                     +mass_flux_z(j-1,k-1,l  )+mass_flux_z(j  ,k-1,l  ) &
                                     +mass_flux_z(j-1,k-1,l+1)+mass_flux_z(j  ,k-1,l+1))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
!$OMP DO
      DO l=z_min-1,z_max+2
        DO k=y_min,y_max+1
          DO j=x_min,x_max+1
            node_mass_post(j,k,l)=0.125_8*(density1(j  ,k-1,l  )*post_vol(j  ,k-1,l  )                     &
                                          +density1(j  ,k  ,l  )*post_vol(j  ,k  ,l  )                     &
                                          +density1(j-1,k-1,l  )*post_vol(j-1,k-1,l  )                     &
                                          +density1(j-1,k  ,l  )*post_vol(j-1,k  ,l  )                     &
                                          +density1(j  ,k-1,l-1)*post_vol(j  ,k-1,l-1)                     &
                                          +density1(j  ,k  ,l-1)*post_vol(j  ,k  ,l-1)                     &
                                          +density1(j-1,k-1,l-1)*post_vol(j-1,k-1,l-1)                     &
                                          +density1(j-1,k  ,l-1)*post_vol(j-1,k  ,l-1))
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
!$OMP DO
      DO l=z_min-1,z_max+2
        DO k=y_min,y_max+1
          DO j=x_min,x_max+1
            ! Staggered cell mass pre advection
            node_mass_pre(j,k,l)=node_mass_post(j,k,l)-node_flux(j,k,l-1)+node_flux(j,k,l)
          ENDDO
        ENDDO
      ENDDO
!$OMP END DO
    ENDIF
!$OMP DO PRIVATE(upwind,donor,downwind,dif,sigma,width,limiter,vdiffuw,vdiffdw,auw,adw,wind)
    DO l=z_min-1,z_max+1
      DO k=y_min,y_max+1
        DO j=x_min,x_max+1
          IF(node_flux(j,k,l).LT.0.0)THEN
            upwind=l+2
            donor=l+1
            downwind=l
            dif=donor
          ELSE
            upwind=l-1
            donor=l
            downwind=l+1
            dif=upwind
          ENDIF

          sigma=ABS(node_flux(j,k,l))/(node_mass_pre(j,k,donor))
          width=celldz(l)
          vdiffuw=vel1(j,k,donor)-vel1(j,k,upwind)
          vdiffdw=vel1(j,k,downwind)-vel1(j,k,donor)
          limiter=0.0
          IF(vdiffuw*vdiffdw.GT.0.0)THEN
            auw=ABS(vdiffuw)
            adw=ABS(vdiffdw)
            wind=1.0_8
            IF(vdiffdw.LE.0.0) wind=-1.0_8
            limiter=wind*MIN(width*((2.0_8-sigma)*adw/width+(1.0_8+sigma)*auw/celldz(dif))/6.0_8,auw,adw)
          ENDIF
          advec_vel(j,k,l)=vel1(j,k,donor)+(1.0_8-sigma)*limiter
          mom_flux(j,k,l)=advec_vel(j,k,l)*node_flux(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO

!$OMP DO
    DO l=z_min,z_max+1
      DO k=y_min,y_max+1
        DO j=x_min,x_max+1
          vel1 (j,k,l)=(vel1(j,k,l)*node_mass_pre(j,k,l)+mom_flux(j,k,l-1)-mom_flux(j,k,l))/node_mass_post(j,k,l)
        ENDDO
      ENDDO
    ENDDO
!$OMP END DO
  ENDIF

!$OMP END PARALLEL

END SUBROUTINE advec_mom_kernel

END MODULE advec_mom_kernel_mod
