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

!>  @brief Fortran viscosity kernel.
!>  @author Wayne Gaudin
!>  @details Calculates an artificial viscosity using the Wilkin's method to
!>  smooth out shock front and prevent oscillations around discontinuities.
!>  Only cells in compression will have a non-zero value.

! NOTES
! The gradients needs checking for 3d
! Strain needs checking

MODULE viscosity_kernel_module

CONTAINS

SUBROUTINE viscosity_kernel(x_min,x_max,y_min,y_max,z_min,z_max,    &
                            xarea,yarea,zarea,                      &
                            celldx,celldy,celldz,                   &
                            density0,                               &
                            pressure,                               &
                            viscosity,                              &
                            xvel0,                                  &
                            yvel0,                                  &
                            zvel0                                   )

  IMPLICIT NONE

  INTEGER     :: x_min,x_max,y_min,y_max,z_min,z_max
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+2 ,z_min-2:z_max+2)    :: xarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+3 ,z_min-2:z_max+2)    :: yarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2 ,z_min-2:z_max+3)    :: zarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2)                                     :: celldx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2)                                     :: celldy
  REAL(KIND=8), DIMENSION(z_min-2:z_max+2)                                     :: celldz
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)     :: density0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)     :: pressure
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2)     :: viscosity
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3,z_min-2:z_max+3)     :: xvel0,yvel0,zvel0

  INTEGER       :: j,k,l
  REAL(KIND=8)  :: ugradx1,ugradx2,vgrady1,vgrady2,wgradz1,wgradz2
  REAL(KIND=8)  :: ugrady1,ugrady2,vgradx1,vgradx2,wgradx1,wgradx2
  REAL(KIND=8)  :: ugradz1,ugradz2,vgradz1,vgradz2,wgrady1,wgrady2
  REAL(KIND=8)  :: xx,yy,zz,xy,xz,yz
  REAL(KIND=8)  :: grad2,pgradx,pgrady,pgradz,pgradx2,pgrady2,pgradz2,grad     &
                  ,ygrad,pgrad,xgrad,zgrad,div,limiter

!$OMP PARALLEL

!$OMP DO PRIVATE(ugradx1,ugradx2,vgrady1,vgrady2,wgradz1,wgradz2,div,                     &
!$OMP            ugrady1,ugrady2,vgradx1,vgradx2,wgradx1,wgradx2,                         &
!$OMP            ugradz1,ugradz2,vgradz1,vgradz2,wgrady1,wgrady2,                         &
!$OMP            pgradx,pgrady,pgradz,pgradx2,pgrady2,pgradz2,limiter,                    &
!$OMP            pgrad,xgrad,ygrad,zgrad,grad,grad2,xx,yy,zz,xy,xz,yz)
  DO l=z_min,z_max
    DO k=y_min,y_max
      DO j=x_min,x_max
        ugradx1=xvel0(j  ,k  ,l  )+xvel0(j  ,k+1,l  )+xvel0(j  ,k  ,l+1)+xvel0(j  ,k+1,l+1)
        ugradx2=xvel0(j+1,k  ,l  )+xvel0(j+1,k+1,l  )+xvel0(j+1,k  ,l+1)+xvel0(j+1,k+1,l+1)
        ugrady1=xvel0(j  ,k  ,l  )+xvel0(j+1,k  ,l  )+xvel0(j  ,k  ,l+1)+xvel0(j+1,k  ,l+1)
        ugrady2=xvel0(j  ,k+1,l  )+xvel0(j+1,k+1,l  )+xvel0(j  ,k+1,l+1)+xvel0(j+1,k+1,l+1)
        ugradz1=xvel0(j  ,k  ,l  )+xvel0(j+1,k  ,l  )+xvel0(j  ,k+1,l  )+xvel0(j+1,k+1,l  )
        ugradz2=xvel0(j  ,k  ,l+1)+xvel0(j+1,k  ,l+1)+xvel0(j  ,k+1,l+1)+xvel0(j+1,k+1,l+1)

        vgradx1=yvel0(j  ,k  ,l  )+yvel0(j  ,k+1,l  )+yvel0(j  ,k  ,l+1)+yvel0(j  ,k+1,l+1)
        vgradx2=yvel0(j+1,k  ,l  )+yvel0(j+1,k+1,l  )+yvel0(j+1,k  ,l+1)+yvel0(j+1,k+1,l+1)
        vgrady1=yvel0(j  ,k  ,l  )+yvel0(j+1,k  ,l  )+yvel0(j  ,k  ,l+1)+yvel0(j+1,k  ,l+1)
        vgrady2=yvel0(j  ,k+1,l  )+yvel0(j+1,k+1,l  )+yvel0(j  ,k+1,l+1)+yvel0(j+1,k+1,l+1)
        vgradz1=yvel0(j  ,k  ,l  )+yvel0(j+1,k  ,l  )+yvel0(j  ,k+1,l  )+yvel0(j+1,k+1,l  )
        vgradz2=yvel0(j  ,k  ,l+1)+yvel0(j+1,k  ,l+1)+yvel0(j  ,k+1,l+1)+yvel0(j+1,k+1,l+1)

        wgradx1=zvel0(j  ,k  ,l  )+zvel0(j  ,k+1,l  )+zvel0(j  ,k  ,l+1)+zvel0(j  ,k+1,l+1)
        wgradx2=zvel0(j+1,k  ,l  )+zvel0(j+1,k+1,l  )+zvel0(j+1,k  ,l+1)+zvel0(j+1,k+1,l+1)
        wgrady1=zvel0(j  ,k  ,l  )+zvel0(j+1,k  ,l  )+zvel0(j  ,k  ,l+1)+zvel0(j+1,k  ,l+1)
        wgrady2=zvel0(j  ,k+1,l  )+zvel0(j+1,k+1,l  )+zvel0(j  ,k+1,l+1)+zvel0(j+1,k+1,l+1)
        wgradz1=zvel0(j  ,k  ,l  )+zvel0(j+1,k  ,l  )+zvel0(j  ,k+1,l  )+zvel0(j+1,k+1,l  )
        wgradz2=zvel0(j  ,k  ,l+1)+zvel0(j+1,k  ,l+1)+zvel0(j  ,k+1,l+1)+zvel0(j+1,k+1,l+1)

        div = (xarea(j,k,l)*(ugradx2-ugradx1)+  yarea(j,k,l)*(vgrady2-vgrady1))+ zarea(j,k,l)*(wgradz2-wgradz1)

        xx = 0.25_8*(ugradx2-ugradx1)/(celldx(j))
        yy = 0.25_8*(vgrady2-vgrady1)/(celldy(k))
        zz = 0.25_8*(wgradz2-wgradz1)/(celldz(l))
        xy = 0.25_8*(ugrady2-ugrady1)/(celldy(k))+0.25_8*(vgradx2-vgradx1)/(celldx(j))
        xz = 0.25_8*(ugradz2-ugradz1)/(celldz(l))+0.25_8*(wgradx2-wgradx1)/(celldx(j))
        yz = 0.25_8*(vgradz2-vgradz1)/(celldz(l))+0.25_8*(wgrady2-wgrady1)/(celldy(k))

        pgradx=(pressure(j+1,k,l)-pressure(j-1,k,l))/(celldx(j)+celldx(j+1))
        pgrady=(pressure(j,k+1,l)-pressure(j,k-1,l))/(celldy(k)+celldy(k+1))
        pgradz=(pressure(j,k,l+1)-pressure(j,k,l-1))/(celldz(l)+celldz(l+1))

        pgradx2 = pgradx*pgradx
        pgrady2 = pgrady*pgrady
        pgradz2 = pgradz*pgradz

        limiter = (xx*pgradx2+yy*pgrady2+zz*pgradz2                    &
                +  xy*pgradx*pgrady+xz*pgradx*pgradz+yz*pgrady*pgradz) &
                / MAX(pgradx2+pgrady2+pgradz2,1.0e-16_8)

        IF ((limiter.GT.0.0).OR.(div.GE.0.0))THEN
          viscosity(j,k,l) = 0.0
        ELSE
          pgradx = SIGN(MAX(1.0e-16_8,ABS(pgradx)),pgradx)
          pgrady = SIGN(MAX(1.0e-16_8,ABS(pgrady)),pgrady)
          pgradz = SIGN(MAX(1.0e-16_8,ABS(pgradz)),pgradz)
          pgrad = SQRT(pgradx*pgradx+pgrady*pgrady+pgradz*pgradz)
          xgrad = ABS(celldx(j)*pgrad/pgradx)
          ygrad = ABS(celldy(k)*pgrad/pgrady)
          zgrad = ABS(celldz(l)*pgrad/pgradz)
          grad  = MIN(xgrad,ygrad,zgrad)
          grad2 = grad*grad

          viscosity(j,k,l)=2.0_8*density0(j,k,l)*grad2*limiter*limiter
        ENDIF

      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP END PARALLEL

END SUBROUTINE viscosity_kernel

END MODULE viscosity_kernel_module
