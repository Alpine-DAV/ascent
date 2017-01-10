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

!>  @brief Fortran chunk initialisation kernel.
!>  @author Wayne Gaudin
!>  @details Calculates mesh geometry for the mesh chunk based on the mesh size.

MODULE initialise_chunk_kernel_module

CONTAINS

SUBROUTINE initialise_chunk_kernel(x_min,x_max,y_min,y_max,z_min,z_max,&
                                   xmin,ymin,zmin,dx,dy,dz,            &
                                   vertexx,                            &
                                   vertexdx,                           &
                                   vertexy,                            &
                                   vertexdy,                           &
                                   vertexz,                            &
                                   vertexdz,                           &
                                   cellx,                              &
                                   celldx,                             &
                                   celly,                              &
                                   celldy,                             &
                                   cellz,                              &
                                   celldz,                             &
                                   volume,                             &
                                   xarea,                              &
                                   yarea,                              &
                                   zarea                               )

  IMPLICIT NONE

  INTEGER      :: x_min,x_max,y_min,y_max,z_min,z_max
  REAL(KIND=8) :: xmin,ymin,zmin,dx,dy,dz
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3) :: vertexx
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3) :: vertexdx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+3) :: vertexy
  REAL(KIND=8), DIMENSION(y_min-2:y_max+3) :: vertexdy
  REAL(KIND=8), DIMENSION(z_min-2:z_max+3) :: vertexz
  REAL(KIND=8), DIMENSION(z_min-2:z_max+3) :: vertexdz
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2) :: cellx
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2) :: celldx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2) :: celly
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2) :: celldy
  REAL(KIND=8), DIMENSION(z_min-2:z_max+2) :: cellz
  REAL(KIND=8), DIMENSION(z_min-2:z_max+2) :: celldz
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2 ,y_min-2:y_max+2,z_min-2:z_max+2) :: volume
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3 ,y_min-2:y_max+2,z_min-2:z_max+2) :: xarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2 ,y_min-2:y_max+3,z_min-2:z_max+2) :: yarea
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2 ,y_min-2:y_max+2,z_min-2:z_max+3) :: zarea

  INTEGER      :: j,k,l

!$OMP PARALLEL
!$OMP DO
  DO j=x_min-2,x_max+3
     vertexx(j)=xmin+dx*float(j-x_min)
  ENDDO
!$OMP END DO

!$OMP DO
  DO j=x_min-2,x_max+3
    vertexdx(j)=dx
  ENDDO
!$OMP END DO

!$OMP DO
  DO k=y_min-2,y_max+3
     vertexy(k)=ymin+dy*float(k-y_min)
  ENDDO
!$OMP END DO

!$OMP DO
  DO k=y_min-2,y_max+3
    vertexdy(k)=dy
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min-2,z_max+3
     vertexz(l)=zmin+dz*float(l-z_min)
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min-2,z_max+3
    vertexdz(l)=dz
  ENDDO
!$OMP END DO

!$OMP DO
  DO j=x_min-2,x_max+2
     cellx(j)=0.5*(vertexx(j)+vertexx(j+1))
  ENDDO
!$OMP END DO

!$OMP DO
  DO j=x_min-2,x_max+2
     celldx(j)=dx
  ENDDO
!$OMP END DO

!$OMP DO
  DO k=y_min-2,y_max+2
     celly(k)=0.5*(vertexy(k)+vertexy(k+1))
  ENDDO
!$OMP END DO

!$OMP DO
  DO k=y_min-2,y_max+2
     celldy(k)=dy
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min-2,z_max+2
     cellz(l)=0.5*(vertexz(l)+vertexz(l+1))
  ENDDO
!$OMP END DO

!$OMP DO
  DO l=z_min-2,z_max+2
     celldz(l)=dz
  ENDDO
!$OMP END DO

!$OMP DO PRIVATE(j,k)
  DO l=z_min-2,z_max+2
    DO k=y_min-2,y_max+2
      DO j=x_min-2,x_max+2
        volume(j,k,l)=dx*dy*dz
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO PRIVATE(j,k)
  DO l=z_min-2,z_max+2
    DO k=y_min-2,y_max+2
      DO j=x_min-2,x_max+2
        xarea(j,k,l)=celldy(k)*celldz(l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO PRIVATE(j,k)
  DO l=z_min-2,z_max+2
    DO k=y_min-2,y_max+2
      DO j=x_min-2,x_max+2
        yarea(j,k,l)=celldx(j)*celldz(l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO

!$OMP DO PRIVATE(j,k)
  DO l=z_min-2,z_max+2
    DO k=y_min-2,y_max+2
      DO j=x_min-2,x_max+2
        zarea(j,k,l)=celldx(j)*celldy(k)
      ENDDO
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE initialise_chunk_kernel

END MODULE initialise_chunk_kernel_module
