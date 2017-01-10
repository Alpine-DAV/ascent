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

!>  @brief Fortran mpi buffer packing kernel
!>  @author Wayne Gaudin
!>  @details Packs/unpacks mpi send and receive buffers

! Notes
! All the 1d indices need updateing to 3d

MODULE pack_kernel_module

CONTAINS

SUBROUTINE clover_pack_message_left(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                    left_snd_buffer,                                           &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                    depth,field_type,                                          &
                                    buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: left_snd_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Pack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

!$OMP PARALLEL DO PRIVATE(index)
  DO l=z_min-depth,z_max+z_inc+depth
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=1,depth
        index=buffer_offset + (j+(k+depth-1)*depth) + ((l+depth-1)*(y_max+y_inc+2*depth)*depth)
        left_snd_buffer(index)=field(x_min+x_inc-1+j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END PARALLEL DO

END SUBROUTINE clover_pack_message_left

SUBROUTINE clover_unpack_message_left(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                      left_rcv_buffer,                                           &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth,field_type,                                          &
                                      buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: left_rcv_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Unpack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

!$OMP PARALLEL DO PRIVATE(index)
  DO l=z_min-depth,z_max+z_inc+depth
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=1,depth
        index=buffer_offset + (j+(k+depth-1)*depth) + ((l+depth-1)*(y_max+y_inc+2*depth)*depth)
        field(x_min-j,k,l)=left_rcv_buffer(index)
      ENDDO
    ENDDO
  ENDDO
!$OMP END PARALLEL DO

END SUBROUTINE clover_unpack_message_left

SUBROUTINE clover_pack_message_right(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                     right_snd_buffer,                                          &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                     depth,field_type,                                          &
                                     buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: right_snd_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Pack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

!$OMP PARALLEL DO PRIVATE(index)
  DO l=z_min-depth,z_max+z_inc+depth
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=1,depth
        index=buffer_offset + (j+(k+depth-1)*depth) + ((l+depth-1)*(y_max+y_inc+2*depth)*depth)
        right_snd_buffer(index)=field(x_max+1-j,k,l)
      ENDDO
    ENDDO
  ENDDO
!$OMP END PARALLEL DO

END SUBROUTINE clover_pack_message_right

SUBROUTINE clover_unpack_message_right(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                       right_rcv_buffer,                                          &
                                       CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                       depth,field_type,                                          &
                                       buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: right_rcv_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Unpack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

!$OMP PARALLEL DO PRIVATE(index)
  DO l=z_min-depth,z_max+z_inc+depth
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=1,depth
        index=buffer_offset + (j+(k+depth-1)*depth) + ((l+depth-1)*(y_max+y_inc+2*depth)*depth)
        field(x_max+x_inc+j,k,l)=right_rcv_buffer(index)
      ENDDO
    ENDDO
  ENDDO
!$OMP END PARALLEL DO

END SUBROUTINE clover_unpack_message_right

SUBROUTINE clover_pack_message_top(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                   top_snd_buffer,                                            &
                                   CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                   depth,field_type,                                          &
                                   buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: top_snd_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Pack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO k=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO l=z_min-depth,z_max+z_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (l-1+depth)*(x_max+x_inc+2*depth) + (k-1)*((x_max+x_inc+2*depth)*(z_max+z_inc+2*depth))
        top_snd_buffer(index)=field(j,y_max+1-k,l)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_pack_message_top

SUBROUTINE clover_unpack_message_top(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                     top_rcv_buffer,                                            &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                     depth,field_type,                                          &
                                     buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: top_rcv_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Unpack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO k=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO l=z_min-depth,z_max+z_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (l-1+depth)*(x_max+x_inc+2*depth) + (k-1)*((x_max+x_inc+2*depth)*(z_max+z_inc+2*depth))
        field(j,y_max+y_inc+k,l)=top_rcv_buffer(index)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_unpack_message_top

SUBROUTINE clover_pack_message_bottom(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                      bottom_snd_buffer,                                         &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth,field_type,                                          &
                                      buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: bottom_snd_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Pack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO k=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO l=z_min-depth,z_max+z_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (l-1+depth)*(x_max+x_inc+2*depth) + (k-1)*((x_max+x_inc+2*depth)*(z_max+z_inc+2*depth))
        bottom_snd_buffer(index)=field(j,y_min+y_inc-1+k,l)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_pack_message_bottom

SUBROUTINE clover_unpack_message_bottom(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                        bottom_rcv_buffer,                                         &
                                        CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                        depth,field_type,                                          &
                                        buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: bottom_rcv_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Unpack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO k=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO l=z_min-depth,z_max+z_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (l-1+depth)*(x_max+x_inc+2*depth) + (k-1)*((x_max+x_inc+2*depth)*(z_max+z_inc+2*depth))
        field(j,y_min-k,l)=bottom_rcv_buffer(index)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_unpack_message_bottom

SUBROUTINE clover_pack_message_back(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                    back_snd_buffer,                                           &
                                    CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                    depth,field_type,                                          &
                                    buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: back_snd_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Pack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO l=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (k-1+depth)*(x_max+x_inc+2*depth) + (l-1)*((x_max+x_inc+2*depth)*(y_max+y_inc+2*depth))
        back_snd_buffer(index)=field(j,k,z_min+z_inc-1+l)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_pack_message_back

SUBROUTINE clover_unpack_message_back(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                      back_rcv_buffer,                                           &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth,field_type,                                          &
                                      buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: back_rcv_buffer(:)

  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Unpack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO l=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (k-1+depth)*(x_max+x_inc+2*depth) + (l-1)*((x_max+x_inc+2*depth)*(y_max+y_inc+2*depth))
        field(j,k,z_min-l)=back_rcv_buffer(index)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_unpack_message_back

SUBROUTINE clover_pack_message_front(x_min,x_max,y_min,y_max,z_min,z_max,field,                 &
                                     front_snd_buffer,                                          &
                                     CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                     depth,field_type,                                          &
                                     buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: front_snd_buffer(:)
  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Pack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO l=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (k-1+depth)*(x_max+x_inc+2*depth) + (l-1)*((x_max+x_inc+2*depth)*(y_max+y_inc+2*depth))
        front_snd_buffer(index)=field(j,k,z_max+1-l)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_pack_message_front

SUBROUTINE clover_unpack_message_front(x_min,x_max,y_min,y_max,z_min,z_max,field,                &
                                      front_rcv_buffer,                                          &
                                      CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA, &
                                      depth,field_type,                                          &
                                      buffer_offset)

  IMPLICIT NONE

  REAL(KIND=8) :: field(-1:,-1:,-1:) ! This seems to work for any type of mesh data
  REAL(KIND=8) :: front_rcv_buffer(:)
  INTEGER      :: CELL_DATA,VERTEX_DATA,X_FACE_DATA,Y_FACE_DATA,Z_FACE_DATA
  INTEGER      :: depth,field_type,x_min,x_max,y_min,y_max,z_min,z_max
  INTEGER      :: j,k,l,x_inc,y_inc,z_inc,index,buffer_offset

  ! Unpack 

  ! These array modifications still need to be added on, plus the donor data location changes as in update_halo
  IF(field_type.EQ.CELL_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.VERTEX_DATA) THEN
    x_inc=1
    y_inc=1
    z_inc=1
  ENDIF
  IF(field_type.EQ.X_FACE_DATA) THEN
    x_inc=1
    y_inc=0
    z_inc=0
  ENDIF
  IF(field_type.EQ.Y_FACE_DATA) THEN
    x_inc=0
    y_inc=1
    z_inc=0
  ENDIF
  IF(field_type.EQ.Z_FACE_DATA) THEN
    x_inc=0
    y_inc=0
    z_inc=1
  ENDIF

  DO l=1,depth
!$OMP PARALLEL DO PRIVATE(index)
    DO k=y_min-depth,y_max+y_inc+depth
      DO j=x_min-depth,x_max+x_inc+depth
        index= buffer_offset + (j+depth) + (k-1+depth)*(x_max+x_inc+2*depth) + ((l-1)*(x_max+x_inc+2*depth)*(y_max+y_inc+2*depth))
        field(j,k,z_max+z_inc+l)=front_rcv_buffer(index)
      ENDDO
    ENDDO
!$OMP END PARALLEL DO
  ENDDO

END SUBROUTINE clover_unpack_message_front

END MODULE pack_kernel_module
