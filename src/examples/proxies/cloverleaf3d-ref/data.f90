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

!>  @brief Holds parameters definitions
!>  @author Wayne Gaudin
!>  @details Parameters used in the CloverLeaf are defined here.

MODULE data_module

   IMPLICIT NONE

   REAL(KIND=8), PARAMETER :: g_version=1.1

   INTEGER,      PARAMETER :: g_ibig=640000

   REAL(KIND=8), PARAMETER :: g_small=1.0e-16  &
                             ,g_big  =1.0e+21

   INTEGER,      PARAMETER :: g_name_len_max=255 &
                             ,g_xdir=1           &
                             ,g_ydir=2           &
                             ,g_zdir=3

   ! These two need to be kept consistent with update_halo
   INTEGER,      PARAMETER :: CHUNK_LEFT   =1    &
                             ,CHUNK_RIGHT  =2    &
                             ,CHUNK_BOTTOM =3    &
                             ,CHUNK_TOP    =4    &
                             ,CHUNK_BACK   =5    &
                             ,CHUNK_FRONT  =6    &
                             ,EXTERNAL_FACE=-1

   INTEGER,         PARAMETER :: FIELD_DENSITY0   = 1         &
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
                                ,FIELD_MASS_FLUX_Z=19         &
                                ,NUM_FIELDS       =19

   INTEGER,         PARAMETER :: CELL_DATA     = 1,        &
                                 VERTEX_DATA   = 2,        &
                                 X_FACE_DATA   = 3,        &
                                 Y_FACE_DATA   = 4,        &
                                 Z_FACE_DATA   = 5


   ! Time step control constants
   INTEGER,         PARAMETER :: SOUND = 1     &
                                ,X_VEL = 2     &
                                ,Y_VEL = 3     &
                                ,Z_VEL = 4     &
                                ,DIVERG= 5

   INTEGER,         PARAMETER :: g_rect=1 &
                                ,g_circ=2 &
                                ,g_point=3


   INTEGER         ::            g_in           & ! File for input data.
                                ,g_out          & 
                                ,g_out_times    & ! File for sim and vis timings logging.
                                ,g_out_stamps     ! File for time stamp logging.


   TYPE parallel_type
      LOGICAL         ::        parallel &
                               ,boss
      INTEGER         ::        max_task  &
                               ,task      &
                               ,boss_task &
                               ,sim_comm
   END TYPE parallel_type

   TYPE(parallel_type) :: parallel

   INTEGER,        PARAMETER :: g_len_max=500

   INTEGER,        PARAMETER :: chunks_per_task = 1

END MODULE data_module
