!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
!* 
!* Produced at the Lawrence Livermore National Laboratory
!* 
!* LLNL-CODE-716457
!* 
!* All rights reserved.
!* 
!* This file is part of Ascent. 
!* 
!* For details, see: http://software.llnl.gov/ascent/.
!* 
!* Please also read ascent/LICENSE
!* 
!* Redistribution and use in source and binary forms, with or without 
!* modification, are permitted provided that the following conditions are met:
!* 
!* * Redistributions of source code must retain the above copyright notice, 
!*   this list of conditions and the disclaimer below.
!* 
!* * Redistributions in binary form must reproduce the above copyright notice,
!*   this list of conditions and the disclaimer (as noted below) in the
!*   documentation and/or other materials provided with the distribution.
!* 
!* * Neither the name of the LLNS/LLNL nor the names of its contributors may
!*   be used to endorse or promote products derived from this software without
!*   specific prior written permission.
!* 
!* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!* ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
!* LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
!* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
!* DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
!* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
!* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
!* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
!* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
!* POSSIBILITY OF SUCH DAMAGE.
!* 
!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!


!------------------------------------------------------------------------------
!
! t_f_ascent_render_2d.f
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module t_f_ascent_render_2d
!------------------------------------------------------------------------------

  use iso_c_binding
  use fruit
  use conduit
  use conduit_blueprint
  use conduit_blueprint_mesh
  use ascent
  implicit none

!------------------------------------------------------------------------------
contains
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------    
! About test
!------------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine t_ascent_render_2d_basic
        type(C_PTR) cdata
        type(C_PTR) cverify_info
        type(C_PTR) cascent
        type(C_PTR) copen_opts
        type(C_PTR) cactions
        type(C_PTR) cadd_plot
        type(C_PTR) cdraw_plots
        integer res
        !----------------------------------------------------------------------
        call set_case_name("t_ascent_render_2d_basic")
        !----------------------------------------------------------------------
        
        cdata  = conduit_node_create()
        cverify_info = conduit_node_create()
        cascent_info = conduit_node_create()
        cascent = ascent_create()

        call conduit_blueprint_mesh_examples_braid("quads",10_8,10_8,0_8,cdata)
        call assert_true( conduit_blueprint_mesh_verify(ncdata,c_verify_info) .eqv. .true., "verify true on braid quads")
        
        cactions = conduit_node_create()
        cadd_plot = conduit_node_append(cactions)
        CALL conduit_node_set_path_char8_str(cadd_plot,"action", "add_plot")
        CALL conduit_node_set_path_char8_str(cadd_plot,"field_name", "braid")
        CALL conduit_node_set_path_char8_str(cadd_plot,"render_options/file_name", "tout_f_render_2d_default_pipeline")
        CALL conduit_node_set_path_int32(cadd_plot,"render_options/width", 512)
        CALL conduit_node_set_path_int32(cadd_plot,"render_options/height",  512)
        cdraw_plots = conduit_node_append(cactions)
        CALL conduit_node_set_path_char8_str(cdraw_plots,"action", "draw_plots")

        copen_opts = conduit_node_create()
        call ascent_open(cascent,copen_opts)
        call ascent_publish(cascent,cdata)
        call ascent_execute(cascent,cactions)
        call ascent_info(cascent,cascent_info)
        call ascent_close(cascent)

        call ascent_destroy(cascent)
        call conduit_node_destroy(cactions)
        call conduit_node_destroy(cverify_info)
        call conduit_node_destroy(cascent_info)
        call conduit_node_destroy(cdata)

    end subroutine t_ascent_render_2d_basic

!------------------------------------------------------------------------------
end module t_f_ascent_render_2d
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
program fortran_test
!------------------------------------------------------------------------------
  use fruit
  use t_f_ascent_render_2d
  implicit none
  logical ok
  
  call init_fruit

  !----------------------------------------------------------------------------
  ! call our test routines
  !----------------------------------------------------------------------------
  call t_ascent_render_2d_basic

  call fruit_summary
  call fruit_finalize
  call is_all_successful(ok)

  if (.not. ok) then
     call exit(1)
  endif

!------------------------------------------------------------------------------
end program fortran_test
!------------------------------------------------------------------------------


