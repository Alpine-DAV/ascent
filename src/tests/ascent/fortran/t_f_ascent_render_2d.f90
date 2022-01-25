!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Ascent.
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
        type(C_PTR) cascent_info
        type(C_PTR) cascent
        type(C_PTR) copen_opts
        type(C_PTR) cactions
        type(C_PTR) cadd_scenes
        integer res
        !----------------------------------------------------------------------
        call set_case_name("t_ascent_render_2d_basic")
        !----------------------------------------------------------------------

        cdata  = conduit_node_create()
        cverify_info = conduit_node_create()
        cascent_info = conduit_node_create()
        cascent = ascent_create()

        call conduit_blueprint_mesh_examples_braid("quads",10_8,10_8,0_8,cdata)
        call assert_true( conduit_blueprint_mesh_verify(cdata,cverify_info) .eqv. .true., "verify true on braid quads")

        cactions = conduit_node_create()
        cadd_scenes = conduit_node_append(cactions)
        CALL conduit_node_set_path_char8_str(cadd_scenes,"action", "add_scenes")
        CALL conduit_node_set_path_char8_str(cadd_scenes,"scenes/scene1/plots/plt1/type", "pseudocolor")
        CALL conduit_node_set_path_char8_str(cadd_scenes,"scenes/scene1/plots/plt1/field", "braid")
        CALL conduit_node_set_path_char8_str(cadd_scenes,"scenes/scene1/image_prefix", "tout_f_render_2d_default_pipeline")

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


