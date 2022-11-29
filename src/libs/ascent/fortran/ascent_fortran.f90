!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
!* Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
!* other details. No copyright assignment is required to contribute to Ascent.
!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!

!------------------------------------------------------------------------------
!
! file: ascent.f90
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module ascent
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only: C_PTR, C_CHAR, C_NULL_CHAR
    implicit none

    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine ascent_about(cnode) &
            bind(C, name="ascent_about")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine ascent_about

    !--------------------------------------------------------------------------
    function ascent_create() result(cascent) &
            bind(C, name="ascent_create")
        use iso_c_binding
        implicit none
        type(C_PTR) :: cascent
    end function ascent_create

     !--------------------------------------------------------------------------
    subroutine ascent_destroy(cascent) &
            bind(C, name="ascent_destroy")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cascent
    end subroutine ascent_destroy

    !--------------------------------------------------------------------------
    subroutine ascent_open(cascent,cnode) &
            bind(C, name="ascent_open")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cascent
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine ascent_open

    !--------------------------------------------------------------------------
    subroutine ascent_publish(cascent, cnode) &
            bind(C, name="ascent_publish")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cascent
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine ascent_publish

    !--------------------------------------------------------------------------
    subroutine ascent_execute(cascent, cnode) &
            bind(C, name="ascent_execute")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cascent
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine ascent_execute

    !--------------------------------------------------------------------------
    subroutine ascent_info(cascent, cnode) &
            bind(C, name="ascent_info")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cascent
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine ascent_info

    !--------------------------------------------------------------------------
    subroutine ascent_close(cascent) &
            bind(C, name="ascent_close")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cascent
    end subroutine ascent_close

    !--------------------------------------------------------------------------
    subroutine ascent_timer_start(timer_name) &
            bind(C, name="ascent_timer_start")
        use iso_c_binding
        implicit none
        character(kind=c_char) :: timer_name(*)
    end subroutine ascent_timer_start

    !--------------------------------------------------------------------------
    subroutine ascent_timer_stop(timer_name) &
            bind(C, name="ascent_timer_stop")
        use iso_c_binding
        implicit none
        character(kind=c_char) :: timer_name(*)
    end subroutine ascent_timer_stop
    !--------------------------------------------------------------------------
    subroutine ascent_timer_write() &
            bind(C, name="ascent_timer_write")
        use iso_c_binding
        implicit none
    end subroutine ascent_timer_write
    !--------------------------------------------------------------------------
    end interface
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
end module ascent
!------------------------------------------------------------------------------
