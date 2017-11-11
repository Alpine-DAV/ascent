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
