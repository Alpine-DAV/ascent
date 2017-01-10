!*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*!
!* Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
!* 
!* Produced at the Lawrence Livermore National Laboratory
!* 
!* LLNL-CODE-716457
!* 
!* All rights reserved.
!* 
!* This file is part of Strawman. 
!* 
!* For details, see: http://software.llnl.gov/strawman/.
!* 
!* Please also read strawman/LICENSE
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
! file: strawman.f90
!
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
module strawman
!------------------------------------------------------------------------------
    use, intrinsic :: iso_c_binding, only: C_PTR, C_CHAR, C_NULL_CHAR
    implicit none

    !--------------------------------------------------------------------------
    interface
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    subroutine strawman_about(cnode) &
            bind(C, name="strawman_about")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine strawman_about

    !--------------------------------------------------------------------------
    function strawman_create() result(csman) &
            bind(C, name="strawman_create")
        use iso_c_binding
        implicit none
        type(C_PTR) :: csman
    end function strawman_create
 
     !--------------------------------------------------------------------------
    subroutine strawman_destroy(csman) &
            bind(C, name="strawman_destroy")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::csman
    end subroutine strawman_destroy
 
    !--------------------------------------------------------------------------
    subroutine strawman_open(csman,cnode) &
            bind(C, name="strawman_open")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::csman
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine strawman_open
 
    !--------------------------------------------------------------------------
    subroutine strawman_publish(csman, cnode) &
            bind(C, name="strawman_publish")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::csman
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine strawman_publish
 
    !--------------------------------------------------------------------------
    subroutine strawman_execute(csman, cnode) &
            bind(C, name="strawman_execute")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::csman
        type(C_PTR), value, intent(IN) ::cnode
    end subroutine strawman_execute
 
    !--------------------------------------------------------------------------
    subroutine strawman_close(csman) &
            bind(C, name="strawman_close")
        use iso_c_binding
        implicit none
        type(C_PTR), value, intent(IN) ::csman
    end subroutine strawman_close
 
    !--------------------------------------------------------------------------
    subroutine strawman_timer_start(timer_name) &
            bind(C, name="strawman_timer_start")
        use iso_c_binding
        implicit none
        character(kind=c_char) :: timer_name(*)
    end subroutine strawman_timer_start
    
    !--------------------------------------------------------------------------
    subroutine strawman_timer_stop(timer_name) &
            bind(C, name="strawman_timer_stop")
        use iso_c_binding
        implicit none
        character(kind=c_char) :: timer_name(*)
    end subroutine strawman_timer_stop
    !--------------------------------------------------------------------------
    subroutine strawman_timer_write() &
            bind(C, name="strawman_timer_write")
        use iso_c_binding
        implicit none
    end subroutine strawman_timer_write
    !--------------------------------------------------------------------------
    end interface
    !--------------------------------------------------------------------------

!------------------------------------------------------------------------------
end module strawman
!------------------------------------------------------------------------------
