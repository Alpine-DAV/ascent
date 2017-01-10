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

!>  @brief String manipulation utilities
!>  @author Wayne Gaudin
!>  @details Provides utilities to manipulate and parse Fortran strings.

MODULE clover_case_change

USE data_module

CONTAINS

FUNCTION tolower(string) RESULT (tolower_result)

  IMPLICIT NONE
  CHARACTER (LEN=*), INTENT(IN) :: string
  CHARACTER (LEN=LEN(string)) :: tolower_result
  INTEGER         :: i,ii

  DO i=1,len(string)
    ii=IACHAR(string(i:i))
    SELECT CASE (ii)
      CASE (65:90)
        tolower_result(i:i)=ACHAr(ii+32)
      CASE DEFAULT
        tolower_result(i:i)=string(i:i)
    END SELECT
  ENDDO

END FUNCTION TOLOWER

FUNCTION toupper(string) RESULT (toupper_result)

  IMPLICIT NONE
  CHARACTER (LEN=*), INTENT(IN) :: string
  CHARACTER (LEN=LEN(string)) :: toupper_result
  INTEGER         :: i,ii

  DO i=1,LEN(string)
    ii=IACHAR(string(i:i))
    SELECT CASE (ii)
      CASE (97:122)
        toupper_result(i:i)=ACHAR(ii-32)
      CASE DEFAULT
        toupper_result(i:i)=string(i:i)
    END SELECT
  ENDDO

END FUNCTION toupper

END MODULE clover_case_change

MODULE clover_isitanint_mod

CONTAINS

FUNCTION isitanint(instring) RESULT(isitanint_result)

  IMPLICIT NONE

  CHARACTER(LEN=*), INTENT(IN) :: instring
  LOGICAL :: isitanint_result

  INTEGER         :: i,ii

  isitanint_result=.TRUE.

  DO i=1,LEN(instring)
    ii=IACHAR(instring(i:i))
    SELECT CASE(ii)
      CASE (43,45,48:57)
        IF(i.NE.1) THEN
          IF(ii.EQ.43 .OR.ii.EQ.45) isitanint_result=.FALSE.
        ENDIF
      CASE DEFAULT
        isitanint_result=.FALSE.
    END SELECT
  ENDDO

END FUNCTION isitanint

END MODULE clover_isitanint_mod

MODULE parse_module

  USE data_module
  USE report_module

  IMPLICIT NONE

  INTEGER, PARAMETER :: len_max=g_len_max &
                       ,dummy=0

  INTEGER         :: iu ! Current unit number.

  CHARACTER(LEN=len_max) :: mask   &
                           ,line   &
                           ,here   &
                           ,sel    &
                           ,rest
CONTAINS

FUNCTION parse_init(iunit,cmask)

  ! Initialise for new set of reads.

  IMPLICIT NONE

  INTEGER :: parse_init

  INTEGER         :: iunit
  CHARACTER(LEN=*)  :: cmask

  INTEGER :: ios

  iu=iunit
  mask=cmask ! Set mask for which part of the file we are interested in.
  line=''
  here=''
  rest=''

  REWIND(UNIT=iunit,IOSTAT=ios)

  parse_init=ios

END FUNCTION parse_init

FUNCTION parse_getline(dummy)

  IMPLICIT NONE

  INTEGER :: parse_getline
  INTEGER        ,INTENT(IN) :: dummy

  INTEGER                :: s,ios,i,parse_out
  CHARACTER(LEN=len_max) :: l,nugget,string_temp1,string_temp2

  DO 
    READ(UNIT=iu,IOSTAT=ios,FMT='(a150)') l ! Read in next line.

    parse_getline=ios

    IF(parse_getline.NE.0) RETURN

      DO i=1,len_trim(l)
        if (IACHAR(l(i:)).LT.32.OR.IACHAR(l(i:i)).GT.128) l(i:i)=' '
      ENDDO

      l=TRIM(ADJUSTL(l))

      s=SCAN(l,'!')
      IF(s.GT.0) l=TRIM(l(1:s-1))
      s=scan(l,';')
      IF(s.GT.0) l=TRIM(l(1:s-1))

      IF( ABS(dummy).NE.1) THEN
         DO i=1,LEN(l)
            IF(IACHAR(l(i:i)).GT.64.AND.IACHAR(l(i:i)).LT.91) &
                 l(i:i)=ACHAR(IACHAR(l(i:i))+32)
         ENDDO
      ENDIF

      IF(dummy.LT.0)THEN
        line=l
      ELSE

        DO i=1,LEN(l)
          IF(l(i:i).EQ.'='.OR.l(i:i).EQ.',') l(i:i)=' '
       ENDDO

       IF(l(1:8).NE.'*select:')THEN
         IF(l(1:1).EQ.'*')THEN
           s=SCAN(l,' ')
           IF(s.EQ.0) s=LEN_TRIM(l)+1
             nugget=TRIM(l(1:s-1))
             IF(nugget(2:4).EQ.'end')THEN
               nugget='*'//TRIM(nugget(5:))

               s=SCAN(here,'*',.TRUE.)

               parse_out=parse_scan(sel,nugget)
               IF(sel.NE.''.AND.parse_out.GT.0)THEN
                 nugget=''
               ENDIF

               IF(nugget.GT.'')THEN
                 string_temp1=TRIM(here(s:))
                 string_temp2=TRIM(nugget(1:))
                 IF(string_temp1.NE.string_temp2)THEN
                   WRITE(*,*) 'l:      ',trim(adjustl(l))
                   WRITE(*,*) 'nugget: ',trim(adjustl(nugget))
                   WRITE(*,*) 'here:   ',trim(adjustl(here))
                   WRITE(*,*) 'sel:    ',trim(adjustl(sel))
                   CALL report_error('parse_getline','Unmatched */*end pair.')
                 ELSE
                   IF(mask.EQ.here) THEN
                     here=here(1:s-1)
                     parse_getline=-1
                     rest='returned_before_eof'
                     RETURN
                   ENDIF
                     here=here(1:s-1)
                   ENDIF
                 ENDIF
               ELSE
                 parse_out=parse_scan(sel,nugget)
                 IF(sel.NE.''.AND.parse_out.GT.0)THEN
                   nugget=''
                 ENDIF

                 here=TRIM(TRIM(here)//nugget)
                 IF(rest.EQ.'returned_before_eof') rest=''
                 rest=TRIM(TRIM(rest)//l(s:))
              ENDIF
                line=''
            ELSE
              IF(here.EQ.mask.OR.mask.EQ.'')THEN
                line=TRIM(l)
              ELSE
                line=''
              ENDIF
            ENDIF
          ELSE
            sel=TRIM(l(9:))
            line=''
          ENDIF
        ENDIF

        IF(line.GT.'') EXIT
     ENDDO

  END FUNCTION parse_getline

  RECURSIVE FUNCTION parse_getword(wrap) RESULT(getword)

    INTEGER                :: stat
    LOGICAL                :: wrap
    CHARACTER(LEN=len_max) :: getword,temp

    INTEGER :: s

    DO WHILE(line(1:1).EQ.' '.AND.LEN_TRIM(line).GT.0)
      line=TRIM(line(2:))
    ENDDO

    s=SCAN(line,' ')
    IF(s.EQ.0) s=LEN_TRIM(line)

    temp=TRIM(line(1:s))

    IF(temp.EQ.ACHAR(92).OR.(temp.EQ.''.AND.wrap))THEN
       temp=''
       stat=parse_getline(dummy)
       IF(stat.EQ.0) temp=parse_getword(wrap)
       getword=temp
    ELSE
       getword=temp

       line=TRIM(line(s+1:))
    ENDIF

  END FUNCTION parse_getword

  FUNCTION parse_getival(word)

    USE clover_module

    CHARACTER(LEN=*)  :: word
    INTEGER         :: temp,parse_getival

    INTEGER :: ios

    READ(UNIT=word,FMT="(I7)",IOSTAT=ios) temp

    IF(ios.NE.0)THEN
       CALL report_error('parse_getival','Error attempting to convert to integer:'//word)
       CALL clover_abort
    ENDIF

    parse_getival=temp

  END FUNCTION parse_getival

  FUNCTION parse_getlval(word)

    USE clover_module

    CHARACTER(LEN=*)  :: word
    LOGICAL :: temp,parse_getlval

    INTEGER :: ios

    ios=0

    SELECT CASE(word)
       CASE('on')
          temp=.TRUE.
       CASE('true')
          temp=.TRUE.
       CASE('off')
          temp=.FALSE.
       CASE('false')
          temp=.FALSE.
       CASE DEFAULT
          ios=99999
    END SELECT

    IF(ios.NE.0)THEN
       CALL report_error('parse_getlval','Error attempting to convert to logical:'//word)
       CALL clover_abort
    ENDIF

    parse_getlval=temp

  END FUNCTION parse_getlval

  FUNCTION parse_getrval(word)

    USE clover_module

    CHARACTER(LEN=*) :: word
    REAL(KIND=8)   :: temp,parse_getrval

    INTEGER :: ios

    ! Make an integer into a float if necessary.

    IF(SCAN(word,'.').EQ.0) word=TRIM(TRIM(word)//'.0')

    READ(UNIT=word,FMT="(E27.20)",IOSTAT=ios) temp

    IF(ios.NE.0)THEN
       CALL report_error('parse_getrval','Error attempting to convert to real:'//word)
       CALL clover_abort
    ENDIF

    parse_getrval=temp

  END FUNCTION parse_getrval

  FUNCTION parse_scan(string,set)

    ! Improved version of F90 SCAN.

    INTEGER                     :: parse_scan
    CHARACTER(LEN=*),INTENT(IN) :: string,set
    CHARACTER(LEN=LEN_MAX)      :: set_temp

    INTEGER :: i,l,temp

    l=LEN_TRIM(set)-1

    temp=0
    set_temp=TRIM(set)
    DO i=1,LEN_TRIM(string)-l
       IF(string(i:i+l).EQ.set_temp)THEN
          temp=i
          EXIT
       ENDIF
    ENDDO

    parse_scan=temp

  END FUNCTION parse_scan

END MODULE parse_module
