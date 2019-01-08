/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief C timer function.
 *  @author Oliver Perks
 *  @details C function to call from fortran.
 */

#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <stdlib.h>

void timer_c_( double *elapsed_time);


void timer_c_( double *elapsed_time)
{
   struct timeval t;
   gettimeofday(&t, (struct timezone *) NULL);
   *elapsed_time = t.tv_sec + t.tv_usec * 1.0E-6;

}

