/*
** Fortran to C index translation macros for Mark Govett's ctof for 
** GPU kernels.  
**
** Tom Henderson
** 9/25/08
**
*/

#ifndef _FTOC_MACROS_
#define _FTOC_MACROS_

#define FTNREF1D(i_index,i_lb) ((i_index)-(i_lb))
#define FTNREF2D(i_index,j_index,i_size,i_lb,j_lb) ((i_size)*(j_index-(j_lb))+(i_index)-(i_lb))
#define FTNREF3D(i_index,j_index,k_index,i_size,j_size,i_lb,j_lb,k_lb) (i_size)*(j_size)*(k_index-k_lb)+(i_size)*(j_index-j_lb)+i_index-i_lb
#define FTNREF4D(i_index,j_index,k_index,l_index,i_size,j_size,k_size,i_lb,j_lb,k_lb,l_lb) (i_size)*(j_size)*(k_size)*(l_index-l_lb)+(i_size)*(j_size)*(k_index-k_lb)+(i_size)*(j_index-j_lb)+i_index-i_lb
#define FTNREF5D(i_index,j_index,k_index,l_index,m_index,i_size,j_size,k_size,l_size,i_lb,j_lb,k_lb,l_lb,m_lb) (i_size)*(j_size)*(k_size)*(l_size)*(m_index-m_lb)+(i_size)*(j_size)*(k_size)*(l_index-l_lb)+(i_size)*(j_size)*(k_index-k_lb)+(i_size)*(j_index-j_lb)+i_index-i_lb

#define FTNSIZE1D(i_lb,i_ub) (i_ub-i_lb+1)
#define FTNSIZE2D(i_lb,i_ub,j_lb,j_ub) (i_ub-i_lb+1)*(j_ub-j_lb+1)
#define FTNSIZE3D(i_lb,i_ub,j_lb,j_ub,k_lb,k_ub) (i_ub-i_lb+1)*(j_ub-j_lb+1)*(k_ub-k_lb+1)
#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) >= (b) ? (b) : (a))
#endif
#define SIGN(a,b) (((b) <  (0) && (a > (0))||((b) > (0) && ((a)<(0)))) ? (-a) : (a))
#define SQR(a) ((a)*(a))
#endif /* _FTOC_MACROS_ */

