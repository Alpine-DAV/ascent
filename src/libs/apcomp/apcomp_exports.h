#ifndef APCOMP_EXPORTS_H
#define APCOMP_EXPORTS_H

#if __GNUC__ >= 4 && (defined(APCOMP_COMPILING_FLAG))
  #define APCOMP_API __attribute__ ((visibility("default")))
#else
  #define APCOMP_API /* hidden by default */
#endif

#endif
