#pragma once

#ifndef NO_V8F
    #define V8F
    #define VSIZE 8
#else
    #define VSIZE 1
#endif
#ifndef OMP_OFF
    #define OMP_ON
#endif
#ifndef N_THREADS
    #define N_THREADS 0
#endif

