#pragma once

#include <cmath>
#include <immintrin.h>
#include <math.h>
#include "perfdef.h"
#ifdef V8F
#include "avx_mathfun.h"
#endif

#define SQRT_PI 1.7724538509055159
#define SQRT_PIO2 0.8862269254527579

// typedef float v8sf __attribute__ ((vector_size (sizeof(float)*8)));
// typedef int v8si __attribute__ ((vector_size (sizeof(float)*8)));

#ifdef V8F
static inline v8sf pointwise_exp_avx(v8sf x){
    for(int i = 0; i < 8; i++){
        x[i] = exp(x[i]);
    }
    return x;
}

static inline v8sf exp_avx(v8sf x){
    return exp256_ps(x);
}

static inline v8sf erf256_ps(v8sf x){
    v8sf signmask = {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0};
    signmask = _mm256_and_ps(signmask, x);
    x = exp_avx(-x*x);
    v8sf ans_abs = float(1/SQRT_PIO2)*_mm256_sqrt_ps(1-x)*(float(SQRT_PIO2) + (float(31)/200)*x - (float(341)/8000)*x*x);
    return _mm256_or_ps(signmask, ans_abs);
}
#endif // V8F

static inline float erf_apx(float x){
    float expx = exp(-x*x);
    float v = float(1/SQRT_PIO2)*sqrt(1-expx)*(float(SQRT_PIO2) + (float(31)/200)*expx - (float(341)/8000)*expx*expx);
    if(x < 0){
        v *= -1;
    }
    return v;
}
