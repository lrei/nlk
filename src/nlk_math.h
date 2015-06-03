/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to 
 * deal in the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/


/** @file nlk_math.h
 * Math function definitions
 */


#ifndef __NLK_MATH_H__
#define __NLK_MATH_H__

#include <stdint.h>
#include <math.h>

#include "nlk_err.h"

#define NLK_MAX_EXP             6
#define NLK_SIGMOID_TABLE_SIZE  10000


#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif
__BEGIN_DECLS


/** @typedef float nlk_real
 * The basic data type for arithmetic operations
 */
typedef float nlk_real;


void        nlk_table_sigmoid_create();
nlk_real    nlk_sigmoid(const nlk_real);


/**
 * Calculates an approximation of exp(-x) for x positive
 *
 * @param   x the value to calculate exp(-x) for
 *
 * @return  exp(-x) - approximate
 *
 * @note
 * Credits to Leon Bottou via Torch7 (BSD License)
 * @endnote
 */
static inline nlk_real
nlk_exp_minus_approx(const nlk_real x)
{
#ifndef NCHECKS
    if(x < 0) {
        NLK_ERROR("x < 0", NLK_EINVAL);
    }
#endif

    /* clipped, x > 13 => exp(-x) < 2.2e-6  so just return 0 (and no NaNs!) */
    if(x < 13.0) { 
        double y;
        y = 1.0+x*(0.125+x*(0.0078125+x*(0.00032552083+x*1.0172526e-5)));
        y *= y;
        y *= y;
        y *= y;
        y = 1/y;
        return y;
    }
    return 0;
}


/**
 * Approximate log function
 *
 * @param x the value to calculate the log of
 *
 * @return approximate log(x)
 *
 * @note
 * Credit goes to Paulo Mineiro: https://code.google.com/p/fastapprox/
 * (under BSD License)
 * @endnote
 */
static inline nlk_real
nlk_log_approx(const nlk_real x)
{
#ifndef NCHECKS
    if(x < 0) {
        NLK_ERROR("x < 0", NLK_EINVAL);
    }
#endif

    union { 
        float f; 
        uint32_t i; 
    } vx = { x };

    float y = vx.i;
    y *= 8.2629582881927490e-8f;
    return y - 87.989971088f;
}

/**
 * A clipped exp
 * Clips at [-13, 30]
 *
 * @param x the value to calculate the exp of
 *
 * @return exp(x)
 */
static inline nlk_real
nlk_exp(const nlk_real x)
{
    if(x < -13) {
        return 0;
    }
    if(x > 30) {
        return 10686474581525;
    }

    return exp(x);
}


/**
 * Approximate exp function
 *
 * @param x the value to calculate the exp of
 *
 * @return exp(x) approximation
 *
 * @note
 * See A Fast, Compact Approximation of the Exponential Function, 
 * Nicol N. Schraudolph, Neural Computation, p853-862, Volume 11, Issue 4, 1999
 * @endnote
 *
 * @TODO check this function
 */
static inline nlk_real
nlk_exp_approx(const nlk_real x)
{
    static union {
        double d;
        struct { int j,i; } n; /* little endian */
    } _eco;

    #define EXP_A (1048576/0.69314718055994530942)
    #define EXP_C 60801

    /* clipping */
    if(x < -13) { 
        return 0;
    }
    if(x > 30) {
        return 10686474581525;
    }

    return (_eco.n.i = EXP_A*(x) + (1072693248 - EXP_C), _eco.d);
}



#endif /* __NLK_MATH_H__ */
