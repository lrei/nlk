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


/** @file nlk_math.c
 * Math functions
 */

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "nlk_err.h"

#include "nlk_math.h"


nlk_real *__sigmoid_table = NULL; /*< global sigmoid table */


/**
 * Create a sigmoid table for computing 1/(exp(-x) + 1)
 * 
 * @param size      table size
 *
 * @return  no return but sets up a global table
 *
 * @note
 * Learned this little performance trick from word2vec.
 * Another trick (not used) is Leon Bottou approx exp(-x) in Torch7
 * @endnote
 */
void
nlk_table_sigmoid_create()
{

    /* allocate array and set fields */
    __sigmoid_table = malloc(NLK_SIGMOID_TABLE_SIZE * sizeof(nlk_real));
    if(__sigmoid_table == NULL) {
        NLK_ERROR_VOID("failed to allocate memory for sigmoid table",
                       NLK_ENOMEM);
        /* unreachable */
    }

    /** @section Precompute the sigmoid values
    * Initialize an array with the values of the sigmoid between 
    * [-max_exp, max_exp] split evenly into the number of elements in the array.
    */
    /* this splits the range [sigma(-max), sigma(max)] into *size* pieces */
    for(size_t ii = 0; ii < NLK_SIGMOID_TABLE_SIZE; ii++) {
        __sigmoid_table[ii] = exp(((nlk_real) ii / 
                                  (nlk_real) NLK_SIGMOID_TABLE_SIZE * 2 - 1) * 
                                  NLK_MAX_EXP);

        __sigmoid_table[ii] = __sigmoid_table[ii] / (__sigmoid_table[ii] + 1);
    }
}

/**
 * Calculates the sigmoid 1/(exp(-x) + 1) for a real valued x.
 * 
 * @param x                 the real valued x for calculating its sigmoid
 *
 * @return the sigmoid of x
 */
nlk_real
nlk_sigmoid(const nlk_real x) 
{
    int idx;
#ifdef CHECK_NANS
    if(isnan(x)) {
        NLK_ERROR("NaN", NLK_ENAN);
        /* unreachable */
    }
#endif

    /*
    if(sigmoid_table == NULL) {
        return 1.0 / (1.0 + exp(-x));
    }
    */

    if(x >= NLK_MAX_EXP) {
        return 1;
    } else if(x  <= -NLK_MAX_EXP) {
        return 0;
    }

    /* calculate index of value in table */
    idx = ((x + NLK_MAX_EXP) * 
           ((double) NLK_SIGMOID_TABLE_SIZE / (nlk_real) NLK_MAX_EXP / 2.0));

    return __sigmoid_table[idx];
}
