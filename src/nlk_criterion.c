/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2014 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
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


/** @file nlk_criterion.c
 * Neural Network Criterions
 */

#include "nlk_math.h"
#include "nlk_array.h"


/** 
 * Negative Log Likelihood error for multiclass (1-of-K) classification.
 * Requires log-probabilities as input (e.g. log_softmax)
 *
 * @param prediction    network output
 * @param target        target output (i.e. class/neuron index)
 *
 * @return the network error
 */
nlk_real
nlk_nll_forward(const NLK_ARRAY *prediction, const size_t target)
{
    return -prediction->data[target];

}

/**  
 * Negative Log Likelihood error gradient for Multiclass Classification
 * Requires log-probabilities as input (e.g. log_softmax)
 *
 * @param prediction    network output
 * @param target        target output (index)
 * @param gradient      resulting gradient (overwritten)
 */
void
nlk_nll_backprop(const NLK_ARRAY *prediction, const size_t target, 
                 NLK_ARRAY *gradient)
{
    nlk_array_zero(gradient);

    gradient->data[target] = -prediction->data[target];

    NLK_CHECK_NAN(gradient->data[target], "NaN in output");
}

void
nlk_nll_backprop_reg(const NLK_ARRAY *prediction,  
                     const size_t target, const nlk_real reg, NLK_ARRAY *gradient)
{
    nlk_array_zero(gradient);

    gradient->data[target] = -prediction->data[target] + reg;

    NLK_CHECK_NAN(gradient->data[target], "NaN in output");
}
