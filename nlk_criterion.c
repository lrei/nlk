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

#include <math.h>

#include "nlk_err.h"
#include "nlk_array.h"


/** @fn void nlk_bin_nl_gradient(const nlk_real prediction, 
 *                                             const nlk_real target, 
 *                                             nlk_real *gradient)
 * Negative Likelihood gradient for a single of  binary 
 * classification (e.g. hierarchical softmax)
 *
 * @param prediction    network output
 * @param target        target output (binary code: 0, 1)
 * @param gradient      resulting gradient (overwritten)
 *
 * @note
 * Apply learning rate to the gradient after :)
 * @endnote
 */
void
nlk_bin_nl_sgradient(const nlk_real prediction, const uint8_t target, 
                      nlk_real *gradient)
{
    *gradient = 1 - target - prediction;
}

/** @fn int nlk_binary_neg_log_likelihood_gradient(const NLK_ARRAY *prediction, 
 *                                                 const uint8_t *target)
 * Negative Likelihood gradient for each of multiple independent binary 
 * classifications (e.g. hierarchical softmax)
 *
 * @param prediction    network output
 * @param target        target output (binary code)
 * @param gradient      resulting gradient (overwritten)
 *
 * @note
 * Apply learning rate to the gradient after :)
 * @endnote
 */
nlk_real
nlk_binary_neg_log_likelihood(const NLK_ARRAY *prediction,
                              const uint8_t *target)
{
    size_t ii;
    nlk_real error = 0;
    size_t len = prediction->rows * prediction->cols;

    /* @todo parallel cblas */
    for(ii = 0; ii < len; ii++) {
        error -= target[ii] * log(prediction->data[ii]) -
                 (1 - target[ii]) * log(1 - prediction->data[ii]);
    }
    
    return NLK_SUCCESS;
}

/** @fn nlk_real nlk_neg_likelihood_gradient(const NLK_ARRAY *prediction, 
 *                                           const uint8_t *target,
 *                                           NLK_ARRAY *gradient)
 * Negative Log Likelihood error gradient for Multiclass Classification
 *
 * @param prediction    network output
 * @param target        target output (index)
 * @param gradient      resulting gradient (overwritten)
 *
 * @return the network error
 *
 * @note
 * Apply learning rate to the gradient after :)
 * @endnote
 */
int
nlk_neg_log_likelihood_gradient(const NLK_ARRAY *prediction, 
                                const size_t target, 
                                NLK_ARRAY *gradient)
{
    nlk_array_zero(gradient);
    gradient->data[target] = -log(prediction->data[target]);

    return NLK_SUCCESS;
}

/** @fn nlk_real nlk_neg_log_likelihood(const NLK_ARRAY *prediction, 
 *                                  const size_t target)
 * Negative Log Likelihood error for multiclass (1-of-K) classification.
 * (e.g. softmax)
 *
 * @param prediction    network output
 * @param target        target output (index)
 *
 * @return the network error
 *
 * @note
 * Does NOT assume log probabilities were passed.
 * @endnote
 */
nlk_real
nlk_negative_log_likelihood(const NLK_ARRAY *prediction, 
                            const size_t target)
{
    return -log(prediction->data[target]);
}
