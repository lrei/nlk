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


/** @file nlk_transfer.c
 * Linear Layer operations
 */

#include <stdlib.h>
#include <stdbool.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_transfer.h"



/**  
 * Sigmoid transfer function
 *
 * @param input             the input to the sigmoid transfer function
 * @param output            output of the layer (overwritten with the result)
 */
void
nlk_sigmoid_forward(const NLK_ARRAY *input, NLK_ARRAY *output)
{
    nlk_array_copy(output, input);
    nlk_sigmoid_array(output);
}

/** 
 * Calculates the gradient of the sigmoid
 *
 * @param output            the output of the sigmoid transfer forward step
 * @param grad_out          the gradient at the output of the sigmoid
 * @param grad_in           the gradient at the input of the sigmoid (result)
 *
 * @return NLK_SUCCESS or error code NLK_E*
 */
void
nlk_sigmoid_backprop(const NLK_ARRAY *output, const NLK_ARRAY *grad_out, 
                    NLK_ARRAY *grad_in)
{
    size_t ii;
    size_t len = output->rows * output->cols; 

    /* @TODO parallel: two cblas calls? */
    for(ii = 0; ii < len; ii++) {
        grad_in->data[ii] = grad_out->data[ii] * output->data[ii] *
                            (1.0 - output->data[ii]);
    }
}


/**
 * Softmax forward
 * defined as f_i(x) = exp(x_i)/a  where a = sum(exp(x_i)) for all i
 * this version uses scalling to prevent nasty overflows
 *
 * @note
 * untested, unused
 * @endnote
 */
void
nlk_softmax_forward(const NLK_ARRAY *input, NLK_ARRAY *output)
{
    size_t idx = input->rows * input->cols;
    nlk_real sum = 0;

    /* copy */
    nlk_array_copy(output, input);

    /* scale => all positve or 0 */
    nlk_array_rescale_max_minus(output);
    
    /* exp(-s) 
     * making the past steps equivalent to having done exp(x_i - max)
     */
    do {
        idx--;
        output->data[idx] = nlk_exp_minus_approx(-output->data[idx]);
        /* sum of exp(z_i) */
        sum += output->data[idx]; 
    } while(idx > 0);
    /* sum = nlk_array_sum(output); */
    
    /* divide */
    nlk_array_scale(1.0/sum, output); 

    NLK_ARRAY_CHECK_NAN(output, "output has NaNs");
}

/**
 * Softmax backprop
 *
 * @note
 * untested, unused
 * @endnote
 */
void
nlk_softmax_backprop(const NLK_ARRAY *output, const NLK_ARRAY *grad_out, 
                     NLK_ARRAY *grad_in)
{
    size_t idx = output->rows * output->cols;
    nlk_real sum = nlk_array_dot(grad_out, output, -1); /*@TODO fix the -1 */

     do {
        idx--;
        grad_in->data[idx] = output->data[idx] * (grad_out->data[idx] - sum);
    } while(idx > 0);


    NLK_ARRAY_CHECK_NAN(grad_in, "grad_in has NaNs");
}


/**
 * Log Softmax forward
 * it is defined as f_i(x) = log(1/a exp(x_i)), where a = sum_j exp(x_j) 
 *
 * @param input     input into the layer
 * @param output    log softmax of the input (result)
 *
 * @note
 *
 * @endnote
 */
void
nlk_log_softmax_forward(const NLK_ARRAY *input, NLK_ARRAY *output)
{
    nlk_real logsum = 0;
    nlk_real max;           /* holds the max value of the array */
    const size_t len = input->rows * input->cols;
    size_t idx = len;

    max =  nlk_array_max(input);
    /* calculate the logsum */
    do {
        idx--;
        logsum += nlk_exp_minus_approx(max - input->data[idx]);
    } while(idx > 0);   

    logsum = max + nlk_log_approx(logsum);

    do {
        output->data[idx] = input->data[idx] - logsum;
        idx++;
    } while(idx < len);

#ifdef CHECK_NANS
        if(nlk_array_has_nan(output)) {
            NLK_ERROR_VOID("output has NaNs", NLK_ENAN);
        }
#endif

}


/**
 * Log Softmax Backprop
 *
 * @param output    log softmax of the input during forward step
 * @param grad_out  the gradient at the output of the log_softmax
 * @param grad_in   the gradient at the input notes (result)
 */
void
nlk_log_softmax_backprop(const NLK_ARRAY *output, const NLK_ARRAY *grad_out, 
                         NLK_ARRAY *grad_in)
{
    size_t idx = grad_out->rows * grad_out->cols;
    nlk_real sum = nlk_array_sum(grad_out);

    do {
        idx--;
        grad_in->data[idx] = grad_out->data[idx] 
                             - (nlk_exp(output->data[idx]) * sum);
    } while(idx > 0);


    NLK_ARRAY_CHECK_NAN(grad_in, "NaN in gradient at input (result)");
}


/**
 * Rectifier - Rectified Linear Unit (ReLu) -  Forward
 */
void
nlk_rectifier_forward(const NLK_ARRAY *input, NLK_ARRAY *output)
{
    nlk_array_copy(output, input);
    nlk_array_rectify(output);
}


/**
 * Rectifier - Rectified Linear Unit (ReLu) - Backprop
 * @param output    log softmax of the input during forward step
 * @param grad_out  the gradient at the output of the rectifier
 * @param grad_in   the gradient at the input notes (result)
 */
void
nlk_rectifier_backprop(const NLK_ARRAY *output, const NLK_ARRAY *grad_out, 
                       NLK_ARRAY *grad_in)
{
    size_t idx = output->len;
    do {
        idx--;
        if(output->data[idx] <= 0) {
            grad_in->data[idx] = 0;
        } else {
            grad_in->data[idx] = grad_out->data[idx];
        }
    } while(idx > 0);


    
}
