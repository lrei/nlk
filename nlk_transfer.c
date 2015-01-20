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


/** @fn void nlk_transfer_concat_forward(NLK_ARRAY *concat_view, 
 *                                       const NLK_ARRAY *source)
 * Concatenated view of all rows in a matrix into a single vector.
 *
 * @param input        the array (matrix) containing the data
 * @param concat_view  an array view (destination of the concat operation)
 *
 * @note
 * This function "performs" a concatenation, given a 2D array (matrix)
 *@code
 *      [row 1,
 *       row 2,
 *       ...
 *       row n]
 * @endcode
 *
 * This function changes concat_view to a 1D array (vector)
 * @code
 *      [ row 1, row 2, ...,  row n]
 * @endcode
 * @endnote
 */
void
nlk_transfer_concat_forward(const NLK_ARRAY *input, NLK_ARRAY *concat)
{
    const size_t len = input->rows * input->cols;
    nlk_carray_copy_carray(concat->data, input->data, len);
}

/** @fn void nlk_transfer_concat_backprop(NLK_ARRAY *grad_int, 
 *                                        const NLK_ARRAY *grad_out)
 * Backwards pass for a concatenation operation. 
 *
 * @param grad_out      gradient at the output of the function
 * @param grad_in_view  gradient at the input of the function
 *
 * grad_in_view is overwritten.
 */ 
void
nlk_transfer_concat_backprop(const NLK_ARRAY *grad_out, 
                             NLK_ARRAY *grad_in)
{
    size_t ii;
    for(ii = 0; ii < grad_in->rows; ii++) {
        nlk_carray_copy_carray(&grad_in->data[ii * grad_in->cols],
                               &grad_out->data[ii * grad_in->cols],
                               grad_in->cols);
    }
}



/** @fn void nlk_sigmoid_forward(const NLK_TABLE *sigmoid_table, 
 *                               const NLK_ARRAY *input,  NLK_ARRAY *output)
 * Sigmoid transfer function
 *
 * @param sigmoid_table     precomputed sigmoid table
 * @param input             the input to the sigmoid transfer function
 * @param output            output of the layer (overwritten with the result)
 *
 * @return NLK_SUCCESS or error code NLK_E*
 */
void
nlk_sigmoid_forward_table(const NLK_TABLE *sigmoid_table, NLK_ARRAY *input)
{
    nlk_array_sigmoid_table(sigmoid_table, input);
}

/** @fn nlk_sigmoid_backprop
 * Calculates the gradient of the sigmoid
 *
 * @param output            the output of the sigmoid transfer forward step
 * @param grad_out          the gradient at the output of the sigmoid
 * @param grad_in           the gradient at the input of the sigmoid (result)
 *
 * @return NLK_SUCCESS or error code NLK_E*
 */
int
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

    return NLK_SUCCESS;
}

/** @fn nlk_average(const NLK_ARRAY *input, NLK_ARRAY *output)
 * Averaging layer: takes matrix where each row represents an input and
 * averages the rows, writting it to the output.
 *
 * @param input     the input matrix
 * @param n_rows    the number of rows to average (first n_rows)
 * @param output    the output array (overwritten)
 *
 */
void
nlk_average(const NLK_ARRAY *input, size_t n_rows, NLK_ARRAY *output) 
{
    size_t ii;

    if(input->cols != output->rows) {
        NLK_ERROR_VOID("number of input matrix columns must be equal to the "
                       "number of output rows", NLK_EBADLEN);
        /* unreachable */
    }

    nlk_array_zero(output);

    /* add the rows */
    for(ii = 0; ii < n_rows; ii++) {
        nlk_row_add_vector(input, ii, output);
    }

    /* average */
    nlk_array_scale(1.0/(nlk_real) n_rows, output);
}
