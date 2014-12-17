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


/** @file array.c
 * Array functions
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <float.h>

#include <cblas.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "tinymt32.h"


tinymt32_t __tinymt;            /* random number generator state */
pthread_mutex_t __rng_mutex;    /* PRNG mutex */
int __seed = -1;                /* current seed */

/** @fn nlk_Array *nlk_array_create(const size_t rows, const size_t cols)
 * Create and allocate an nlk_Array
 *
 * @param rows      the number of rows
 * @param cols      the number of columns
 *
 * @return nlk_Array or NULL on error
 */
nlk_Array *
nlk_array_create(const size_t rows, const size_t cols)
{
    nlk_Array *array;
    int r;

    /* 0 dimensions are not allowed */
    if (rows == 0 || cols == 0) {
        NLK_ERROR_NULL("Array rows and column numbers must be non-zero "
                       "positive integers", NLK_EINVAL);
        /* unreachable */
    }

    /* allocate space for array struct */
    array = (nlk_Array *) malloc(sizeof(nlk_Array));
    if(array == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for array struct", 
                       NLK_ENOMEM);
        /* unreachable */
    }

    /* allocate space for data */
    r = posix_memalign((void **)&array->data, 128, 
                       rows * cols * sizeof(nlk_real));

    if(r != 0) {
        NLK_ERROR_NULL("failed to allocate memory for block data", NLK_ENOMEM);
        /* unreachable */
    }

    array->cols = cols;
    array->rows = rows;

    return array;
}

/** @fn nlk_Array *nlk_array_create_copy(const nlk_Array *array)
 * Create a copy of an array.
 *
 * @param source    array to copy
 *
 * @return the copy
 */
nlk_Array *
nlk_array_create_copy(const nlk_Array *source)
{
    nlk_Array *dest = nlk_array_create(source->rows, source->cols);
    if(dest == NULL) {
        NLK_ERROR_NULL("failed to create empty array", NLK_FAILURE);
        /* unreachable */
    }

    cblas_scopy(source->rows * source->cols, source->data, 1, dest->data, 1);

    return dest;
}


/** @fn int nlk_array_copy_row(nlk_Array *dest, const size_t dest_row, 
 *                            const nlk_Array *source, size_t source_row)
 * Copies a row from a source array row to a destination array row
 *
 * @param dest          the destination array
 * @param dest_row      the destination row index
 * @param source        the source array
 * @param source_row    the source array index
 *
 * @return NLK_SUCCESS or NLK_EINVAL or NLK_EBADLEN
 *
 * @note
 * If the number of columns in the destination array is larger than the source 
 * array, this function will copy up to source->cols without complaining.
 * If the number of columns in the destination array is smaller, this function
 * will fail with NLK_EBADLEN.
 * @endnote
 */
int
nlk_array_copy_row(nlk_Array *dest, const size_t dest_row, const 
                   nlk_Array *source, const size_t source_row)
{
    if(dest_row > dest->rows) {
        NLK_ERROR("Destination row out of range", NLK_EINVAL);
        /* unreachable */
    }
    if(source_row > source->rows) {
        NLK_ERROR("Source row out of range", NLK_EINVAL);
        /* unreachable */
    }
    if(source->cols > dest->cols) {
        NLK_ERROR("Destination array has a smaller number of columns than "
                  "the source array", NLK_EBADLEN);
        /* unreachable */
    }
    
    cblas_scopy(source->cols, &source->data[source_row * source->cols], 1, 
                &dest->data[dest_row * dest->cols], 1);
}

/** @fn int nlk_array_copy(nlk_Array *dest, const nlk_Array *source)
 * Copies an array from a source to a destination
 *
 * @param dest          the destination array
 * @param source        the source array
 *
 * @return NLK_SUCCESS or NLK_EBADLEN
 *
 * @note
 * Arrays must have the same dimensions
 * @endnote
 */

int 
nlk_array_copy(nlk_Array *dest, const nlk_Array *source)
{
    const size_t len = dest->rows * dest->cols;

    if(dest->rows != source->rows || dest->cols != source->cols) {
        NLK_ERROR("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }

    cblas_scopy(len, source->data, 1, dest->data, 1);

    return NLK_SUCCESS;
}

void
nlk_carray_copy_carray(nlk_real *dest, const nlk_real *source, size_t length)
{
    cblas_scopy(length, source, 1, dest, 1);
}


/** @fn void nlk_set_seed(const int seed)
 * Set the seed for the random number generation 
 *
 * @param seed the seed to set
 *
 * @return no return, global seed variable and random generator state 
 *                    are changed
 */
void 
nlk_set_seed(const int seed)
{
    __seed = seed;
    tinymt32_init(&__tinymt, seed);
}

/** @fn int nlk_get_seed()
 * Get the current seed for the random number generation 
 *
 * @return  the current seed
 */
int
nlk_get_seed()
{
    return __seed;
}

/** @fn uint32_t nlk_random_uint()
 * Returns a random unsigned 32bit integer
 *
 * @return a random unsigned 32bit integer
 */
uint32_t
nlk_random_uint()
{
    return tinymt32_generate_uint32(&__tinymt);
}

/** @fn nlk_array_init_wity_carray(nlk_Array *arr, const nlk_real *carr)
 * Initialize an array with the values from a C array
 */
void 
nlk_array_init_wity_carray(nlk_Array *arr, const nlk_real *carr)
{
    const size_t len = arr->rows * arr->cols;
    cblas_scopy(len, carr, 1, arr->data, 1);
}


/* nlk_init_uniform - initialize an array with numbers drawn from a uniform 
 *                    distribution in the [low, high) range
 * Params:
 *  array - the nlk_array to initialize
 *  low - the lower bound of the uniform random distribution
 *  high - the upper bound of the uniform random distribution
 *
 * Returns:
 *  No return (void), array data is initialized according to the distribution
 */
void 
nlk_array_init_uniform(nlk_Array *array,
                       const nlk_real low, const nlk_real high)
{
    size_t ii;
    const size_t length = array->rows * array->cols;
    nlk_real diff = high - low;

    for(ii = 0; ii < length; ii++) {
        array->data[ii] = low + diff * tinymt32_generate_float(&__tinymt);
    }
}


/** @fn void nlk_zero(nlk_Array *array)
 * Initialize an array with 0s
 *
 * @param array     the nlk_array to initialize
 *
 * @return no return (void); array data is zeroed.
 */
void
nlk_array_zero(nlk_Array *array)
{
    memset(array->data, 0, array->rows * array->cols * sizeof(nlk_real));
}

/** @fn
 * Elementwise comparison between the values of an array and a C array
 *
 * @param arr       the array
 * @param carr      the C array
 * @param tolerance the tolerance of the comparison for each value
 *
 * @return true if forall ii, abs(arr[ii] - carr[ii]) < tolerance, else false 
 */
bool
nlk_array_compare_carray(nlk_Array *arr, nlk_real *carr, nlk_real tolerance)
{
    const size_t len = arr->rows * arr->cols;
    return nlk_carray_compare_carray(arr->data, carr, len, tolerance);

}
bool
nlk_carray_compare_carray(nlk_real *carr1, nlk_real *carr2, size_t len,
                          nlk_real tolerance)
{
    for(size_t ii = 0; ii < len; ii++) {
        if(fabs(carr1[ii] - carr2[ii]) >= tolerance) {
            return false;
        }
    }
    return true;
}

/** @fn void nlk_print_array(const nlk_Array *array, const size_t row_limit, 
 *                           const size_t col_limit)
 *  Print an array
 */
void
nlk_print_array(const nlk_Array *array, const size_t row_limit, 
                const size_t col_limit)
{
    size_t rr;
    size_t rows;
    size_t cc;
    size_t cols;

    if(row_limit < array->rows) {
        rows = row_limit;
    } else {
        rows = array->rows;
    }
    if(col_limit < array->cols) {
        cols = col_limit;
    } else {
        cols = array->cols;
    }

    /* print header */
    printf("Array %zu x %zu:\n", array->rows, array->cols);
    /* print (limited) content */
    for(rr = 0; rr < rows; rr++) {
        for(cc = 0; cc < cols; cc++) {
            if(rows < array->rows && rr == rows - 1) {
                printf("... ");
            } else {
                printf("%.5f ", array->data[rr * array->cols + cc]);
            }
        }
        if(cols < array->cols) {
            printf("...");
        }
        printf("\n");
    }
}

/* nlk_array_free - free array */
void nlk_array_free(nlk_Array *array)
{
    free(array->data);
    free(array);
}

/** @fn void nlk_array_scale(const nlk_real scalar, nlk_Array *array)
 * Scale an array by *scalar*
 *
 * @param scalar    the scalar
 * @param array     the array to scale (overwritten)
 *
 * @return no return (void), array is overwritten
 */
void
nlk_array_scale(const nlk_real scalar, nlk_Array *array)
{
    size_t ii;
    size_t len = array->rows * array->cols;

    cblas_sscal(len, scalar, array->data, 1);
}

/** @fn void nlk_array_normalize_row_vectors(nlk_Array *m)
 * Normalizes the row vectors of a matrix
 *
 * @param m     the matrix
 *
 * @return no return, matrix is overwritten
 */
void
nlk_array_normalize_row_vectors(nlk_Array *m)
{
    size_t row;
    nlk_real len;

    for(row = 0; row < m->rows; row++) {
        len = cblas_snrm2(m->cols, &m->data[row * m->cols], 1);
        cblas_sscal(m->cols, 1.0/len, &m->data[row * m->cols], 1); 
    }
}

/** @fn void nlk_array_normalize_vectors(nlk_Array *v)
 * Normalizes a vector
 *
 * @param v     the vector
 *
 * @return no return, vector is overwritten
 */
void
nlk_array_normalize_vector(nlk_Array *v)
{
    nlk_real len;

    len = cblas_snrm2(v->rows, v->data, 1);
    cblas_sscal(v->rows, 1.0/len, v->data, 1); 
}

/** @fn nlk_real nlk_dot(nlk_Array *v1, nlk_Array *v2) 
 * Compute the dot product of two vectors
 */
nlk_real
nlk_array_dot(const nlk_Array *v1, nlk_Array *v2, uint8_t dim)
{
    if(dim == 0 && v1->rows != v2->rows) {
        NLK_ERROR("array dimensions (rows) do not match.", NLK_EBADLEN);
        /* unreachable */
    } else if(dim == 0) {
        return cblas_sdot(v1->rows, v1->data, 1, v2->data, 1);
    } else if(dim == 1 && v1->cols != v2->cols) {
        NLK_ERROR("array dimensions (cols) do not match.", NLK_EBADLEN);
        /* unreachable */
    } else if(dim == 1) {
        return cblas_sdot(v1->cols, v1->data, 1, v2->data, 1);
    } else {
        NLK_ERROR("invalid array dimension", NLK_EINVAL);
    }

}

/** @fn nlk_real nlk_dot_array_carray(nlk_Array *v1, nlk_real *carr) 
 * Compute the dot product a vector array and a c array
 */
nlk_real
nlk_array_dot_carray(const nlk_Array *v1, nlk_real *carr)
{
    return cblas_sdot(v1->rows, v1->data, 1, carr, 1);
}

/** @fn int nlk_array_add(const nlk_Array *a1, nlk_Array *a2)
 * Array (vector, matrix) element-wise addition
 * 
 * @param a1    array
 * @param a2    array, will be overwritten with the result
 *
 * @return NLK_SUCCESS or NLK_EBADLEN, overwittes a2
 */
int
nlk_array_add(const nlk_Array *a1, nlk_Array *a2)
{
    const size_t len = a1->rows * a1->cols;
    
    if(a1->cols != a2->cols || a1->rows != a2->rows) {
        NLK_ERROR("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }

    cblas_saxpy(len, 1, a1->data, 1, a2->data, 1); 

    return NLK_SUCCESS;
}

/** @fn int nlk_vector_add_row(const nlk_Array *v, nlk_Array *m, size_t row)
 * Adds a row vector to a matrix row
 * 
 * @param v     a column vector
 * @param m     a matrix
 * @param row   the matrix row, overwritten with the result
 *
 * @return NLK_SUCCESS, NLK_EBADLEN, NLK_EINVAL
 */
int
nlk_vector_add_row(const nlk_Array *v, nlk_Array *m, size_t row)
{
    if(v->rows != m->cols) {
        NLK_ERROR("vector rows do not match matrix columns.", NLK_EBADLEN);
        /* unreachable */
    }
    if(row > m->rows) {
        NLK_ERROR("row outside of matrix bounds.", NLK_EINVAL);
        /* unreachable */
    }

    cblas_saxpy(m->cols, 1, v->data, 1, &m->data[row * m->cols], 1); 

    return NLK_SUCCESS;
}

/** @fn int nlk_row_add_vector(const nlk_Array *m, nlk_Array *v, size_t row)
 * Adds a matrix row to a vector
 * 
 * @param m     a matrix
 * @param v     a column vector, overwitten with the result
 * @param row   the matrix row
 *
 * @return NLK_SUCCESS, NLK_EBADLEN, NLK_EINVAL
 */
int
nlk_row_add_vector(const nlk_Array *m, size_t row, nlk_Array *v)
{
    if(v->rows != m->cols) {
        NLK_ERROR("vector rows do not match matrix columns.", NLK_EBADLEN);
        /* unreachable */
    }
    if(row > m->rows) {
        NLK_ERROR("row outside of matrix bounds.", NLK_EINVAL);
        /* unreachable */
    }

    cblas_saxpy(m->cols, 1, &m->data[row * m->cols], 1, v->data, 1); 

    return NLK_SUCCESS;
}

/** @fn void nlk_array_add_carray(const nlk_Array *arr, nlk_real *carr)
 * Array element-wise addition addition with a C array
 * 
 * @param arr       array
 * @param carray    a C array, will be overwritten with the result
 *
 * @return no return, overwittes carray
 */
void
nlk_array_add_carray(const nlk_Array *arr, nlk_real *carr)
{
    const size_t len = arr->rows * arr->cols;
    
    cblas_saxpy(len, 1, arr->data, 1, carr, 1); 
}

/** @fn int nlk_array_mul(const nlk_Array *a1, nlk_Array *a2)
 *  Elementwise array multiplication
 *
 *  @param a1   first array
 *  @param a2   second array (overwritten)
 *
 *  @return NLK_EBADLEN or NLK_SUCCESS, a2 is overwritten
 */
int
nlk_array_mul(const nlk_Array *a1, nlk_Array *a2)
{
    const size_t len = a1->rows * a1->cols;
    
    if(a1->cols != a2->cols || a1->rows != a2->rows) {
        NLK_ERROR("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }

    /*#pragma omp parallel for*/
    for(size_t ii = 0; ii < len; ii++) {
        a2->data[ii] *= a1->data[ii];
    }

    return NLK_SUCCESS;
}
/** @fn nlk_real nlk_array_abs_sum(const nlk_Array *arr)
 * Sum of absolute array values (?asum)
 *
 * @param arr the array
 *
 * @return the sum of absolute array values
 */
nlk_real
nlk_array_abs_sum(const nlk_Array *arr)
{
    const size_t len = arr->rows * arr->cols;
    return cblas_sasum(len, arr->data, 1);
}

nlk_real
nlk_carray_abs_sum(const nlk_real *carr, size_t length)
{
    return cblas_sasum(length, carr, 1);
}

/** @fn size_t nlk_array_non_zero(const nlk_Array *arr)
 * Count the number of non-zero elements in the array
 *
 * @param arr the array
 *
 * @return number of non-zero elements in the array
 */
size_t
nlk_array_non_zero(const nlk_Array *arr)
{
    size_t ii = 0;
    size_t non_zero = 0;
    const size_t len = arr->rows * arr->cols;

    for(ii = 0; ii < len; ii++) {
        if(arr->data[ii] != 0.0) {
            non_zero += 1;
        }
    }
    return non_zero;
}

/** @fn int nlk_add_scaled_vectors(const nlk_real s, const nlk_Array *v1,
 *                                 nlk_Array *v2)
 * Scaled Vector addition (?axpy): a2  = s * a1 + a2
 * 
 * @param s     the scalar
 * @param a1    a vector
 * @paran a2    a vector, will be overwritten with the result
 *
 * @return NLK_SUCCESS on success NLK_E on failure; result overwrittes a2.
 */
int
nlk_add_scaled_vectors(const nlk_real s, const nlk_Array *v1, nlk_Array *v2)
{
    const size_t len = v1->rows * v1->cols;
    size_t ii;
    
    if(v1->cols != v2->cols || v1->rows != v2->rows) {
        NLK_ERROR("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }
    if(v1->cols != 1) {
      NLK_ERROR("Arrays must be (row) vectors.", NLK_EBADLEN);
        /* unreachable */
    }

    cblas_saxpy(v1->rows, s, v1->data, 1,
                v2->data, 1);

    return NLK_SUCCESS;
}

/* nlk_vector_transposed_multiply_add - Multiplies vector a1 by the transpose 
 *                                  of vector a2, then adds matrix a3. 
 *                                  m = v1 * v2' + m 
 *                                  (?ger)
 *                                  
 * Params:
 *  v1 [m] - vector
 *  v2 [n] - vector, will be transposed
 *  m [m][n]- matrix, will be overwritten
 *
 * Returns:
 *  NLK_SUCCESS on success, otherwise an NLK_E*
 *  m will be overwritten with result
 *
 * Usage:
 *  In a linear layer, the weights [layer_size][input_size] are updated by
 *      
 *      nlk_vector_transposed_multiply_add(grad_out, input, weights)
 *
 *  Where grad_out [layer_size] is the gradient at the layer's output and 
 *  input [input_size] is the input to the layer in that step.
 */
int
nlk_vector_transposed_multiply_add(const nlk_Array *v1, const nlk_Array *v2, 
                                   nlk_Array *m)
{
    if(v1->rows != m->rows) {
        NLK_ERROR("vector v1 length must be equal to number of matrix rows", 
                  NLK_EBADLEN);
        /* unreachable */
    }
    if(v2->rows != m->cols) {
        NLK_ERROR("vector v2 length must be equal to number of matrix columns",
                  NLK_EBADLEN);
        /* unreachable */
    }

     cblas_sger(CblasRowMajor, m->rows, m->cols, 1, v1->data, 
                1, v2->data, 1, m->data, m->cols);
     return NLK_SUCCESS;
}

/* nlk_matrix_vector_multiply_add - Multiplies a matrix by a vector 
 *                                  v2 = m * v1 + v2 or v2 = m' * v1 + v2
 * Params:
 *  m [n][m] - the matrix
 *  trans - NLK_TRANSPOSE or NLK_NOTRANSPOSE
 *  v1 [n] if transpose, v [m] otherwise - the vector multiplied by the matrix
 *  v2 [m] if transpose, result [n] otherwise - result vector
 *
 *  Returns:
 *    NLK_SUCCESS or NLK_E* on error
 *    result is overwritten
 *
 * Usage:
 *  In a linear layer, the gradient at the layer's input is updated by calling
 *
 *  nlk_matrix_vector_multiply_add(weights, NLK_TRANSPOSE, grad_out, grad_in)
 *
 *  Where grad_out is the gradient at the layer's output and grad_in is the 
 *  gradient at the layer's input
 */
int 
nlk_matrix_vector_multiply_add(const nlk_Array *m, const NLK_OPTS trans, 
                               const nlk_Array *v1, nlk_Array *v2)
{
     /* First we need to make sure all dimensions make sense */
    if(trans == NLK_TRANSPOSE && v1->rows != m->rows) {
        NLK_ERROR("vector (v1) lenght must be equal to the number of "
                  "matrix rows", NLK_EBADLEN);
    }
    if(trans == NLK_NOTRANSPOSE && v1->rows != m->cols) {
        NLK_ERROR("vector (v1) lenght must be equal to the number of "
                  "matrix columns", NLK_EBADLEN);
    }
    if(trans == NLK_TRANSPOSE && v2->rows != m->cols) {
        NLK_ERROR("result vector (v2) length must be equal to the number of "
                  "matrix columns", NLK_EBADLEN);
        /* unreachable */
    }
    if(trans == NLK_NOTRANSPOSE && v2->rows != m->rows) {
        NLK_ERROR("result vector (v2) length must be equal to the number of "
                  "matrix rows", NLK_EBADLEN);
        /* unreachable */
    }

    cblas_sgemv(CblasRowMajor, trans,
                m->rows, m->cols, 1, m->data, m->cols,
                v1->data, 1, 1, v2->data, 1); 

    return NLK_SUCCESS;
}

/** @fn void nlk_init_sigmoid(nlk_Array *array, const uint8_t max_exp)
* Initialize an array with the values of the sigmoid between 
* [-max_exp, max_exp] split evenly into the number of elements in the array.
*
* @param array     the nlk_array to initialize
* @param max_exp   function domain will be [-max_exp, max_exp]
*
* @return no return (void); array data is initialized accordingly.
*
* @note
* Intended for use within a sigmoid table.
* @endnote
*/
void
nlk_array_init_sigmoid(nlk_Array *array, const uint8_t max_exp) {
    size_t size = array->rows * array->cols;

    /* this splits the range [sigma(-max), sigma(max)] into *size* pieces */
    /*#pragma omp parallel for*/
    for(size_t ii = 0; ii < size; ii++) {
        array->data[ii] = exp(((nlk_real) ii / (nlk_real) size * 2 - 1) 
                              * max_exp);
        array->data[ii] = array->data[ii] / (array->data[ii] + 1);
    }
}


/** @fn nlk_Table *nlk_table_sigmoid_create(size_t size, uint8_t max_exp)
 * Create a sigmoid table for computing 1/(exp(-x) + 1)
 * 
 * @param size      table size
 * @param max_exp   table range is [-max_exp, max_exp]
 *
 * @return  returns the sigmoid table or NULL
 *
 * @note
 * Learned this little performance trick from word2vec.
 * Another trick (not used) is Leon Bottou approx exp(-x) in Torch7
 * @endnote
 */
nlk_Table *
nlk_table_sigmoid_create(const size_t table_size, const nlk_real max_exp)
{
    nlk_Table *table;
    size_t ii;

    /* allocate structure */
    table = (nlk_Table *) malloc(sizeof(nlk_Table));
    if(table == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for table struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    /* allocate array and set fields */
    table->table = nlk_array_create(table_size, 1);
    if(table->table == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for table array",
                       NLK_ENOMEM);
        /* unreachable */
    }
    table->size = table_size;
    table->max =  max_exp;
    table->min = -max_exp;

    /* precompute the values */
    nlk_array_init_sigmoid(table->table, max_exp);

    return table;
}

/**
 * Thread safe random pool creation (positive integers).
 *
 * @param table_size    the number of random numbers to generate
 *
 * @return the table (random pool)
 *
 * @note
 * Values will be stored in nlk_reals and this should be limited to the type's
 * range. 
 * @endnote
 */
nlk_Table *
nlk_random_pool_create(size_t table_size, const uint32_t max)
{
    nlk_Table *table;
    size_t ii;
    uint32_t r;

    /* allocate structure */
    table = (nlk_Table *) malloc(sizeof(nlk_Table));
    if(table == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for table struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    /* allocate array and set fields */
    table->table = nlk_array_create(table_size, 1);
    if(table->table == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for table array",
                       NLK_ENOMEM);
        /* unreachable */
    }
    table->max = max;
    table->size = table_size;
    table->pos = 0;

    /* generate the numbers */
    pthread_mutex_lock(&__rng_mutex);
    for(ii = 0; ii < table->size; ii++) {
        r = tinymt32_generate_uint32(&__tinymt) % (uint32_t) table->max;
        table->table->data[ii] = r;
    }
    pthread_mutex_unlock(&__rng_mutex);

    return table;
}

void
nlk_random_pool_reset(nlk_Table *table)
{
    size_t ii;
    uint32_t r;
    table->pos = 0;

    /* generate the numbers */
    pthread_mutex_lock(&__rng_mutex);
    for(ii = 0; ii < table->size; ii++) {
        r = tinymt32_generate_uint32(&__tinymt) % (uint32_t) table->max;
        table->table->data[ii] = r;
    }
    pthread_mutex_unlock(&__rng_mutex);
}

uint32_t
nlk_random_pool_get(nlk_Table *table)
{
    if(table->pos >= table->size) {
        nlk_random_pool_reset(table);
    }
    table->pos += 1;

    return (uint32_t) table->table->data[table->pos - 1];
}

/** @fn nlk_table_free(nlk_Table *table)
 * Free a table.
 *
 * @param table the table to free
 */
void
nlk_table_free(nlk_Table *table)
{
    nlk_array_free(table->table);
    free(table);
}

/** @fn nlk_real nlk_sigmoid_table(const nlk_Table *sigmoid_table, 
 *                                 const nlk_real x) 
 * Calculates the sigmoid 1/(exp(-x) + 1) for a real valued x.
 * 
 * @param sigmoid_table     a precomputed sigmoid table
 * @param x                 the real valued x for calculating its sigmoid
 *
 * @return the sigmoid of x
 */
nlk_real
nlk_sigmoid_table(const nlk_Table *sigmoid_table, const double x) 
{
    int idx;

    if(sigmoid_table == NULL) {
        return 1.0 / (1.0 + exp(-x));
    }

    if(x >= sigmoid_table->max) {
        return 1;
    } else if(x <= sigmoid_table->min) {
        return 0;
    }

    idx = ((x + sigmoid_table->max) * 
           ((double) sigmoid_table->size / sigmoid_table->max / 2.0));

    return sigmoid_table->table->data[idx];
}

/** @fn nlk_array_sigmoid_table(const nlk_Table *sigmoid_table, 
 *                              const nlk_Array *input, nlk_Array *output)
 * Calculates the elemtwise sigmoid for an entire array
 *
 * @param sigmoid_table     a precomputed sigmoid table
 * @param arr            the in/output array (overwritten with the result)
 *
 * @return NLK_SUCCESS or NLK_E* on error
 */
int
nlk_array_sigmoid_table(const nlk_Table *sigmoid_table, nlk_Array *arr)
{
    const size_t len = arr->rows * arr->cols;


    /*#pragma omp parallel for*/
    for(size_t ii = 0; ii < len; ii++) {
        size_t idx;
        if(arr->data[ii] >= sigmoid_table->max) {
            arr->data[ii] = 1;
        } else if(arr->data[ii] <= -sigmoid_table->max) {
            arr->data[ii] = 0;
        }
        else {
            idx = (arr->data[ii] + sigmoid_table->max) * 
                  (sigmoid_table->size / sigmoid_table->max / 2);
            arr->data[ii] = sigmoid_table->table->data[idx];

        }
    }

    return NLK_SUCCESS;
}

/** @fn nlk_array_log(const nlk_Array *input, nlk_Array *output)
 * Calculates the elemtwise logarithm for an entire array
 *
 * @param input             the input array
 * @param output            the output array (overwritten with the result)
 *
 * @return NLK_SUCCESS or NLK_E* on error
 */
int
nlk_array_log(const nlk_Array *input, nlk_Array *output)
{
    const size_t len = input->rows * input->cols;

    if(input->cols != output->cols || input->rows != output->rows) {
        NLK_ERROR("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }

    /*#pragma omp parallel for*/
    for(size_t ii = 0; ii < len; ii++) {
        output->data[ii] = log(input->data[ii]);
    }

    return NLK_SUCCESS;
}

/** @fn nlk_exp_minus
 * Calculates exp(-x) via Leon Bottou's/Torch7 approximate function or
 * standard exp(-x) function if EXACT_EXPONENTIAL was defined.
 * @note
 * x should be positive.
 * @endnote
 */
nlk_real nlk_exp_minus(nlk_real x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
    return exp(-x);
#else
 /* fast approximation of exp(-x) for x positive */
#define A0 (1.0)
#define A1 (0.125)
#define A2 (0.0078125)
#define A3 (0.00032552083)
#define A4 (1.0172526e-5)
    double y;
    if (x < 13.0) {
        y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
        y *= y;
        y *= y;
        y *= y;
        y = 1/y;
        return (nlk_real) y;
    } 
    return 0;
#undef A0
#undef A1
#undef A2
#undef A3
#undef A4
#endif
}

void 
nlk_array_sigmoid_approx(nlk_Array *arr)
{
    const size_t len = arr->rows * arr->cols;

#pragma omp parallel for
    for(size_t ii = 0; ii < len; ii++) {
        arr->data[ii] = 1.0 / 1.0 + nlk_exp_minus(arr->data[ii]);
    }


}
