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
#include <float.h>

#include <cblas.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "tinymt32.h"


/**
 * Create and allocate an NLK_ARRAY
 *
 * @param rows      the number of rows
 * @param cols      the number of columns
 *
 * @return NLK_ARRAY or NULL on error
 */
NLK_ARRAY *
nlk_array_create(const size_t rows, const size_t cols)
{
    NLK_ARRAY *array;
    int r;

    /* 0 dimensions are not allowed */
    if (rows == 0 || cols == 0) {
        NLK_ERROR_NULL("Array rows and column numbers must be non-zero "
                       "positive integers", NLK_EINVAL);
        /* unreachable */
    }

    /* allocate space for array struct */
    array = (NLK_ARRAY *) malloc(sizeof(NLK_ARRAY));
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

/**
 * "Resizes" an array. In reality, creates a new array and copies the contents
 * of the old array. If creation fails returns NULL and the original array
 * remains intact. If creation succeeds the old array is freed and the pointer
 * to it is invalidated.
 *
 * @param old   the array to resize
 * @param rows  the new number of rows
 * @param cols  the new number of columns
 *
 * @return the resized array
 *
 * @note
 * If new size is smaller than the old, only elements up to new length are 
 * copied. If new size is larger, the new values are NOT initialized.
 * @endnote
 */
NLK_ARRAY *
nlk_array_resize(NLK_ARRAY *old, const size_t rows, const size_t cols)
{
    size_t row_limit;
    size_t col_limit;

    NLK_ARRAY *array = nlk_array_create(rows, cols);
    if(array == NULL) {
        return NULL;
    }
    
    /* determine bounds for copy operation */
    if(old->rows < rows) {
        row_limit = old->rows;
    } else {
        row_limit = rows;
    }
    if(old->cols < cols) {
        col_limit = old->cols;
    } else {
        col_limit = cols;
    }
    
    for(size_t rr = 0; rr < row_limit; rr++) {
        cblas_scopy(col_limit, &old->data[rr * col_limit], 1, 
                    &array->data[rr * col_limit], 1);
    }

    nlk_array_free(old);
    old = NULL;

    return array;
}

/**
 * Create a copy of an array.
 *
 * @param source    array to copy
 *
 * @return the copy
 */
NLK_ARRAY *
nlk_array_create_copy(const NLK_ARRAY *source)
{
    NLK_ARRAY *dest = nlk_array_create(source->rows, source->cols);
    if(dest == NULL) {
        NLK_ERROR_NULL("failed to create empty array", NLK_FAILURE);
        /* unreachable */
    }

    cblas_scopy(source->rows * source->cols, source->data, 1, dest->data, 1);

    return dest;
}


/**
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
nlk_array_copy_row(NLK_ARRAY *dest, const size_t dest_row, const 
                   NLK_ARRAY *source, const size_t source_row)
{
#ifndef NCHECKS
    if(dest_row > dest->rows) {
        NLK_ERROR("Destination row out of range", NLK_EINVAL);
        /* unreachable */
    }
    if(source_row > source->rows) {
        NLK_ERROR("Source row out of range", NLK_EINVAL);
        /* unreachable */
    }
    if(source->cols > dest->cols) {
        NLK_ERROR("Destination array has a smaller number of columns "
                  "than the source array", NLK_EBADLEN);
        /* unreachable */
    }
#endif
    
    cblas_scopy(source->cols, &source->data[source_row * source->cols], 1, 
                &dest->data[dest_row * dest->cols], 1);

    return NLK_SUCCESS;
}

/**
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

void
nlk_array_copy(NLK_ARRAY *dest, const NLK_ARRAY *source)
{
    const size_t len = dest->rows * dest->cols;

#ifndef NCHECKS
    if(dest->rows != source->rows || dest->cols != source->cols) {
        NLK_ERROR_VOID("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }
#endif

    cblas_scopy(len, source->data, 1, dest->data, 1);
}

void
nlk_carray_copy_carray(nlk_real *dest, const nlk_real *source, size_t length)
{
    cblas_scopy(length, source, 1, dest, 1);
}

/**
 * Returns a random unsigned 32bit integer
 *
 * @return a random unsigned 32bit integer
 */
uint32_t
nlk_random_uint(tinymt32_t *rng)
{
    return tinymt32_generate_uint32(rng);
}

/**
 * Mikolov's random number generator function (16 bit unsigned int)
 * @param next_random   I/O variable that contains (a pointer to) the previous 
 *                      random number when the unction is called and the next 
 *                      random number when the function returns
 */
void
nlk_random_cheap(size_t *next_random) 
{
    *next_random = *next_random * (unsigned long long)25214903917 + 11;
}

/**
 * Initialize an array with the values from a C array
 */
void 
nlk_array_init_wity_carray(NLK_ARRAY *arr, const nlk_real *carr)
{
    const size_t len = arr->rows * arr->cols;
    cblas_scopy(len, carr, 1, arr->data, 1);
}


/**
 * Initialize an array with numbers drawn from a uniform distribution in the 
 * [low, high) range.
 *
 * @params array    the nlk_array to initialize
 * @param low       the lower bound of the uniform random distribution
 * @param high      the upper bound of the uniform random distribution
 * @param rng       the random number generator
 */
void 
nlk_array_init_uniform(NLK_ARRAY *array, const nlk_real low, 
                       const nlk_real high, tinymt32_t *rng)
{
    size_t ii;
    const size_t length = array->rows * array->cols;
    nlk_real diff = high - low;

    for(ii = 0; ii < length; ii++) {
        array->data[ii] = low + diff * tinymt32_generate_float(rng);
    }
}

/**
 * Initialize a C array with numbers drawn from a uniform distribution in the 
 * [low, high) range.
 *
 * @params carr     the array to initialize
 * @param low       the lower bound of the uniform random distribution
 * @param high      the upper bound of the uniform random distribution
 * @param length    the length of the C array to be initialized
 */

void 
nlk_carray_init_uniform(nlk_real *carr, const nlk_real low, 
                        const nlk_real high, size_t length, tinymt32_t *rng)
{
    size_t ii;
    nlk_real diff = high - low;

    for(ii = 0; ii < length; ii++) {
        carr[ii] = low + diff * tinymt32_generate_float(rng);
    }
}




/**
 * Initialize an array with 0s
 *
 * @param array     the nlk_array to initialize
 *
 * @return no return (void); array data is zeroed.
 */
void
nlk_array_zero(NLK_ARRAY *array)
{
    memset(array->data, 0, array->rows * array->cols * sizeof(nlk_real));
}

/**
 * Elementwise comparison between the values of an array and a C array
 *
 * @param arr       the array
 * @param carr      the C array
 * @param tolerance the tolerance of the comparison for each value
 *
 * @return true if forall ii, abs(arr[ii] - carr[ii]) < tolerance, else false 
 */
bool
nlk_array_compare_carray(NLK_ARRAY *arr, nlk_real *carr, nlk_real tolerance)
{
    const size_t len = arr->rows * arr->cols;
    return nlk_carray_compare_carray(arr->data, carr, len, tolerance);

}

/**
 * Elementwise comparison between the values of two C arrays
 *
 * @param carr1     a C array
 * @param carr2     another C array
 * @param tolerance the tolerance of the comparison for each value
 *
 * @return  true if forall ii, abs(carr1[ii] - carr2[ii]) < tolerance,
 *          else false
 */
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

/**
 *  Print an array (via printf)
 *
 *  @param array        the array to print
 *  @param row_limit    the maximum number of rows to print
 *  @param col_limit    the maximum number of columns to print
 */
void
nlk_print_array(const NLK_ARRAY *array, const size_t row_limit, 
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

/**
 *  Save an array to a file pointer
 *
 *  @param array        the array to save
 *  @param fp           the file pointer
 */
void
nlk_array_save(NLK_ARRAY *array, FILE *fp)
{
    const size_t len = array->cols * array->rows;

    /* write header */
    fprintf(fp, "%zu %zu\n", array->rows, array->cols);

    /* write data */
    fwrite(array->data, sizeof(nlk_real), len, fp);
}

/**
 *  Load an array from a file pointer
 *
 *  @param fp           the file pointer
 *
 *  @return the array or NULL
 */
NLK_ARRAY *
nlk_array_load(FILE *fp)
{
    size_t rows;
    size_t cols;
    size_t ret;
    size_t len;
    NLK_ARRAY *array;

    /* read header */
    ret = fscanf(fp, "%zu", &rows);
    if(ret <= 0) {
        goto nlk_array_load_err;
    }
    ret = fscanf(fp, "%zu", &cols);
    if(ret <= 0) {
        goto nlk_array_load_err;
    }
    ret = fgetc(fp); /* the newline */
    if(ret <= 0) {
        goto nlk_array_load_err;
    }

    array = nlk_array_create(rows, cols);
    len = array->cols * array->rows;

    /* read data */
    ret = fread(array->data, sizeof(nlk_real), len, fp);
    if(ret != len) {
        NLK_ERROR_NULL("read length does not match expected length",
                       NLK_FAILURE);
        /* unreachable */
    }

    return array;

    /* generic header error */
nlk_array_load_err:
    NLK_ERROR_NULL("unable to read header information", NLK_FAILURE);
}


/**
 * Free the memory of an array 
 *
 * @param array the array to free
 */
void 
nlk_array_free(NLK_ARRAY *array)
{
    if(array != NULL) {
        free(array->data);
        free(array);
        array = NULL;
    } else {
        NLK_ERROR_VOID("Double free attempt", NLK_FAILURE);
    }
}

/**
 * Scale an array by *scalar*
 *
 * @param scalar    the scalar
 * @param array     the array to scale (overwritten)
 *
 * @return no return (void), array is overwritten
 */
void
nlk_array_scale(const nlk_real scalar, NLK_ARRAY *array)
{
    size_t len = array->rows * array->cols;

    cblas_sscal(len, scalar, array->data, 1);
}

/**
 * Normalizes the row vectors of a matrix
 *
 * @param m     the matrix
 *
 * @return no return, matrix is overwritten
 */
void
nlk_array_normalize_row_vectors(NLK_ARRAY *m)
{
    size_t row;
    nlk_real len;

    for(row = 0; row < m->rows; row++) {
        len = cblas_snrm2(m->cols, &m->data[row * m->cols], 1);
        cblas_sscal(m->cols, 1.0/len, &m->data[row * m->cols], 1); 
    }
}

/**
 * Normalizes a vector
 *
 * @param v     the vector
 *
 * @return no return, vector is overwritten
 */
void
nlk_array_normalize_vector(NLK_ARRAY *v)
{
    const nlk_real len = cblas_snrm2(v->rows, v->data, 1);

    cblas_sscal(v->rows, 1.0/len, v->data, 1); 
}

/**
 * Compute the dot product of two vectors
 *
 * @param v1    the first vector
 * @param v2    the second vector
 * @param dim   0 (for row vectors) or 1 (for column vectors)
 */
nlk_real
nlk_array_dot(const NLK_ARRAY *v1, NLK_ARRAY *v2, uint8_t dim)
{
#ifndef NCHECKS
    if(dim == 0 && v1->rows != v2->rows) {
        NLK_ERROR("array dimensions (rows) do not match.", NLK_EBADLEN);
        /* unreachable */
    } else 
#endif
        if(dim == 0) {
        return cblas_sdot(v1->rows, v1->data, 1, v2->data, 1);
#ifndef NCHECKS
    } else if(dim == 1 && v1->cols != v2->cols) {
        NLK_ERROR("array dimensions (cols) do not match.", NLK_EBADLEN);
        /* unreachable */
#endif
    } else if(dim == 1) {
        return cblas_sdot(v1->cols, v1->data, 1, v2->data, 1);
    } else {
        NLK_ERROR("invalid array dimension", NLK_EINVAL);
        /* unreachable */
    }
}

/**
 * Compute the dot product of rows of a different matrices
 *
 * @param m1    a matrix
 * @param row1  the row of m1 to use
 * @param m2    another matrix
 * @param row2  the row of m2 to use
 *
 * @return the dot product of the matrix rows
 */
nlk_real
nlk_array_row_dot(const NLK_ARRAY *m1, size_t row1, NLK_ARRAY *m2, size_t row2)
{
#ifndef NCHECKS
    if(m1->cols != m2->cols) {
        NLK_ERROR("array dimensions (columns) do not match.", NLK_EBADLEN);
        /* unreachable */
    }
#endif
    return cblas_sdot(m1->cols, &m1->data[row1 * m1->cols], 1, 
                      &m2->data[row2 * m2->cols], 1);
}

/**
 * Compute the dot product a vector array and a c array
 * 
 * @param v1    an array
 * @param carr  a C array
 *
 * @return the dot product
 */
nlk_real
nlk_array_dot_carray(const NLK_ARRAY *v1, nlk_real *carr)
{
    return cblas_sdot(v1->rows, v1->data, 1, carr, 1);
}

/**
 * Array (vector, matrix) element-wise addition
 * 
 * @param a1    array
 * @param a2    array, will be overwritten with the result
 */
void
nlk_array_add(const NLK_ARRAY *a1, NLK_ARRAY *a2)
{
#ifndef NCHECKS
    if(a1->cols != a2->cols || a1->rows != a2->rows) {
        NLK_ERROR_VOID("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }
#endif

    cblas_saxpy(a1->rows * a1->cols, 1, a1->data, 1, a2->data, 1); 
}

/**
 * Adds a row vector to a matrix row
 * 
 * @param v     a column vector
 * @param m     a matrix
 * @param row   the matrix row, overwritten with the result
 */
void
nlk_vector_add_row(const NLK_ARRAY *v, NLK_ARRAY *m, size_t row)
{
#ifndef NCHECKS
    if(v->rows != m->cols) {
        NLK_ERROR_VOID("vector rows do not match matrix columns.", 
                       NLK_EBADLEN);
        /* unreachable */
    }
    if(row > m->rows) {
        NLK_ERROR_VOID("row outside of matrix bounds.", NLK_EINVAL);
        /* unreachable */
    }
#endif

    cblas_saxpy(m->cols, 1, v->data, 1, &m->data[row * m->cols], 1); 
}

/**
 * Adds a matrix row to a vector
 * 
 * @param m     a matrix
 * @param v     a column vector, overwitten with the result
 * @param row   the matrix row
 */
void
nlk_row_add_vector(const NLK_ARRAY *m, size_t row, NLK_ARRAY *v)
{
#ifndef NCHECKS
    if(v->rows != m->cols) {
        NLK_ERROR_VOID("vector rows do not match matrix columns.", 
                       NLK_EBADLEN);
        /* unreachable */
    }
    if(row > m->rows) {
        NLK_ERROR_VOID("row outside of matrix bounds.", NLK_EINVAL);
        /* unreachable */
    }
#endif

    cblas_saxpy(m->cols, 1, &m->data[row * m->cols], 1, v->data, 1); 
}

/**
 * Array element-wise addition addition with a C array
 * 
 * @param arr       array
 * @param carray    a C array, will be overwritten with the result
 *
 * @return no return, overwittes carray
 */
void
nlk_array_add_carray(const NLK_ARRAY *arr, nlk_real *carr)
{
    cblas_saxpy(arr->rows * arr->cols, 1, arr->data, 1, carr, 1); 
}

/**
 *  Elementwise array multiplication (NON-PARALLEL)
 *
 *  @param a1   first array
 *  @param a2   second array (overwritten)
 */
void
nlk_array_mul(const NLK_ARRAY *a1, NLK_ARRAY *a2)
{
    const size_t len = a1->rows * a1->cols;
    
#ifndef NCHECKS
    if(a1->cols != a2->cols || a1->rows != a2->rows) {
        NLK_ERROR_VOID("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }
#endif

    /*#pragma omp parallel for*/
    for(size_t ii = 0; ii < len; ii++) {
        a2->data[ii] *= a1->data[ii];
    }

}
/**
 * Sum of absolute array values (?asum)
 *
 * @param arr the array
 *
 * @return the sum of absolute array values
 */
nlk_real
nlk_array_abs_sum(const NLK_ARRAY *arr)
{
    return cblas_sasum(arr->rows * arr->cols, arr->data, 1);
}

/**
 * Sum of absolute C array values (?asum)
 *
 * @param arr the array
 *
 * @return the sum of absolute array values
 */
nlk_real
nlk_carray_abs_sum(const nlk_real *carr, size_t length)
{
    return cblas_sasum(length, carr, 1);
}

/**
 * Count the number of non-zero elements in the array
 *
 * @param arr the array
 *
 * @return number of non-zero elements in the array
 */
size_t
nlk_array_non_zero(const NLK_ARRAY *arr)
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

/**
 * Scaled Vector addition (?axpy): a2  = s * a1 + a2
 * 
 * @param s     the scalar
 * @param a1    a vector
 * @paran a2    a vector, will be overwritten with the result
 *
 * @return NLK_SUCCESS on success NLK_E on failure; result overwrittes a2.
 */
void
nlk_add_scaled_vectors(const nlk_real s, const NLK_ARRAY *v1, NLK_ARRAY *v2)
{
    
#ifndef NCHECKS
    if(v1->cols != v2->cols || v1->rows != v2->rows) {
        NLK_ERROR_VOID("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }
    if(v1->cols != 1) {
      NLK_ERROR_VOID("Arrays must be (row) vectors.", NLK_EBADLEN);
        /* unreachable */
    }
#endif

    cblas_saxpy(v1->rows, s, v1->data, 1, v2->data, 1);
}

/**
 * Multiplies vector a1 by the transpose of vector a2, then adds matrix a3. 
 * m = v1 * v2' + m (?ger)
 *                                  
 * @param v1    vector [m]
 * @param v2    vector [n], will be transposed
 * @param m     matrix [m][n], will be overwritten with the result
 *
 */
void
nlk_vector_transposed_multiply_add(const NLK_ARRAY *v1, const NLK_ARRAY *v2, 
                                   NLK_ARRAY *m)
{
#ifndef NCHECKS
    if(v1->rows != m->rows) {
        NLK_ERROR_VOID("vector v1 length must be equal to number of matrix "
                       "rows", NLK_EBADLEN);
        /* unreachable */
    }
    if(v2->rows != m->cols) {
        NLK_ERROR_VOID("vector v2 length must be equal to number of matrix "
                       "columns", NLK_EBADLEN);
        /* unreachable */
    }
#endif

     cblas_sger(CblasRowMajor, m->rows, m->cols, 1, v1->data, 
                1, v2->data, 1, m->data, m->cols);
}

/**
 * Multiplies a matrix by a vector 
 * v2 = m * v1 + v2 or v2 = m' * v1 + v2
 *
 *  @param m        the matrix m [n][m]
 *  @param trans    NLK_TRANSPOSE or NLK_NOTRANSPOSE
 *  @param v1       if transpose vector [n], vector [m] otherwise;
 *                  the vector multiplied by the matrix
 *  @param v2       result vector [m] if transpose, result [n] otherwise;
 *
 */
void 
nlk_matrix_vector_multiply_add(const NLK_ARRAY *m, const NLK_OPTS trans, 
                               const NLK_ARRAY *v1, NLK_ARRAY *v2)
{
#ifndef NCHECKS
     /* First we need to make sure all dimensions make sense */
    if(trans == NLK_TRANSPOSE && v1->rows != m->rows) {
        NLK_ERROR_VOID("vector (v1) lenght must be equal to the number of "
                       "matrix rows", NLK_EBADLEN);
    }
    if(trans == NLK_NOTRANSPOSE && v1->rows != m->cols) {
        NLK_ERROR_VOID("vector (v1) lenght must be equal to the number of "
                       "matrix columns", NLK_EBADLEN);
    }
    if(trans == NLK_TRANSPOSE && v2->rows != m->cols) {
        NLK_ERROR_VOID("result vector (v2) length must be equal to the number "
                       "of matrix columns", NLK_EBADLEN);
        /* unreachable */
    }
    if(trans == NLK_NOTRANSPOSE && v2->rows != m->rows) {
        NLK_ERROR_VOID("result vector (v2) length must be equal to the number " 
                       "of matrix rows", NLK_EBADLEN);
        /* unreachable */
    }
#endif

    cblas_sgemv(CblasRowMajor, trans,
                m->rows, m->cols, 1, m->data, m->cols,
                v1->data, 1, 1, v2->data, 1); 
}

/**
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
* Not Parallel.
* @endnote
*/
void
nlk_array_init_sigmoid(NLK_ARRAY *array, const uint8_t max_exp) {
    size_t size = array->rows * array->cols;

    /* this splits the range [sigma(-max), sigma(max)] into *size* pieces */
    for(size_t ii = 0; ii < size; ii++) {
        array->data[ii] = exp(((nlk_real) ii / (nlk_real) size * 2 - 1) 
                              * max_exp);
        array->data[ii] = array->data[ii] / (array->data[ii] + 1);
    }
}


/**
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
NLK_TABLE *
nlk_table_sigmoid_create(const size_t table_size, const nlk_real max_exp)
{
    NLK_TABLE *table;

    /* allocate structure */
    table = (NLK_TABLE *) malloc(sizeof(NLK_TABLE));
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
 * Index table creation (positive integers).
 *
 * @param table_size    the table size
 * @param max           the maximum value to generate
 *
 * @return the table
 */
NLK_TABLE_INDEX *nlk_table_index_create(size_t table_size, size_t max)
{
    NLK_TABLE_INDEX *table;

    /* allocate structure */
    table = (NLK_TABLE_INDEX *) malloc(sizeof(NLK_TABLE));
    if(table == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for table struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    /* allocate array and set fields */
    table->table = (size_t *) malloc(table_size *  sizeof(size_t));
    if(table->table == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for table array",
                       NLK_ENOMEM);
        /* unreachable */
    }
    table->size = table_size;
    table->pos = (size_t)0;
    table->max = max;

    return table;
}


/**
 * Free a table.
 *
 * @param table the table to free
 */
void
nlk_table_free(NLK_TABLE *table)
{
    if(table != NULL) {
        nlk_array_free(table->table);
        free(table);
        table = NULL;
    }
}

/**
 * Free an index table.
 *
 * @param table the table to free
 */
void
nlk_table_index_free(NLK_TABLE_INDEX *table)
{
    if(table != NULL) {
        free(table->table);
        free(table);
        table = NULL;
    }
}

/**
 * Calculates the sigmoid 1/(exp(-x) + 1) for a real valued x.
 * 
 * @param sigmoid_table     a precomputed sigmoid table
 * @param x                 the real valued x for calculating its sigmoid
 *
 * @return the sigmoid of x
 */
nlk_real
nlk_sigmoid_table(const NLK_TABLE *sigmoid_table, const double x) 
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

/**
 * Calculates the elemtwise sigmoid for an entire array
 *
 * @param sigmoid_table     a precomputed sigmoid table
 * @param arr            the in/output array (overwritten with the result)
 *
 * @return NLK_SUCCESS or NLK_E* on error
 */
int
nlk_array_sigmoid_table(const NLK_TABLE *sigmoid_table, NLK_ARRAY *arr)
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

/**
 * Calculates the elemtwise logarithm for an entire array
 *
 * @param input             the input array
 * @param output            the output array (overwritten with the result)
 *
 * @note
 * Not parallel.
 * @endnote
 */
void
nlk_array_log(const NLK_ARRAY *input, NLK_ARRAY *output)
{
    const size_t len = input->rows * input->cols;

#ifndef NCHECKS
    if(input->cols != output->cols || input->rows != output->rows) {
        NLK_ERROR_VOID("array dimensions do not match.", NLK_EBADLEN);
        /* unreachable */
    }
#endif

    /*#pragma omp parallel for*/
    for(size_t ii = 0; ii < len; ii++) {
        output->data[ii] = log(input->data[ii]);
    }

}
