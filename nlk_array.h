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


/** @file nlk_array.h
 * Array definitions
 */


#ifndef __NLK_ARRAY_H__
#define __NLK_ARRAY_H__


#include <stdlib.h>
#include <stdbool.h>
#include <cblas.h>

#include "nlk_types.h"
#include "tinymt32.h"


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

#define NLK_MAX_EXP             6
#define NLK_SIGMOID_TABLE_SIZE  10000
#define NLK_RAND_POOL_SIZE      1000000

/* @enum NLK_OPTS
 * transpose options
 */
typedef enum nlk_opts { 
    NLK_NOTRANSPOSE  = CblasNoTrans,
    NLK_TRANSPOSE    = CblasTrans
} NLK_OPTS;

/** @typedef float nlk_real
 * The basic data type for arithmetic operations
 */
typedef float nlk_real;


/* @ struct nlk_array
 * 1D or 2D array (vector or matrix)
 */
struct nlk_array {
    size_t rows;    /* the number of rows in the matrix (size) */
    size_t cols;    /* the number of columns in the matrix (size) */
    nlk_real *data; /* pointer to the beginning of the matrix data */ 
};
typedef struct nlk_array nlk_Array;


/** @struct nlk_table
 * For (fast) precalculated lookup based math.
 * E.g. Exponent table for use with softmax
 */
struct nlk_table {
    nlk_Array *table;
    size_t size;
    nlk_real max;
    nlk_real min;
    size_t pos;
};
typedef struct nlk_table nlk_Table;


/*
 * Constructors, copy
 */
nlk_Array *nlk_array_create(const size_t, const size_t);

void nlk_set_seed(const int);
int nlk_get_seed();

uint32_t nlk_random_uint();

nlk_Array *nlk_array_create_copy(const nlk_Array *);

int nlk_array_copy_row(nlk_Array *, const size_t, const nlk_Array *,
                       const size_t);

int nlk_array_copy(nlk_Array *, const nlk_Array *);
void nlk_carray_copy_carray(nlk_real *, const nlk_real *, size_t);

bool nlk_array_compare_carray(nlk_Array *, nlk_real *, nlk_real);
bool nlk_carray_compare_carray(nlk_real *, nlk_real *, size_t, nlk_real);


void nlk_print_array(const nlk_Array *, const size_t, const size_t);

/*
 * Init functions
 */
void nlk_array_init_wity_carray(nlk_Array *, const nlk_real *);

void nlk_array_init_sigmoid(nlk_Array *, const uint8_t);

void nlk_array_zero(nlk_Array *);

/*
 * Random numbers
 */
void nlk_array_init_uniform(nlk_Array *, const nlk_real, const nlk_real);
nlk_Table *nlk_random_pool_create(size_t, const uint32_t);
void nlk_random_pool_reset(nlk_Table *);
uint32_t nlk_random_pool_get(nlk_Table *);



/*
 * Free Array
 */
void nlk_array_free(nlk_Array *);


/*
 * Basic Linear Algebra Operations
 */
/* scale */
void nlk_array_scale(const nlk_real, nlk_Array *);

/* normalize matrix row vectors */
void nlk_array_normalize_row_vectors(nlk_Array *);

/* normalize a vector */
void nlk_array_normalize_vector(nlk_Array *);

/* vector dot product */
nlk_real nlk_array_dot(const nlk_Array *, nlk_Array *, uint8_t);
nlk_real nlk_array_dot_carray(const nlk_Array *, nlk_real *);

/* elementwise addition */
int nlk_array_add(const nlk_Array *, nlk_Array *);
void nlk_array_add_carray(const nlk_Array *, nlk_real *);
int nlk_vector_add_row(const nlk_Array *, nlk_Array *, size_t);
int nlk_row_add_vector(const nlk_Array *, size_t, nlk_Array *);

/* elementwise multiplication */
int nlk_array_mul(const nlk_Array *, nlk_Array *);

/* sum of absolute array values */
nlk_real nlk_array_abs_sum(const nlk_Array *arr);
nlk_real nlk_carray_abs_sum(const nlk_real *, size_t);

/* number of non-zero elements in array */
size_t nlk_array_non_zero(const nlk_Array *arr);

/* scaled vector addition */
int nlk_add_scaled_vectors(const nlk_real, const nlk_Array *, nlk_Array *);

int nlk_vector_transposed_multiply_add(const nlk_Array *, const nlk_Array *, 
                                       nlk_Array *);

int nlk_matrix_vector_multiply_add(const nlk_Array *, const NLK_OPTS, 
                                   const nlk_Array *, nlk_Array *);

/*
 * Math for transfer functions
 */
nlk_Table *nlk_table_sigmoid_create(const size_t, const nlk_real);
void nlk_table_free(nlk_Table *);
nlk_real nlk_sigmoid_table(const nlk_Table *, const double);
int nlk_array_sigmoid_table(const nlk_Table *, nlk_Array *);
void nlk_array_sigmoid_approx(nlk_Array *);

int nlk_array_log(const nlk_Array *, nlk_Array *);

__END_DECLS
#endif /* __NLK_ARRAY__ */
