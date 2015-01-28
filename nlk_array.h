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

#ifdef ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "tinymt32.h"


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


/* @enum NLK_OPTS
 * transpose options
 */
typedef enum nlk_opts_t { 
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
struct nlk_array_t {
    size_t rows;    /* the number of rows in the matrix (size) */
    size_t cols;    /* the number of columns in the matrix (size) */
    nlk_real *data; /* pointer to the beginning of the matrix data */ 
};
typedef struct nlk_array_t NLK_ARRAY;


/** @struct nlk_table
 * For (fast) precalculated lookup based math.
 * E.g. Exponent table for use with softmax
 */
struct nlk_table_t {
    NLK_ARRAY *table;
    size_t size;
    nlk_real max;
    nlk_real min;
};
typedef struct nlk_table_t NLK_TABLE;

/** @struct nlk_table_index
 * Table Lookup of Indexes
 * E.g. NEG table
 */
struct nlk_table_index_t {
    size_t *table;
    size_t size;
    size_t max;
    size_t pos;
};
typedef struct nlk_table_index_t NLK_TABLE_INDEX;


/*
 * Constructors, copy
 */
NLK_ARRAY *nlk_array_create(const size_t, const size_t);
NLK_ARRAY *nlk_array_create_view(const size_t, const size_t);

NLK_ARRAY *nlk_array_resize(NLK_ARRAY *, const size_t, const size_t);

NLK_ARRAY *nlk_array_create_copy(const NLK_ARRAY *);

int nlk_array_copy_row(NLK_ARRAY *, const size_t, const NLK_ARRAY *,
                       const size_t);

void nlk_array_copy(NLK_ARRAY *, const NLK_ARRAY *);
void nlk_carray_copy_carray(nlk_real *, const nlk_real *, size_t);

bool nlk_array_compare_carray(NLK_ARRAY *, nlk_real *, nlk_real);
bool nlk_carray_compare_carray(nlk_real *, nlk_real *, size_t, nlk_real);


void nlk_print_array(const NLK_ARRAY *, const size_t, const size_t);

/*
 * Init functions
 */
void nlk_array_init_wity_carray(NLK_ARRAY *, const nlk_real *);

void nlk_array_init_sigmoid(NLK_ARRAY *, const uint8_t);

void nlk_array_zero(NLK_ARRAY *);

/*
 * Random numbers & Index Table
 */
void nlk_array_init_uniform(NLK_ARRAY *, const nlk_real, const nlk_real, 
                            tinymt32_t *);
void nlk_carray_init_uniform(nlk_real *, const nlk_real, const nlk_real,
                             size_t, tinymt32_t *);
NLK_TABLE_INDEX *nlk_table_index_create(size_t, size_t);


/* 
 * Save/Load 
 */
void nlk_array_save(NLK_ARRAY *, FILE *fp);
NLK_ARRAY *nlk_array_load(FILE *fp);


/*
 * Free Array
 */
void nlk_array_free(NLK_ARRAY *);


/*
 * Basic Linear Algebra Operations
 */
/* scale */
void nlk_array_scale(const nlk_real, NLK_ARRAY *);

/* normalize matrix row vectors */
void nlk_array_normalize_row_vectors(NLK_ARRAY *);

/* normalize a vector */
void nlk_array_normalize_vector(NLK_ARRAY *);

/* vector dot product */
nlk_real nlk_array_dot(const NLK_ARRAY *, NLK_ARRAY *, uint8_t);
nlk_real nlk_array_dot_carray(const NLK_ARRAY *, nlk_real *);
nlk_real nlk_array_row_dot(const NLK_ARRAY *, size_t, NLK_ARRAY *, size_t);


/* elementwise addition */
void nlk_array_add(const NLK_ARRAY *, NLK_ARRAY *);
void nlk_array_add_carray(const NLK_ARRAY *, nlk_real *);
void nlk_vector_add_row(const NLK_ARRAY *, NLK_ARRAY *, size_t);
void nlk_row_add_vector(const NLK_ARRAY *, size_t, NLK_ARRAY *);
void nlk_add_scaled_row_vector(const nlk_real, const NLK_ARRAY *, 
                               const size_t, NLK_ARRAY *);
void nlk_add_scaled_vector_row(const nlk_real, const NLK_ARRAY *, NLK_ARRAY *, 
                               const size_t row);




/* elementwise multiplication */
void nlk_array_mul(const NLK_ARRAY *, NLK_ARRAY *);

/* sum of absolute array values */
nlk_real nlk_array_abs_sum(const NLK_ARRAY *arr);
nlk_real nlk_carray_abs_sum(const nlk_real *, size_t);

/* number of non-zero elements in array */
size_t nlk_array_non_zero(const NLK_ARRAY *arr);

/* scaled vector addition */
void nlk_add_scaled_vectors(const nlk_real, const NLK_ARRAY *, NLK_ARRAY *);

void nlk_vector_transposed_multiply_add(const NLK_ARRAY *, const NLK_ARRAY *, 
                                        NLK_ARRAY *);

void nlk_matrix_vector_multiply_add(const NLK_ARRAY *, const NLK_OPTS, 
                                    const NLK_ARRAY *, NLK_ARRAY *);

/*
 * Lookup Math & Transfer Function Math
 */
void nlk_table_free(NLK_TABLE *);

NLK_TABLE *nlk_table_sigmoid_create(const size_t, const nlk_real);
nlk_real nlk_sigmoid_table(const NLK_TABLE *, const double);
int nlk_array_sigmoid_table(const NLK_TABLE *, NLK_ARRAY *);

NLK_TABLE_INDEX *nlk_table_index_create(size_t table_size, size_t max);
void nlk_table_index_free(NLK_TABLE_INDEX *table);


void nlk_array_sigmoid_approx(NLK_ARRAY *);

void nlk_array_log(const NLK_ARRAY *, NLK_ARRAY *);

__END_DECLS
#endif /* __NLK_ARRAY__ */
