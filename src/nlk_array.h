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


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "nlk_math.h"

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


/* @ struct nlk_array
 * 1D or 2D array (vector or matrix)
 */
struct nlk_array_t {
    size_t rows;    /* the number of rows in the matrix (size) */
    size_t cols;    /* the number of columns in the matrix (size) */
    nlk_real *data; /* pointer to the beginning of the matrix data */ 
};
typedef struct nlk_array_t NLK_ARRAY;


/* debug helpers */
void nlk_print_array(const struct nlk_array_t *, const size_t, const size_t);
bool nlk_array_has_nan(const struct nlk_array_t *);
bool nlk_array_has_nan_row(const struct nlk_array_t *, const size_t);
bool nlk_carray_has_nan(const nlk_real *, const size_t);

#ifdef CHECK_NANS
#define NLK_ARRAY_CHECK_NAN(A, M) \
    do { \
        if(nlk_array_has_nan(A)) { \
            NLK_ERROR_ABORT(M, NLK_ENAN); \
        } \
    } while(0)

#define NLK_ARRAY_CHECK_NAN_ROW(A, R, M) \
    do { \
        if(nlk_array_has_nan_row(A, R)) { \
            NLK_ERROR_ABORT(M, NLK_ENAN); \
        } \
    } while(0)

#define NLK_CARRAY_CHECK_NAN(C, L, M) \
    do { \
        if(nlk_carray_has_nan(C, L)) { \
            NLK_ERROR_ABORT(M, NLK_ENAN); \
        } \
    } while(0)

#define NLK_CHECK_NAN(V, M) \
do { \
        if(!isfinite(V)) { \
            NLK_ERROR_ABORT(M, NLK_ENAN); \
        } \
    } while(0)

#else
#define NLK_ARRAY_CHECK_NAN(A, M)
#define NLK_ARRAY_CHECK_NAN_ROW(A, R, M)
#define NLK_CARRAY_CHECK_NAN(C, L, M)
#define NLK_CHECK_NAN(V, M)
#endif

/*
 * Constructors, copy
 */
struct  nlk_array_t *nlk_array_create(const size_t, const size_t);
void    nlk_array_row_view(const NLK_ARRAY *, const size_t, NLK_ARRAY *);

struct  nlk_array_t *nlk_array_resize(struct nlk_array_t *, const size_t, 
                                      const size_t);
struct  nlk_array_t *nlk_array_create_copy(const struct nlk_array_t *);
struct  nlk_array_t *nlk_array_create_copy_limit(const struct nlk_array_t *,
                                                 size_t);
int     nlk_array_copy_row(struct nlk_array_t *, const size_t,
                           const struct nlk_array_t *, const size_t);
void    nlk_array_copy_row_carray(struct nlk_array_t *, const size_t, 
                                  nlk_real *);
int     nlk_array_copy_row_vector(struct nlk_array_t *, const unsigned int, 
                                  const struct nlk_array_t *, const size_t);


void nlk_array_copy(struct nlk_array_t *, const struct nlk_array_t *);
void nlk_carray_copy_carray(nlk_real *, const nlk_real *, size_t);

/* compare */
bool nlk_array_compare_carray(struct nlk_array_t *, nlk_real *, nlk_real);
bool nlk_carray_compare_carray(nlk_real *, nlk_real *, size_t, nlk_real);
bool nlk_carray_compare_exact_carray(nlk_real *, nlk_real *, size_t);
bool nlk_array_compare_exact_carray(struct nlk_array_t *, nlk_real *);

/*
 * Init functions
 */
void nlk_array_init_wity_carray(struct nlk_array_t *, const nlk_real *);

void nlk_array_init_sigmoid(struct nlk_array_t *, const uint8_t);

void nlk_array_zero(struct nlk_array_t *);

/*
 * Random numbers & Index Table
 */
void nlk_array_init_uniform(struct nlk_array_t *, const nlk_real, 
                            const nlk_real);
void nlk_carray_init_uniform(nlk_real *, const nlk_real, const nlk_real,
                             size_t);

/* 
 * Save/Load 
 */
void nlk_array_save(struct nlk_array_t *, FILE *);
void nlk_array_save_rows(struct nlk_array_t *, FILE *, size_t, size_t);
void nlk_array_save_text(struct nlk_array_t *, FILE *);

struct nlk_array_t *nlk_array_load(FILE *);
struct nlk_array_t *nlk_array_load_text(FILE *);


/*
 * Free Array
 */
void nlk_array_free(struct nlk_array_t *);


/*
 * Basic Linear Algebra Operations
 */
/* scale */
void nlk_array_scale(const nlk_real, struct nlk_array_t *);

/* add constant */
void nlk_array_add_constant(const nlk_real, struct nlk_array_t *);

/* normalize matrix row vectors */
void nlk_array_normalize_row_vectors(struct nlk_array_t *);

/* normalize a vector */
void nlk_array_normalize_vector(struct nlk_array_t *);

/* vector dot product */
nlk_real nlk_array_dot(const struct nlk_array_t *, const struct nlk_array_t *, 
                       int8_t);
nlk_real nlk_array_dot_carray(const struct nlk_array_t *, nlk_real *);
nlk_real nlk_array_row_dot(const struct nlk_array_t *, size_t, 
                           struct nlk_array_t *, size_t);


/* elementwise addition */
void nlk_array_add(const struct nlk_array_t *, struct nlk_array_t *);
void nlk_array_scaled_add(const nlk_real, const struct nlk_array_t *, 
                          struct nlk_array_t *);
void nlk_array_add_carray(const struct nlk_array_t *, nlk_real *);
void nlk_array_add_carray_partial(const struct nlk_array_t *, nlk_real *, 
                                  const size_t);
void nlk_vector_add_row(const struct nlk_array_t *, struct nlk_array_t *, 
                        size_t);
void nlk_row_add_vector(const struct nlk_array_t *, size_t, 
                        struct nlk_array_t *);
void nlk_add_scaled_row_vector(const nlk_real, const struct nlk_array_t *, 
                               const size_t, unsigned int dim, 
                               struct nlk_array_t *);
void nlk_add_scaled_vector_row(const nlk_real, const struct nlk_array_t *, 
                               struct nlk_array_t *, const size_t row);


/* elementwise multiplication */
void nlk_array_mul(const struct nlk_array_t *, struct nlk_array_t *);

/* sum of array values */
nlk_real nlk_array_sum(const struct nlk_array_t *arr);
nlk_real nlk_carray_sum(const nlk_real *, size_t);
nlk_real nlk_array_abs_sum(const struct nlk_array_t *arr);
nlk_real nlk_carray_abs_sum(const nlk_real *, size_t);
nlk_real nlk_array_squared_sum(const struct nlk_array_t *);


/* number of non-zero elements in array */
size_t nlk_array_non_zero(const struct nlk_array_t *arr);

/* scaled vector addition */
void nlk_add_scaled_vectors(const nlk_real, const struct nlk_array_t *, 
                            struct nlk_array_t *);

/* multiply */
void nlk_vector_transposed_multiply_add(const struct nlk_array_t *, 
                                        const struct nlk_array_t *, 
                                        struct nlk_array_t *);

void nlk_matrix_vector_multiply_add(const struct nlk_array_t *, 
                                    const NLK_OPTS, const struct nlk_array_t *, 
                                    struct nlk_array_t *);

/*
 * Lookup Math & Transfer Function Math
 */
int     nlk_sigmoid_array(struct nlk_array_t *);
void    nlk_array_sigmoid_approx(struct nlk_array_t *);
void    nlk_array_log(const struct nlk_array_t *, struct nlk_array_t *);

/* max */
size_t      nlk_array_max_i(const NLK_ARRAY *);
nlk_real    nlk_array_max(const NLK_ARRAY *);
nlk_real    nlk_array_rescale_max_minus(NLK_ARRAY *array);


__END_DECLS
#endif /* __NLK_ARRAY_H__ */
