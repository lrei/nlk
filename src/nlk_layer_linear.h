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



/** @file nlk_layer_linear.h
 * Linear Layers
 */


#ifndef __NLK_LAYER_LINEAR_H__
#define __NLK_LAYER_LINEAR_H__


#include <stdbool.h>

#include "nlk_array.h"
#include "nlk_vocabulary.h"


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

/** @enum NLK_FORMAT
 * File formats for saving weights
 */
enum nlk_file_format_t {
    NLK_FILE_W2V_TXT = 0,
    NLK_FILE_W2V_BIN = 1,
    NLK_FILE_BIN = 2,
    NLK_FILE_TXT = 3,
};
typedef enum nlk_file_format_t NLK_FILE_FORMAT;


/** @struct nlk_layer_lookup
 * A lookup table layer is usually used to convert between a list of indexes 
 * and their corresponding vectors.
 */
struct nlk_layer_lookup_t {
    NLK_ARRAY   *weights;       /**< weights  [table_size][layer_size] */
};
typedef struct nlk_layer_lookup_t NLK_LAYER_LOOKUP;

/** @struct nlk_layer_linear
 * A linear layer: y = Weights * x + bias, where y is the output and x the
 * input i.e. this layer applies a linear transformation on its input
 */
struct nlk_layer_linear_t {
    NLK_ARRAY   *weights;           /**< weights [input_size][layer_size] */
    NLK_ARRAY   *bias;              /**< the layer bias */
    NLK_ARRAY   *output;            /**< ouput */
    NLK_ARRAY   *grad_in;           /**< layer gradient (at input) */
};
typedef struct nlk_layer_linear_t NLK_LAYER_LINEAR;


/* 
 *  Initialization for lookup or linear layers
 */
/* Initialize a linear layer that is followed by a sigmoid */
void nlk_layer_lookup_init_sigmoid(struct nlk_layer_lookup_t *);
void nlk_layer_lookup_init_sigmoid_from(struct nlk_layer_lookup_t *, size_t);
/*
 * Lookup Layer 
 */
/* Create a Lookup Layer */
struct nlk_layer_lookup_t *nlk_layer_lookup_create(const size_t, const size_t);
struct nlk_layer_lookup_t *nlk_layer_lookup_create_from_array(NLK_ARRAY *);

int nlk_layer_lookup_resize(struct nlk_layer_lookup_t *, const size_t);

/* Initialize the lookup layer */
void nlk_layer_lookup_init(struct nlk_layer_lookup_t *);
void nlk_layer_lookup_init_array(NLK_ARRAY *);

/*
 * Simple Lookup (1st Layer)
 */
/* Simple lookup forward pass with multiple inputs */
void nlk_layer_lookup_forward_lookup(struct nlk_layer_lookup_t *, 
                                     const size_t *, 
                                     const size_t, NLK_ARRAY *);
/* Lookup forward pass with built-in averaging of multiple inputs */
void nlk_layer_lookup_forward_lookup_avg(struct nlk_layer_lookup_t *, 
                                         const size_t *, 
                                         const size_t, NLK_ARRAY *);
void nlk_layer_lookup_forward_lookup_avg_p(struct nlk_layer_lookup_t *, 
                                           const size_t *, const size_t, 
                                           NLK_ARRAY *);
/* Lookup forward pass with concatenation of multiple inputs */
void nlk_layer_lookup_forward_lookup_concat(struct nlk_layer_lookup_t *, 
                                            const size_t *, 
                                            const size_t, NLK_ARRAY *);
void nlk_layer_lookup_forward_lookup_concat_p(struct nlk_layer_lookup_t *, 
                                              const size_t *, const size_t, 
                                              NLK_ARRAY *);

/* Lookup forward for just one input */
void nlk_layer_lookup_forward_lookup_one(struct nlk_layer_lookup_t *,
                                         const size_t, NLK_ARRAY *);

/*
 * Lookup with input (not 1st Layer)
 */
/* Lookup with input (not 1st layer */
void nlk_layer_lookup_forward(struct nlk_layer_lookup_t *, const NLK_ARRAY *, 
                              const size_t, nlk_real *output);

/*
 * Lookup Backprop
 */
/* Lookup Layer backprop with multiple inputs */
void nlk_layer_lookup_backprop_lookup(struct nlk_layer_lookup_t *, 
                                      const size_t *, const size_t, 
                                      const NLK_ARRAY *);
void nlk_layer_lookup_backprop_lookup_concat(struct nlk_layer_lookup_t *, 
                                             const size_t *, const size_t, 
                                             const NLK_ARRAY *);

/* Lookup Layer backprop with a single input */
void nlk_layer_lookup_backprop_lookup_one(struct nlk_layer_lookup_t *, 
                                          const size_t, const NLK_ARRAY *);
/* Lookup Backprop with accumulator */
void nlk_layer_lookup_backprop_acc(struct nlk_layer_lookup_t *, const bool,
                                   const NLK_ARRAY *, const size_t, 
                                   const nlk_real, NLK_ARRAY *);

/*
 * Lookup Serialization (Save & Load)
 */
/* save */
void nlk_layer_lookup_save(struct nlk_layer_lookup_t *, FILE *);
void nlk_layer_lookup_save_path(struct nlk_layer_lookup_t *, char *);
void nlk_layer_lookup_save_rows_path(struct nlk_layer_lookup_t *, char *, 
                                     size_t, size_t); 
/* load */
struct nlk_layer_lookup_t *nlk_layer_lookup_load(FILE *);
struct nlk_layer_lookup_t *nlk_layer_lookup_load_path(char *);


/*
 * Linear Layer
 */
/* Create a Linear Layer */
struct nlk_layer_linear_t *nlk_layer_linear_create(const size_t, const size_t,
                                                   bool);

/* Initialize the lookup layer */
void nlk_layer_linear_init(struct nlk_layer_lookup_t *);
void nlk_layer_linear_init_from(struct nlk_layer_lookup_t *, size_t);
void nlk_layer_linear_init_sigmoid(struct nlk_layer_linear_t *);




/* Linear Layer forward pass */
void nlk_layer_linear_forward(const NLK_LAYER_LINEAR *, const NLK_ARRAY *);



/* Linear Layer backward pass */
void nlk_layer_linear_backprop(NLK_LAYER_LINEAR *, const NLK_ARRAY *, 
                               const NLK_ARRAY *);



/*
 * ### Free ###
 */

/* Free Lookup Layer Memory */
void nlk_layer_lookup_free(struct nlk_layer_lookup_t *);

/* Free Linear Layer Memory */
void nlk_layer_linear_free(NLK_LAYER_LINEAR *);


__END_DECLS
#endif /* __NLK_LAYER_LINEAR_H__ */