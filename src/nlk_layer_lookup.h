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



/** @file nlk_layer_lookup.h
 * Lookup Layer definition
 */

#ifndef __NLK_LAYER_LOOKUP_H__
#define __NLK_LAYER_LOOKUP_H__


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


/** @struct nlk_layer_lookup
 * A lookup table layer is usually used to convert between a list of indexes 
 * and their corresponding vectors.
 */
struct nlk_layer_lookup_t {
    NLK_ARRAY   *weights;           /**< weights  [table_size][layer_size] */
    bool         update;            /**< should weights change? */
    nlk_real     learn_rate;        /**< layer specific learning rate */
    nlk_real     learn_rate_decay;  /**< layer specific learning rate decay */
};
typedef struct nlk_layer_lookup_t NLK_LAYER_LOOKUP;


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

/* Initialize a linear layer that is followed by a sigmoid */
void nlk_layer_lookup_init_sigmoid(struct nlk_layer_lookup_t *);
void nlk_layer_lookup_init_sigmoid_from(struct nlk_layer_lookup_t *, size_t);
void nlk_layer_lookup_init_sigmoid_ids(struct nlk_layer_lookup_t *,
                                       const size_t *, const size_t);

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
                                             const size_t, const NLK_ARRAY *);

/* Lookup Layer backprop with a single input */
void nlk_layer_lookup_backprop_lookup_one(struct nlk_layer_lookup_t *, 
                                          const size_t, const NLK_ARRAY *);
void nlk_layer_lookup_backprop_lookup_concat_one(struct nlk_layer_lookup_t *, 
                                                 const size_t, const size_t,
                                                 const NLK_ARRAY *);

/* Lookup Backprop with accumulator */
void nlk_layer_lookup_backprop_acc(struct nlk_layer_lookup_t *,
                                   const NLK_ARRAY *, const size_t, 
                                   const nlk_real, NLK_ARRAY *);


/*
 * Lookup Serialization (Save & Load)
 */
/* save */
void nlk_layer_lookup_save(const struct nlk_layer_lookup_t *, FILE *);
void nlk_layer_lookup_save_path(const struct nlk_layer_lookup_t *, 
                                const char *);
void nlk_layer_lookup_save_rows_path(struct nlk_layer_lookup_t *, char *, 
                                     size_t, size_t); 
/* load */
struct nlk_layer_lookup_t *nlk_layer_lookup_load(FILE *);
struct nlk_layer_lookup_t *nlk_layer_lookup_load_path(char *);




/* Free Lookup Layer Memory */
void nlk_layer_lookup_free(struct nlk_layer_lookup_t *);


__END_DECLS
#endif /* __NLK_LAYER_LOOKUP_H__ */
