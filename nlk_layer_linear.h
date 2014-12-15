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
struct nlk_layer_lookup {
    size_t      max_indices;        /**< number of input indices (maximum) */
    nlk_Array   *weights;           /**<  [table_size][layer_size] */
};
typedef struct nlk_layer_lookup nlk_Layer_Lookup;

/** @struct nlk_layer_linear
 * A linear layer: y = Weights * x + bias, where y is the output and x the
 * input i.e. this layer applies a linear transformation on its input
 */
struct nlk_layer_linear {
    nlk_Array   *weights;           /**< weights [input_size][layer_size] */
    nlk_Array   *bias;              /**< the layer bias */
    nlk_Array   *output;            /**< ouput */
    nlk_Array   *grad_in;           /**< layer gradient (at input) */
};
typedef struct nlk_layer_linear nlk_Layer_Linear;


/* 
 *  Initialization for lookup or linear layers
 */
/* Initialize a linear layer that is followed by a sigmoid */
void nlk_layer_linear_init_sigmoid(nlk_Layer_Linear *);
void nlk_layer_lookup_init_sigmoid(nlk_Layer_Lookup *);

/*
 * Lookup Layer 
 */
/* Create a Lookup Layer */
nlk_Layer_Lookup *nlk_layer_lookup_create(const size_t, const size_t,
                                          const size_t);

/* Initialize the lookup layer */
void nlk_layer_lookup_init(nlk_Layer_Lookup *);
/* Simple lookup forward pass (1st layer) */
void nlk_layer_lookup_forward_lookup(nlk_Layer_Lookup *, const size_t *,
                                     const size_t, nlk_Array *);
/* Lookup with input (not 1st layer */
void nlk_layer_lookup_forward(nlk_Layer_Lookup *, const nlk_Array *, 
                              const size_t, nlk_real *output);
/* Lookup Layer backward pass */
void nlk_layer_lookup_backprop_lookup(nlk_Layer_Lookup *, const size_t, 
                                      const nlk_Array *);
void nlk_layer_lookup_backprop_acc(nlk_Layer_Lookup *, const nlk_Array *, 
                                   const size_t, const nlk_real, 
                                   nlk_Array *, nlk_Array *, nlk_Array *);

/*
 * Linear Layer
 */
/* Create a Linear Layer */
nlk_Layer_Linear *nlk_layer_linear_create(const size_t, const size_t, bool);


/* Linear Layer forward pass */
void nlk_layer_linear_forward(const nlk_Layer_Linear *, const nlk_Array *);



/* Linear Layer backward pass */
void nlk_layer_linear_backprop(nlk_Layer_Linear *, const nlk_Array *, 
                               const nlk_Array *);


/* save */
int nlk_layer_lookup_save(char *, const bool, nlk_Vocab **, 
                          nlk_Layer_Lookup *);

/*
 * ### Free ###
 */

/* Free Lookup Layer Memory */
void nlk_layer_lookup_free(nlk_Layer_Lookup *);

/* Free Linear Layer Memory */
void nlk_layer_linear_free(nlk_Layer_Linear *);


__END_DECLS
#endif /* __NLK_LAYER_LINEAR_H__ */
