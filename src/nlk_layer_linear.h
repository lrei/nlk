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


/** @struct nlk_layer_linear
 * A linear layer: y = Weights * x + bias, where y is the output and x the
 * input i.e. this layer applies a linear transformation on its input
 */
struct nlk_layer_linear_t {
    NLK_ARRAY   *weights;           /**< weights [input_size][layer_size] */
    NLK_ARRAY   *bias;              /**< the layer bias */
};
typedef struct nlk_layer_linear_t NLK_LAYER_LINEAR;


/* 
 *  Initialization for linear layers
 */
void nlk_layer_linear_init_sigmoid(struct nlk_layer_linear_t *layer);


/*
 * Linear Layer
 */
/* Create a Linear Layer */
struct nlk_layer_linear_t *nlk_layer_linear_create(const size_t, const size_t,
                                                   bool);
struct nlk_layer_linear_t *nlk_layer_linear_create_from_arrays(NLK_ARRAY *, 
                                                               NLK_ARRAY *);


/* Linear Layer forward pass */
void nlk_layer_linear_forward(const NLK_LAYER_LINEAR *, const NLK_ARRAY *,
                              NLK_ARRAY *);



/* Linear Layer backward pass */
void nlk_layer_linear_backprop(NLK_LAYER_LINEAR *, const NLK_ARRAY *, 
                               const NLK_ARRAY *, NLK_ARRAY *);


/* Free Linear Layer Memory */
void nlk_layer_linear_free(NLK_LAYER_LINEAR *);

/* Load & Save */
void nlk_layer_linear_save(struct nlk_layer_linear_t *, FILE *);
struct nlk_layer_linear_t *nlk_layer_linear_load(FILE *);


__END_DECLS
#endif /* __NLK_LAYER_LINEAR_H__ */
