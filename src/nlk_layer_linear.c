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


/** @file nlk_layer_linear.c
 * Linear Layer operations
 */

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_layer_linear.h"


/** 
 * Create a Linear Layer
 *
 * @param output_size   size of the output of the layer (output dimensions)
 * @param input_size    size of the input into the layer (input dimensions)
 * @param bias          if true this layer has a bias vector
 *
 * @return the Linear_Layer or NULL if creation failed
 */
struct nlk_layer_linear_t *
nlk_layer_linear_create(const size_t output_size,  const size_t input_size, 
                        bool bias)
{
    struct nlk_layer_linear_t *layer;
   
    /* Allocate memory for struct, create the members */
    layer = (NLK_LAYER_LINEAR *) malloc(sizeof(NLK_LAYER_LINEAR));
    if(layer == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for layer struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    layer->weights = nlk_array_create(output_size, input_size);
    if(bias) {
        layer->bias = nlk_array_create(output_size, 1);
    } else {
        layer->bias = NULL;
    }

    return layer;
}

/**
 * Create a Linear Layer from an already initialized weights array
 *
 * @param weights   an initialized weights array
 * @param bias      an initialized bias array
 *
 * @return pointer to a Linear Layer or NULL if creation failed
 */
struct nlk_layer_linear_t *
nlk_layer_linear_create_from_arrays(NLK_ARRAY *weights, NLK_ARRAY *bias)
{
    struct nlk_layer_linear_t *layer;

    /*
     * Allocate memory for struct, create the members
     */
    layer = (struct nlk_layer_linear_t *) 
                malloc(sizeof(struct nlk_layer_linear_t));

    if(layer == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for layer struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    layer->weights = weights;
    layer->bias = bias;

    return layer;
}

/** 
 * Initializes the linear layer weights 
 *
 * This assumes the transfer function that follows is the sigmoid
 * Initializations is done by drawing from a uniform distribution in the range
 * [ -4 * sqrt(6 / (fan_in + fan_out)), 4 * sqrt(6 / (fan_in + fan_out)) )
 * This strategy follows:
 *
 *  Bengio, X. Glorot, 
 *  Understanding the difficulty of training deep feedforward neural networks, 
 *  AISTATS 2010
 * 
 * The bias array is simply zeroed.
 *
 * @param layer     the Linear Layer to initialize
 *
 * @return no return, the linearlayer's weight matrix will be overwritten
 */
void
nlk_layer_linear_init_sigmoid(struct nlk_layer_linear_t *layer)
{
    nlk_real l = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_real h = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h);
    if(layer->bias != NULL) {
        nlk_array_zero(layer->bias);
    }
}

/**  
 * Linear Layer forward pass f(x) = W.x + b 
 * sets layer->output = layer->weights * input + layer->bias
 *
 * @param layer the linear layer
 * @param input the input to the layer
 *
 * return no return, layer->output (vector) is overwritten with result
 */
void
nlk_layer_linear_forward(const NLK_LAYER_LINEAR *layer, const NLK_ARRAY *input,
                         NLK_ARRAY *output)
{
#ifndef NCHECKS
    if(layer->bias->rows != output->rows) {
        nlk_debug("bias = [%zu, %zu], output = [%zu, %zu]", 
                  layer->bias->rows, layer->bias->cols, 
                  output->rows, output->cols);
        NLK_ERROR_VOID("bias and output dimensions do not match", NLK_EBADLEN);
        /* unreachable */
    }

#endif


    /* bias needs to be added to matrix-vector product */
    if(layer->bias != NULL) {
        nlk_array_copy(output, layer->bias);
    } else {
        nlk_array_zero(output);
    }

    nlk_matrix_vector_multiply_add(layer->weights, NLK_NOTRANSPOSE,
                                   input, output);


    NLK_ARRAY_CHECK_NAN(output, "output has NaNs");
}

/** 
 * Linear Layer backward pass 
 *      gradient_input = weights' * gradient_output 
 *      weights += gradient_output * input'
 *      bias += gradient_output
 *
 * @param layer     the linear layer
 * @param input     the input vector (corresponding to this gradient/output)
 * @param grad_out  gradient at output layer
 *
 * @return no return, layer->grad_in, layer->weights and layer->bias are 
 *         updated
 */
void 
nlk_layer_linear_backprop(NLK_LAYER_LINEAR *layer, const NLK_ARRAY *input, 
                          const NLK_ARRAY *grad_out, NLK_ARRAY *grad_in)
{
    /* gradient (at input)  */
    nlk_array_zero(grad_in);
    nlk_matrix_vector_multiply_add(layer->weights, NLK_TRANSPOSE, 
                                   grad_out, grad_in);
    /* update weights */
    nlk_vector_transposed_multiply_add(grad_out, input, layer->weights);
    /* update bias */
    if(layer->bias != NULL) {
        nlk_array_add(grad_out, layer->bias);  
    }
}

/**
 * Free Linear Layer memory 
 *
 * @param layer    the Linear Layer
 */
void 
nlk_layer_linear_free(NLK_LAYER_LINEAR *layer)
{
    nlk_array_free(layer->weights);
    if(layer->bias != NULL) {
        nlk_array_free(layer->bias);
    }
    free(layer);
}

/**
 * Save a linear layer to a file pointer
 *
 * @param layer the linear layer
 * @param fp    the file pointer
 */
void
nlk_layer_linear_save(struct nlk_layer_linear_t *layer, FILE *fp)
{
    if(layer->bias != NULL) {
        fprintf(fp, "%d\n", 1);
        nlk_array_save(layer->bias, fp);
    } else {
        fprintf(fp, "%d\n", 0);
    }

    nlk_array_save(layer->weights, fp);
}

/** 
 * Load linear layer from a file pointer (NLK_FILE_BIN)
 *
 * @param fp    the file pointer
 *
 * @return the loaded layer
 */
struct nlk_layer_linear_t * 
nlk_layer_linear_load(FILE *in)
{
    struct nlk_layer_linear_t *layer;
    NLK_ARRAY *bias = NULL;
    NLK_ARRAY *weights = NULL;
    int tmp;
    int ret = 0;

    ret = fscanf(in, "%d\n", &tmp);
    if(ret <= 0) {
        NLK_ERROR_NULL("bad header", NLK_EINVAL);
        /* unreachable */
    }

    if(tmp != 0) {
        bias = nlk_array_load(in);
    }

    weights = nlk_array_load(in);
    if(weights == NULL) {
        NLK_ERROR_NULL("unable to load weights", NLK_EINVAL);
    }

    layer = nlk_layer_linear_create_from_arrays(weights, bias);

    return layer;
}


