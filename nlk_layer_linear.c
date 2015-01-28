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
#include "nlk_vocabulary.h"
#include "nlk_layer_linear.h"


/**
 * Create a Lookup Layer 
 *
 * @param table_size    e.g. the size of the vocabulary 
 * @param layer_size    the size of this layer  
 *
 * @return pointer to a Lookup_Layer or NULL if creation failed
 */
struct nlk_layer_lookup_t *
nlk_layer_lookup_create(const size_t table_size, const size_t layer_size)
{
    struct nlk_layer_lookup_t *layer;
   
    /*
     * Allocate memory for struct, create the members
     */
    layer = (struct nlk_layer_lookup_t *) malloc(sizeof(struct nlk_layer_lookup_t));
    if(layer == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for layer struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    layer->weights = nlk_array_create(table_size, layer_size);

    if(layer->weights == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for lookup layer weights",
                       NLK_ENOMEM);
        /* unreachable */
    }

    /* default weight initialization, zero */
    nlk_array_zero(layer->weights);

    layer->frozen_limit = 0;

    return layer;
}

/**
 * Create a Lookup Layer from an already initialized weight array
 *
 * @param weights   an initialized weights array
 *
 * @return pointer to a Lookup_Layer or NULL if creation failed
 */
struct nlk_layer_lookup_t *
nlk_layer_lookup_create_from_array(NLK_ARRAY *weights)
{
    struct nlk_layer_lookup_t *layer;

    /*
     * Allocate memory for struct, create the members
     */
    layer = (struct nlk_layer_lookup_t *) malloc(sizeof(struct nlk_layer_lookup_t));
    if(layer == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for layer struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    layer->weights = weights;

    layer->frozen_limit = 0;

    return layer;
}

/** 
 * Resize a lookup layer increasing or decreasing the table size. Copies old
 * values up to new table_size if larger or old table_size if smaller.
 * Does not initialize new weights if new table_size is larger - caller must 
 * do it!
 *
 * @param layer         the lookup layer to resize
 * @param table_size    the new table size.
 *
 * @return NLK_SUCCESS or NLK_FAILURE
 */
int
nlk_layer_lookup_resize(struct nlk_layer_lookup_t *layer, const size_t table_size)
{
    NLK_ARRAY *weights = nlk_array_resize(layer->weights, table_size,
                                          layer->weights->cols);
    if(weights == NULL) {
        return NLK_FAILURE;
    }
    
    layer->weights = weights;

    return NLK_SUCCESS;
}

/** 
 * Initializes the lookup layer weights 
 * Initializations is done by drawing from a uniform distribution in the range
 * [-0.5 / layer_size, 0.5 / layer_size )
 * This follows the word2vec initialization for the lookup layer
 *
 * @param layer     the Lookup Layer to initialize
 * @param rng       the random number generator
 *
 * @return no return, the lookup layer's weight matrix will be overwritten
 */
void
nlk_layer_lookup_init(struct nlk_layer_lookup_t *layer, tinymt32_t *rng)
{
    nlk_real low = -0.5 / layer->weights->cols;
    nlk_real high = 0.5 / layer->weights->cols; 
    nlk_array_init_uniform(layer->weights, low, high, rng);
}

/** @fn void nlk_layer_lookup_freeze(struct nlk_layer_lookup_t *layer)
 * "Freezes" weight updates in this layer, preventing backpropagation updates
 *
 * @param layer             the Lookup Layer to initialize
 * @param frozen_limit      after and incl. this index, weights are not frozen
 *
 * @return no return, the lookup layer's weight matrix will be overwritten
 */
void
nlk_layer_lookup_freeze(struct nlk_layer_lookup_t *layer, size_t frozen_limit)
{
    layer->frozen_limit = frozen_limit;
}


/** @fn nlk_layer_lookup_init_sigmoid(NLK_LAYER_LINEAR *layer)
 * Initializes the linear layer weights 
 *
 *
 * This assumes the transfer function that follows is the sigmoid
 * Initializations is done by drawing from a uniform distribution in the range
 * [ -4 * sqrt(6 / (fan_in + fan_out)), 4 * sqrt(6 / (fan_in + fan_out)) )
 * This strategy follows:
 *
 *  Bengio, X. Glorot, 
 *  Understanding the difficulty of training deep feedforward neuralnetworks, 
 *  AISTATS 2010
 * 
 * @param layer     the Lookup Layer to initialize
 *
 * @return no return, the lookup layer's weight matrix will be overwritten
 */
void
nlk_layer_lookup_init_sigmoid(struct nlk_layer_lookup_t *layer, tinymt32_t *rng)
{
    nlk_real l = -4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows + 
                                                layer->weights->cols));
    nlk_real h =  4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows +
                                                layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h, rng);
}

/** 
 * Same as above (nlk_layer_lookup_init) but only for weights after a given row
 *
 * @param layer the lookup layer
 * @param from  the starting row
 */
void
nlk_layer_lookup_init_sigmoid_from(struct nlk_layer_lookup_t *layer, size_t from,
                                   tinymt32_t *rng)
{
    size_t len = (layer->weights->rows - from) * layer->weights->cols;
    nlk_real l = -4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows + 
                                                layer->weights->cols));
    nlk_real h =  4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows +
                                                layer->weights->cols));
    nlk_carray_init_uniform(&layer->weights->data[from * layer->weights->cols], 
                            l, h, len, rng);
}

/** 
 * Lookup Layer forward pass with just ids (1st layer)
 *
 * @param layer         the lookup layer
 * @param indices       the indices to lookup 
 * @param n_indices     the number of indices in the indices array (array size)
 * @param output        the output of the lookup forward pass
 *
 * return no return, output is overwritten with result: n_indices * layer->cols
 */
void
nlk_layer_lookup_forward_lookup(struct nlk_layer_lookup_t *layer, const size_t *indices, 
                                const size_t n_indices, NLK_ARRAY *output)
{
    size_t ii;

#ifndef NCHECKS
    if (n_indices == 0) {
        NLK_ERROR_VOID("empty input - indices parameter must be non-zero",
                       NLK_EINVAL);
        /* unreachable */
    }
#endif

    /* copy content from indices to the ouput */
    for(ii = 0; ii < n_indices; ii++) {
        /** @warning this bit of code is duplicated for NCHECKS */
#ifndef NCHECKS
        size_t ret;
        ret = nlk_array_copy_row(output, ii , layer->weights, indices[ii]); 
        if(ret != NLK_SUCCESS) {
            NLK_ERROR_VOID("Invalid lookup", ret);
        }
#else
        nlk_array_copy_row(output, ii , layer->weights, indices[ii]); 
#endif
    }
}

/** 
 * Lookup Layer forward pass with just ids (1st layer) but averages them
 * together. I.E. a shortcut for lookup followed by average
 *
 * @param layer         the lookup layer
 * @param indices       the indices to lookup 
 * @param n_indices     the number of indices in the indices array (array size)
 * @param output        the output of the lookup forward pass
 *
 * return no return, output is overwritten with result: 1 * layer->cols
 */
void
nlk_layer_lookup_forward_lookup_avg(struct nlk_layer_lookup_t *layer, 
                                    const size_t *indices,
                                    const size_t n_indices, NLK_ARRAY *output)
{
    size_t ii;
    nlk_real s;

#ifndef NCHECKS
    if (n_indices == 0) {
        NLK_ERROR_VOID("empty input - indices parameter must be non-zero",
                       NLK_EINVAL);
        /* unreachable */
    }
#endif

    s =  1.0 / (nlk_real) n_indices;

    /* copy content from indices to the ouput */
    for(ii = 0; ii < n_indices; ii++) {
       nlk_add_scaled_row_vector(s, layer->weights, indices[ii], 0,  output);
    }
}


/** 
 * Lookup Layer forward pass with just one id (1st layer) followed by a
 * transformation of the output vector from a column vector to a row vector
 * I.E. A shortcut for lookup followed by concat
 *
 * @param layer         the lookup layer
 * @param index         the index to lookup 
 * @param output        the output of the lookup forward pass
 *
 * return no return, output is overwritten with result: n_indices * layer->cols
 */
void
nlk_layer_lookup_forward_lookup_one(struct nlk_layer_lookup_t *layer, 
                                    const size_t index, NLK_ARRAY *output)
{
    /** @warning this function is duplicated for NCHECKS */

#ifndef NCHECKS
    /* copy content from index columns to the ouput rows */
    size_t ret;
    ret = nlk_array_copy_row_vector(output, 0, layer->weights, index); 
    if(ret != NLK_SUCCESS) {
        NLK_ERROR_VOID("Invalid lookup", ret);
    }
#else
    /* copy content from index columns to the ouput rows */
    nlk_array_copy_row_vector(output, 0, layer->weights, index); 
#endif

}

/** 
 * Lookup Layer forward pass with input (not first layer)
 * output = input * layer->weights for index (dot product)
 *
 * @param layer     the lookup layer
 * @param input     the input to the layer
 * @param index    the index to lookup 
 *
 * return no return, output is overwritten with result
 */
void
nlk_layer_lookup_forward(struct nlk_layer_lookup_t *layer, const NLK_ARRAY *input, 
                         const size_t index, nlk_real *output)
{
    *output = nlk_array_dot_carray(input, 
                &layer->weights->data[index * layer->weights->cols]);
}

/** 
 * Lookup Layer backward pass for accumulating gradient
 *
 * @param layer         the lookup layer
 * @param index         the index corresponing to this gradient
 * @param grad_out      gradient at the layer above (gradient at output)
 * @param gradient      gradient at input for this step (overwritten)
 * @param grad_acc      gradient at input accumulator (overwritten)
 * @param temp          array for weight update (overwritten/temporary)
 *
 * return no return, 
 */
void
nlk_layer_lookup_backprop_acc(struct nlk_layer_lookup_t *layer, const NLK_ARRAY *input,
                              const size_t index, const nlk_real grad_out, 
                              NLK_ARRAY *grad_acc)
{
    /* gradient at input (accumulate) */
    nlk_add_scaled_row_vector(grad_out, layer->weights, index, 1, grad_acc);
    

     /* learn weights for this layer */
    nlk_add_scaled_vector_row(grad_out, input, layer->weights, index);
}

/**
 *
 */
void
nlk_layer_lookup_backprop_lookup(struct nlk_layer_lookup_t *layer, 
                                 const size_t *indices, const size_t n_indices, 
                                 const NLK_ARRAY *grad_out)
{
    size_t ii;
    /* update weights */
    for(ii = 0; ii < n_indices; ii++) {
        nlk_array_add_carray(grad_out,
                    &layer->weights->data[indices[ii] * layer->weights->cols]);
    }

    /* no need to calc grad at input - 1st layer*/
}

/**
 *
 */
void
nlk_layer_lookup_backprop_lookup_one(struct nlk_layer_lookup_t *layer, 
                                     const size_t index, 
                                     const NLK_ARRAY *grad_out)
{
    /* update weights */
    nlk_array_add_carray(grad_out,
                &layer->weights->data[index * layer->weights->cols]);

    /* no need to calc grad at input - 1st layer*/
}

/** 
 * Free Lookup Layer memory 
 *
 * @param layer the Lookup Layer
 */
void 
nlk_layer_lookup_free(struct nlk_layer_lookup_t *layer)
{
    nlk_array_free(layer->weights);
    free(layer);
}

/** 
 * Save a lookup layer to disk
 *
 * @param filepath  the path of the output file to open
 * @param format    file format (word2vec compatible text, bin or NKL bin)
 * @param layer     the layer to save
 */
int
nlk_layer_lookup_export(char *filepath, const nlk_Format format,
                        struct nlk_vocab_t **vocab, struct nlk_layer_lookup_t *layer)
{
    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }
    nlk_layer_lookup_export_file(out, format, vocab, layer);
    fclose(out);
    return NLK_SUCCESS;
}

/** 
 * Export a lookup layer to a file pointer
 *
 * @param out       the file pointer to save to
 * @param format    file format (word2vec compatible text, bin or NKL bin)
 * @param layer     the layer to save
 *
 */
void
nlk_layer_lookup_export_file(FILE *out, nlk_Format format, 
                             struct nlk_vocab_t **vocab, 
                             struct nlk_layer_lookup_t *layer)
{
    size_t w_idx;
    size_t cc;
    struct nlk_vocab_t *vi;
    size_t vocab_size = layer->weights->rows;
    size_t layer_size = layer->weights->cols;

    /* print header */
    if(format == NLK_FILE_W2V_TXT || format == NLK_FILE_W2V_BIN) {
        fprintf(out, "%zu %zu\n", vocab_size, layer_size);
        /** @section W2V compatible */
        for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
            fprintf(out, "%s ", vi->word);
            w_idx = vi->index;
            
            if(format == NLK_FILE_W2V_BIN) { 
                /** @subsection W2V BIN */
                for(cc = 0; cc < layer_size; cc++) {
                    fwrite(&layer->weights->data[w_idx * layer_size + cc], 
                           sizeof(nlk_real), 1, out);
                } /* end of word weight */
            }
            else {
                /** @subsection W2V TXT */
                for(cc = 0; cc < layer_size; cc++) {
                    fprintf(out, "%lf ", layer->weights->data[w_idx * 
                                                              layer_size +
                                                              cc]);
                }
            }
            fprintf(out, "\n");
        }
    }
    else {
        /** @section NLK binary format
         * Does not use the vocabulary
         */
        nlk_array_save(layer->weights, out);
    }
}

/**
 * Save a lookp layer to a file pointer
 *
 * @param layer the lookup layer
 */
void
nlk_layer_lookup_save(struct nlk_layer_lookup_t *layer, FILE *fp)
{
    nlk_array_save(layer->weights, fp);
}

/** 
 * Load lookup layer from a file pointer (NLK_FILE_BIN)
 *
 * @param fp    the file pointer
 *
 * @return the loaded layer
 */
struct nlk_layer_lookup_t * 
nlk_layer_lookup_load(FILE *in)
{
    struct nlk_layer_lookup_t *layer;

    NLK_ARRAY *weights = nlk_array_load(in);
    if(weights == NULL) {
        return NULL;
    }
    layer = nlk_layer_lookup_create_from_array(weights);

    return layer;
}

/** 
 * Load lookup layer from a file path (NLK_FILE_BIN)
 *
 * @param filepath  the path of the file
 *
 * @return the loaded layer
 */
struct nlk_layer_lookup_t * 
nlk_layer_lookup_load_path(char *filepath)
{
    FILE *fp = fopen(filepath, "rb");
    if(fp == NULL) {
        NLK_ERROR_NULL("unable to open file.", NLK_FAILURE);
        /* unreachable */
    }
    return nlk_layer_lookup_load(fp);
}



/** 
 * Create a Linear Layer
 *
 * @param input_size    length of each input vector
 * @param layer_size    the size of this layer  
 * @param bias          if true this layer has a bias vector
 *
 * @return the Linear_Layer or NULL if creation failed
 */
NLK_LAYER_LINEAR *
nlk_layer_linear_create(const size_t input_size,  const size_t layer_size, 
                        bool bias)
{
    NLK_LAYER_LINEAR *layer;
   
    /* Allocate memory for struct, create the members */
    layer = (NLK_LAYER_LINEAR *) malloc(sizeof(NLK_LAYER_LINEAR));
    if(layer == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for layer struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    layer->weights = nlk_array_create(input_size, layer_size);
    if(bias) {
        layer->bias = nlk_array_create(layer_size, 1);
    } else {
        layer->bias = NULL;
    }
    layer->output = nlk_array_create(layer_size, 1);
    layer->grad_in = nlk_array_create(input_size, 1);

    return layer;
}

/** @fn nlk_layer_linear_init_sigmoid(NLK_LAYER_LINEAR *layer)
 * Initializes the linear layer weights 
 *
 *
 * This assumes the transfer function that follows is the sigmoid
 * Initializations is done by drawing from a uniform distribution in the range
 * [ -4 * sqrt(6 / (fan_in + fan_out)), 4 * sqrt(6 / (fan_in + fan_out)) )
 * This strategy follows:
 *
 *  Bengio, X. Glorot, 
 *  Understanding the difficulty of training deep feedforward neuralnetworks, 
 *  AISTATS 2010
 * 
 * The bias array is simply zeroed.
 *
 * @param layer     the Linear Layer to initialize
 *
 * @return no return, the lookup layer's weight matrix will be overwritten
 */
void
nlk_layer_linear_init_sigmoid(NLK_LAYER_LINEAR *layer, tinymt32_t *rng)
{
    nlk_real l = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_real h = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h, rng);
    if(layer->bias != NULL) {
        nlk_array_zero(layer->bias);
    }
}

/** @fn void nlk_layer_linear_forward(const NLK_LAYER_LINEAR *layer, 
 *                                    const NLK_ARRAY *input)
 * Linear Layer forward pass f(x) = W.x + b 
 * sets layer->output = layer->weights * input + layer->bias
 *
 * @param layer the linear layer
 * @param input the input to the layer
 *
 * return no return, layer->output (vector) is overwritten with result
 */
void
nlk_layer_linear_forward(const NLK_LAYER_LINEAR *layer, const NLK_ARRAY *input)
{
    /* bias needs to be added to matrix-vector product */
    if(layer->bias != NULL) {
        memcpy(layer->output, layer->bias,
               layer->output->rows * sizeof(nlk_real));
    } else {
        nlk_array_zero(layer->output);
    }

    nlk_matrix_vector_multiply_add(layer->weights, NLK_NOTRANSPOSE,
                                   input, layer->output);
}

/** @fn void nlk_layer_linear_backprop(NLK_LAYER_LINEAR *layer,
 *                                const NLK_ARRAY *input, 
 *                                const NLK_ARRAY *grad_out)
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
                          const NLK_ARRAY *grad_out)
{
    /* gradient (at input)  */
    nlk_matrix_vector_multiply_add(layer->weights, NLK_NOTRANSPOSE, 
                                   grad_out, layer->grad_in);
    /* update weights */
    nlk_vector_transposed_multiply_add(grad_out, input, layer->weights);
    /* update bias */
    if(layer->bias != NULL) {
        nlk_array_add(grad_out, layer->bias);  
    }
}

/** @fn void nlk_layer_linear_free(NLK_LAYER_LINEAR *layer)
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
    nlk_array_free(layer->output);
    nlk_array_free(layer->grad_in);

    free(layer);
}
