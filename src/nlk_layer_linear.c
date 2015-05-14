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
    layer = (struct nlk_layer_lookup_t *) 
                malloc(sizeof(struct nlk_layer_lookup_t));

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
    layer = (struct nlk_layer_lookup_t *) 
                malloc(sizeof(struct nlk_layer_lookup_t));

    if(layer == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for layer struct",
                       NLK_ENOMEM);
        /* unreachable */
    }

    layer->weights = weights;

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
nlk_layer_lookup_resize(struct nlk_layer_lookup_t *layer, 
                        const size_t table_size)
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
 * Initializes the lookup layer weights (word2vec) 
 * Initializations is done by drawing from a uniform distribution in the range
 * [-0.5 / layer_size, 0.5 / layer_size )
 * This follows the word2vec initialization for the lookup layer
 *
 * @param layer     the Lookup Layer to initialize
 *
 * @return no return, the lookup layer's weight matrix will be overwritten
 */
void
nlk_layer_lookup_init(struct nlk_layer_lookup_t *layer)
{
    nlk_real low = -0.5 / layer->weights->cols;
    nlk_real high = 0.5 / layer->weights->cols; 
    nlk_array_init_uniform(layer->weights, low, high);
}

/** 
 * Initializes lookup layer weights (word2vec) for a given weights array
 * Initializations is done by drawing from a uniform distribution in the range
 * [-0.5 / layer_size, 0.5 / layer_size )
 * This follows the word2vec initialization for the lookup layer
 *
 * @param weights     the weights to initialize
 *
 * @return no return, the weight matrix will be overwritten
 */
void
nlk_layer_lookup_init_array(NLK_ARRAY *weights)
{
    nlk_real low = -0.5 / weights->cols;
    nlk_real high = 0.5 / weights->cols; 
    nlk_array_init_uniform(weights, low, high);
}

/**
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
nlk_layer_lookup_init_sigmoid(struct nlk_layer_lookup_t *layer)
{
    nlk_real l = -4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows + 
                                                layer->weights->cols));
    nlk_real h =  4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows +
                                                layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h);
}

/** 
 * Same as above (nlk_layer_lookup_init) but only for weights after a given row
 *
 * @param layer the lookup layer
 * @param from  the starting row
 */
void
nlk_layer_lookup_init_sigmoid_from(struct nlk_layer_lookup_t *layer, 
                                   size_t from)
{
    size_t len = (layer->weights->rows - from) * layer->weights->cols;
    nlk_real l = -4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows + 
                                                layer->weights->cols));
    nlk_real h =  4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows +
                                                layer->weights->cols));
    nlk_carray_init_uniform(&layer->weights->data[from * layer->weights->cols], 
                            l, h, len);
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
nlk_layer_lookup_forward_lookup(struct nlk_layer_lookup_t *layer, 
                                const size_t *indices, 
                                const size_t n_indices, NLK_ARRAY *output)
{
    size_t ii;

#ifndef NCHECKS
    if(n_indices == 0) {
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

    /* add averaged word vectors */
    nlk_array_zero(output);
    for(ii = 0; ii < n_indices; ii++) {
       nlk_add_scaled_row_vector(s, layer->weights, indices[ii], 0,  output);
    }
}


/** 
 * Lookup Layer forward pass with ids for 1st layer vectors and a 
 * Paragraph Vector. They are averaged together. 
 *
 * @param layer         the lookup layer
 * @param indices       the indices to lookup 
 * @param n_indices     the number of indices in the indices array (array size)
 * @param output        the paragraph vector overwritten with the output of the 
 *                      forward pass
 *
 * return no return, output is overwritten with result: 1 * layer->cols
 */
void
nlk_layer_lookup_forward_lookup_avg_p(struct nlk_layer_lookup_t *layer, 
                                      const size_t *indices,
                                      const size_t n_indices, 
                                      NLK_ARRAY *output)
{
    size_t ii;
    nlk_real s;

    if(n_indices == 0) {
        return; /* nothing to do */
    }

    s =  1.0 / (nlk_real) (n_indices + 1);

    /* add averaged word vectors */
    for(ii = 0; ii < n_indices; ii++) {
       nlk_add_scaled_row_vector(s, layer->weights, indices[ii], 0,  output);
    }
}

/** 
 * Lookup Layer forward pass with just ids (1st layer) but concatenates them
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
nlk_layer_lookup_forward_lookup_concat(struct nlk_layer_lookup_t *layer, 
                                       const size_t *indices,
                                       const size_t n_indices, 
                                       NLK_ARRAY *output)
{
    size_t ii;
    size_t cols = layer->weights->cols;

#ifndef NCHECKS
    if(n_indices == 0) {
        NLK_ERROR_VOID("empty input - indices parameter must be non-zero",
                       NLK_EINVAL);
        /* unreachable */
    }
    if(n_indices * cols > output->rows * output->cols) {
       NLK_ERROR_VOID("output array too small for concatenation",
                       NLK_EBADLEN);
    }
    if(n_indices * cols < output->rows * output->cols) {
       NLK_ERROR_VOID("input array too small for concatenation",
                       NLK_EBADLEN);
    }
#endif

    /* copy content from indices to the ouput */
    for(ii = 0; ii < n_indices; ii++) {
        /* cblas_scopy(layer->weights->cols, &layer->weights->data[indices[ii]
         *             layer->weights->cols], 1, &output->data[ii * cols], 1);
         */
        nlk_array_copy_row_carray(layer->weights, indices[ii], 
                                  &output->data[ii * cols]); 
    }
}

/**
 * PV should already be in output
 */
void
nlk_layer_lookup_forward_lookup_concat_p(struct nlk_layer_lookup_t *layer, 
                                        const size_t *indices,
                                        const size_t n_indices, 
                                        NLK_ARRAY *output)
{
    size_t ii;
    size_t cols = layer->weights->cols;

#ifndef NCHECKS
    if(n_indices == 0) {
        NLK_ERROR_VOID("empty input - indices parameter must be non-zero",
                       NLK_EINVAL);
        /* unreachable */
    }
    if(n_indices * cols + cols > output->rows * output->cols) {
       NLK_ERROR_VOID("output array too small for concatenation",
                       NLK_EBADLEN);
    }
    if(n_indices * cols + cols < output->rows * output->cols) {
       NLK_ERROR_VOID("input array too small for concatenation",
                       NLK_EBADLEN);
    }
#endif

    /* copy content from indices to the ouput */
    for(ii = 0; ii < n_indices; ii++) {
        /* cblas_scopy(layer->weights->cols, &layer->weights->data[indices[ii]
         *             layer->weights->cols], 1, &output->data[ii * cols], 1);
         */
        nlk_array_copy_row_carray(layer->weights, indices[ii], 
                                  &output->data[ii * cols + cols]); 
    }
}


/** 
 * Lookup Layer forward pass with just one id (1st layer) followed by a
 * transformation of the output vector from a column vector to a row vector
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
nlk_layer_lookup_forward(struct nlk_layer_lookup_t *layer, 
                         const NLK_ARRAY *input, 
                         const size_t index, nlk_real *output)
{
    *output = nlk_array_dot_carray(input, 
                &layer->weights->data[index * layer->weights->cols]);
}

/** 
 * Lookup Layer backward pass for accumulating gradient
 *
 * @param layer         the lookup layer
 * @param update        update weights for this layer
 * @param index         the index corresponing to this gradient
 * @param grad_out      gradient at the layer above (gradient at output)
 * @param grad_acc      gradient at input accumulator (overwritten)
 */
void
nlk_layer_lookup_backprop_acc(struct nlk_layer_lookup_t *layer, 
                              const bool update, const NLK_ARRAY *input, 
                              const size_t index, 
                              const nlk_real grad_out, NLK_ARRAY *grad_acc)
{
    /* gradient at input (accumulate) */
    nlk_add_scaled_row_vector(grad_out, layer->weights, index, 1, grad_acc);
    

     /* learn weights for this layer */
    if(update) {
        nlk_add_scaled_vector_row(grad_out, input, layer->weights, index);
    }
}

/**
 * Lookup Layer backward pass for the first layer of the net (no need to acc)
 *
 * @param layer         the lookup layer
 * @param indices       the indices corresponing to this gradient
 * @param grad_out      gradient at the layer above (gradient at output)
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
 * Lookup Layer backward pass for the first layer of the net (no need to acc) - 
 * concatenate version
 *
 * @param layer         the lookup layer
 * @param indices       the indices corresponing to this gradient
 * @param start_at      where the indices start at in the gradient
 * @param grad_out      gradient at the layer above (gradient at output)
 */
void
nlk_layer_lookup_backprop_lookup_concat(struct nlk_layer_lookup_t *layer, 
                                        const size_t *indices, 
                                        const size_t n_indices,
                                        const size_t start_at,
                                        const NLK_ARRAY *grad_out)
{
    size_t ii;
    const size_t cols = layer->weights->cols;

#ifndef NCHECKS
    if(n_indices == 0) {
        NLK_ERROR_VOID("empty input - indices parameter must be non-zero",
                       NLK_EINVAL);
        /* unreachable */
    }
    if(n_indices * cols + start_at * cols > grad_out->rows * grad_out->cols) {
        NLK_ERROR_VOID("gradient smaller than input", NLK_ERANGE);
    }
#endif


    /* update weights */
    for(ii = 0; ii < n_indices; ii++) {
#ifndef NCHECKS
        if(indices[ii] >= layer->weights->rows) {
            NLK_ERROR_VOID("index out of range", NLK_ERANGE);
            /* unreachable */
    }
#endif
        cblas_saxpy(cols, 1, &grad_out->data[ii * cols + start_at * cols], 1, 
                    &layer->weights->data[indices[ii] * cols], 1); 
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
 *
 */
void
nlk_layer_lookup_backprop_lookup_concat_one(struct nlk_layer_lookup_t *layer, 
                                            const size_t index, 
                                            const size_t grad_index,
                                            const NLK_ARRAY *grad_out)
{
    const size_t cols = layer->weights->cols;

    /* update weights */
    cblas_saxpy(cols, 1, &grad_out->data[grad_index * cols], 1, 
                &layer->weights->data[index * cols], 1); 
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
 * Save a lookp layer to a file pointer
 *
 * @param layer the lookup layer
 * @param fp    the file pointer
 */
void
nlk_layer_lookup_save(struct nlk_layer_lookup_t *layer, FILE *fp)
{
    nlk_array_save(layer->weights, fp);
}

/**
 * Save a lookp layer to a file
 *
 * @param layer         the lookup layer
 * @param file_path     the file path
 */
void 
nlk_layer_lookup_save_path(struct nlk_layer_lookup_t *layer, char *filepath)
{
    FILE *fp = fopen(filepath, "wb");
    if(fp == NULL) {
        NLK_ERROR_VOID("unable to open file.", NLK_FAILURE);
        /* unreachable */
    }

    nlk_layer_lookup_save(layer, fp);
    fclose(fp);
}

/**
 * Save part of a lookp layer to a file (save row weight vectors)
 *
 * @param layer         the lookup layer
 * @param file_path     the file path
 * @param start         the first row to save (0 index)
 * @param end           the first row not to save
 */
void
nlk_layer_lookup_save_rows_path(struct nlk_layer_lookup_t *layer, 
                                char *filepath, size_t start, size_t end) {

    FILE *fp = fopen(filepath, "wb");
    if(fp == NULL) {
        NLK_ERROR_VOID("unable to open file.", NLK_FAILURE);
        /* unreachable */
    }

    nlk_array_save_rows(layer->weights, fp, start, end);
    fclose(fp);
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
struct nlk_layer_linear_t *
nlk_layer_linear_create(const size_t input_size,  const size_t layer_size, 
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

/** 
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
nlk_layer_linear_init_sigmoid(struct nlk_layer_linear_t *layer)
{
    nlk_real l = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_real h = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h);
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
