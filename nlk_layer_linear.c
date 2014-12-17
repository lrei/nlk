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


/** @fn nlk_Layer_Lookup *nlk_layer_lookup_create(const size_t table_size, 
 *                                                const size_t layer_size,
 *                                                const size_t n_indices)
 * Create a Lookup Layer 
 *
 * @param table_size    e.g. the size of the vocabulary 
 * @param layer_size    the size of this layer  
 *
 * @return pointer to a Lookup_Layer or NULL if creation failed
 *
 * @note
 * The number of indices can be something like a word context size.
 * Weights are initialized to zero by this function.
 * @endnote
 */
nlk_Layer_Lookup *
nlk_layer_lookup_create(const size_t table_size, const size_t layer_size)
{
    nlk_Layer_Lookup *layer;
   
    /*
     * Allocate memory for struct, create the members
     */
    layer = (nlk_Layer_Lookup *) malloc(sizeof(nlk_Layer_Lookup));
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

/** @fn void nlk_layer_lookup_init(nlk_Layer_Lookup *layer)
 * Initializes the lookup layer weights 
 * Initializations is done by drawing from a uniform distribution in the range
 * [-0.5 / layer_size, 0.5 / layer_size )
 * This follows the word2vec initialization for the lookup layer
 *
 * @param layer     the Lookup Layer to initialize
 *
 * @return no return, the lookup layer's weight matrix will be overwritten
 */
void
nlk_layer_lookup_init(nlk_Layer_Lookup *layer)
{
    nlk_real low = -0.5 / layer->weights->cols;
    nlk_real high = 0.5 / layer->weights->cols; 
    nlk_array_init_uniform(layer->weights, low, high);
}


/** @fn nlk_layer_lookup_init_sigmoid(nlk_Layer_Linear *layer)
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
nlk_layer_lookup_init_sigmoid(nlk_Layer_Lookup *layer)
{
    nlk_real l = -4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows + 
                                                layer->weights->cols));
    nlk_real h =  4.0 * sqrtf(6.0 / (nlk_real) (layer->weights->rows +
                                                layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h);
}

/** @fn nlk_layer_lookup_forward_lookup(nlk_Layer_Lookup *layer, 
 *                                      const size_t *indices, 
 *                                      size_t n_indices)
 * Lookup Layer forward pass with just ids (1st layer)
 *
 * @param layer         the lookup layer
 * @param indices       the indices to lookup 
 * @param n_indices     the number of indices in the indices array (array size)
 * @param output_view   the output of the lookup forward pass
 *
 * return no return, output_view is overwritten with result
 */
void
nlk_layer_lookup_forward_lookup(nlk_Layer_Lookup *layer, const size_t *indices, 
                                const size_t n_indices, nlk_Array *output)
{
    size_t ii;

    if (n_indices == 0) {
        NLK_ERROR_VOID("empty input - indices parameter must be non-zero",
                       NLK_EINVAL);
        /* unreachable */
    }

    /* copy content from indices to the ouput */
    for(ii = 0; ii < n_indices; ii++) {
        nlk_array_copy_row(output, ii , layer->weights, indices[ii]); 
    }

}

/** @fn nlk_layer_lookup_forward(nlk_Layer_Lookup *layer, nlk_Array *input,
 *                               size_t *indices, size_t n_indices)
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
nlk_layer_lookup_forward(nlk_Layer_Lookup *layer, const nlk_Array *input, 
                         const size_t index, nlk_real *output)
{
    *output = nlk_array_dot_carray(input, 
                &layer->weights->data[index * layer->weights->cols]);
}

/** @fn  void nlk_layer_lookup_backprop_accumulate(nlk_Layer_Lookup *layer, 
 *                                          const size_t index, 
 *                                          const nlk_real gradient)
 * Lookup Layer backward pass for accumulating gradient
 *
 * @param layer         the lookup layer
 * @param index         the index corresponing to this gradient
 * @param grad_out      gradient at the layer above (gradient at output)
 * @param gradient      gradient at input for this step (overwritten)
 * @param gradient_acc  gradient at input accumulator (overwritten)
 * @param temp          array for weight update (overwritten/temporary)
 *
 * return no return, 
 */
void
nlk_layer_lookup_backprop_acc(nlk_Layer_Lookup *layer, const nlk_Array *input,
                              const size_t index, const nlk_real grad_out, 
                              nlk_Array *gradient, nlk_Array *gradient_acc, 
                              nlk_Array *temp)
{
    /* gradient at input (accumulate) */
    nlk_array_copy_row(gradient, 0, layer->weights, index);
    nlk_array_scale(grad_out, gradient);
    nlk_array_add(gradient, gradient_acc);

    /* learn weights for this layer */
    nlk_array_copy(temp, input);
    nlk_array_scale(grad_out, temp);
    nlk_vector_add_row(temp, layer->weights, index);
}


void
nlk_layer_lookup_backprop_lookup(nlk_Layer_Lookup *layer, 
                                 const size_t *indices, const size_t n_indices, 
                                 const nlk_Array *grad_out)
{
    size_t ii;
    /* update weights */
    for(ii = 0; ii < n_indices; ii++) {
        nlk_array_add_carray(grad_out,
                &layer->weights->data[indices[ii] * layer->weights->cols]);
    }

    /* no need to calc grad at input - 1st layer*/
}

/** @fn void nlk_layer_lookup_free(nlk_Layer_Lookup *layer)
 * Free Lookup Layer memory 
 *
 * @param layer the Lookup Layer
 */
void 
nlk_layer_lookup_free(nlk_Layer_Lookup *layer)
{
    nlk_array_free(layer->weights);
    free(layer);
}

/** @fn int nlk_layer_lookup_save(const char *filepath,
 *                                const nlk_Format format, 
 *                                const nlk_Layer_Lookup *layer)
 * Save a lookup layer to disk - word2vec compatible file
 *
 * @note
 * I don't like the word2vec binary format because it needlessly mixes the 
 * vocabulary with the weights making it far more complicated and to read than 
 * it would otherwise be.
 * @endnote
 */
int
nlk_layer_lookup_save(char *filepath, const nlk_Format format,
                      nlk_Vocab **vocab, nlk_Layer_Lookup *layer)
{
    size_t w_idx;
    size_t cc;
    nlk_Vocab *vi;
    size_t vocab_size = layer->weights->rows;
    size_t layer_size = layer->weights->cols;

    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }

    /* print header */
    fprintf(out, "%zu %zu\n", vocab_size, layer_size);

    if(format == NLK_FILE_W2V_TXT || format == NLK_FILE_W2V_BIN) {
        for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
            fprintf(out, "%s ", vi->word);
            w_idx = vi->index;
            
            if(format == NLK_FILE_W2V_BIN) {
                for(cc = 0; cc < layer_size; cc++) {
                    fwrite(&layer->weights->data[w_idx * layer_size + cc], 
                           sizeof(nlk_real), 1, out);
                } /* end of word weight */
            }
            else {
                for(cc = 0; cc < layer_size; cc++) {
                    fprintf(out, "%lf ", layer->weights->data[w_idx * layer_size +
                                                             cc]);
                }
            }
            fprintf(out, "\n");
        }
    } else {
        fwrite(layer->weights->data, sizeof(nlk_real),
               layer->weights->rows * layer->weights->cols, out);
    }
    fclose(out);
}

/** @fn nlk_Layer_Lookup *nlk_layer_lookup_load(char *filepath)
 * Load lookup layer from a file (NLK_FILE_BIN)
 *
 * @param filepath  the path of the file
 *
 * @return the loaded layer
 */
nlk_Layer_Lookup * 
nlk_layer_lookup_load(char *filepath)
{
    nlk_Layer_Lookup *layer;
    size_t rows;
    size_t cols;
    size_t ret;

    FILE *in = fopen(filepath, "rb");
    if(in == NULL) {
        NLK_ERROR_NULL(strerror(errno), errno);
        /* unreachable */
    }
    
    /* read header */
    ret = fscanf(in, "%zu", &rows);
    ret = fscanf(in, "%zu", &cols);
    ret = fgetc(in); /* the newline */

    layer = nlk_layer_lookup_create(rows, cols);

    ret = fread(layer->weights->data, sizeof(nlk_real), rows * cols, in);
    if(ret != rows * cols) {
        NLK_ERROR_NULL("file length does not match header information",
                        NLK_FAILURE);
        /* unreachable */
    }

    return layer;
}


/** @fn nlk_Layer_Linear *nlk_layer_linear_create(const size_t input_size,  
 *                                               const size_t layer_size, 
 *                                               bool bias)
 * Create a Linear Layer
 *
 * @param input_size    length of each input vector
 * @param layer_size    the size of this layer  
 * @param bias          if true this layer has a bias vector
 *
 * @return the Linear_Layer or NULL if creation failed
 */
nlk_Layer_Linear *
nlk_layer_linear_create(const size_t input_size,  const size_t layer_size, 
                        bool bias)
{
    nlk_Layer_Linear *layer;
   
    /* Allocate memory for struct, create the members */
    layer = (nlk_Layer_Linear *) malloc(sizeof(nlk_Layer_Linear));
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

/** @fn nlk_layer_linear_init_sigmoid(nlk_Layer_Linear *layer)
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
nlk_layer_linear_init_sigmoid(nlk_Layer_Linear *layer)
{
    nlk_real l = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_real h = -4 * sqrtf(6 / (layer->weights->rows + layer->weights->cols));
    nlk_array_init_uniform(layer->weights, l, h);
    if(layer->bias != NULL) {
        nlk_array_zero(layer->bias);
    }
}

/** @fn void nlk_layer_linear_forward(const nlk_Layer_Linear *layer, 
 *                                    const nlk_Array *input)
 * Linear Layer forward pass f(x) = W.x + b 
 * sets layer->output = layer->weights * input + layer->bias
 *
 * @param layer the linear layer
 * @param input the input to the layer
 *
 * return no return, layer->output (vector) is overwritten with result
 */
void
nlk_layer_linear_forward(const nlk_Layer_Linear *layer, const nlk_Array *input)
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

/** @fn void nlk_layer_linear_backprop(nlk_Layer_Linear *layer,
 *                                const nlk_Array *input, 
 *                                const nlk_Array *grad_out)
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
nlk_layer_linear_backprop(nlk_Layer_Linear *layer, const nlk_Array *input, 
                          const nlk_Array *grad_out)
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

/** @fn void nlk_layer_linear_free(nlk_Layer_Linear *layer)
 * Free Linear Layer memory 
 *
 * @param layer    the Linear Layer
 */
void 
nlk_layer_linear_free(nlk_Layer_Linear *layer)
{
    nlk_array_free(layer->weights);
    if(layer->bias != NULL) {
        nlk_array_free(layer->bias);
    }
    nlk_array_free(layer->output);
    nlk_array_free(layer->grad_in);

    free(layer);
}
