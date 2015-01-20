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


/** @file nlk_neuralnet.c
* Neural Net
*/


#include <stdio.h>

#include "nlk_err.h"
#include "nlk_layer_linear.h"

#include "nlk_neuralnet.h"


/**
 * Create an empty neural network
 *
 * @param n_layers  the number of layers the network will contain
 *
 * @return the neural network structure or NULL
 */
struct nlk_neuralnet_t *
nlk_neuralnet_create(size_t n_layers)
{
    struct nlk_neuralnet_t *nn;

    nn = (struct nlk_neuralnet_t *) malloc(sizeof(struct nlk_neuralnet_t));
    if(nn == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for neural net", 
                        NLK_ENOMEM);
        /* unreachable */
    }

    nn->n_layers = n_layers;
    nn->pos = 0;

    nn->layers = (union nlk_layer_t *) malloc(sizeof(union nlk_layer_t *) *
                                              n_layers);
    if(nn->layers == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for neural net", 
                        NLK_ENOMEM);
        /* unreachable */
    }
    nn->types = calloc(n_layers, sizeof(unsigned short int));
    if(nn->types == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for neural net", 
                        NLK_ENOMEM);
        /* unreachable */
    }

    return nn;
}

/**
 * Add a Lookup Layer to a Neural Network
 *
 * @param nn    the neural network
 * @param lk    the lookup layer
 */
void 
nlk_neuralnet_add_layer_lookup(struct nlk_neuralnet_t *nn,
                               struct nlk_layer_lookup_t *lk)
{
    if(nn->pos == nn->n_layers) {
        NLK_ERROR_VOID("unable to add layer because neural network is 'full'",
                        NLK_FAILURE);
        /* unreachable */
    }

    nn->layers[nn->pos].lk = lk;
    nn->types[nn->pos] = NLK_LAYER_LOOKUP_TYPE;
    nn->pos++;
}

/**
 * Add a Linear Layer to a Neural Network
 *
 * @param nn    the neural network
 * @param ll    the linear layer
 */
void 
nlk_neuralnet_add_layer_linear(struct nlk_neuralnet_t *nn,
                               struct nlk_layer_linear_t *ll)
{
    if(nn->pos == nn->n_layers) {
        NLK_ERROR_VOID("unable to add layer because neural network is 'full'",
                        NLK_FAILURE);
        /* unreachable */
    }

    nn->layers[nn->pos].ll = ll;
    nn->types[nn->pos] = NLK_LAYER_LINEAR_TYPE;
    nn->pos++;
}

/**
 * Free memory used by a neural network (including its layers)
 *
 * @param nn    the neural network
 */
void
nlk_neuralnet_free(struct nlk_neuralnet_t *nn)
{
    size_t ii;

    /* free each layer */
    for(ii = 0; ii < nn->n_layers; ii++) {
        switch(nn->types[ii]) {
            case NLK_LAYER_LOOKUP_TYPE:
                if(nn->layers[ii].lk != NULL) {
                    nlk_layer_lookup_free(nn->layers[ii].lk);
                }
                break;
            case NLK_LAYER_LINEAR_TYPE:
                if(nn->layers[ii].ll != NULL) {
                    nlk_layer_linear_free(nn->layers[ii].ll);
                }
                break;
            default:
                NLK_ERROR_VOID("unknown layer type", NLK_FAILURE);
                /* unreachable */
        }
    }

    free(nn->layers);
    free(nn->types);

    free(nn);
}

/**
 * Save Neural Network to a file at a given path
 *
 * @param nn        the neural network to save
 * @param filepath  the path of the file to save to
 */
int
nlk_neuralnet_save_path(struct nlk_neuralnet_t *nn, char *filepath)
{
    FILE *fp;
    int ret;

    fp = fopen(filepath, "wb");
    if(fp == NULL) {
        NLK_ERROR("unable to open file", NLK_FAILURE);
    }
    
    ret = nlk_neuralnet_save(nn, fp);

    fclose(fp);
    return ret;
}

/**
 * Save Neural Network to a File Pointer
 *
 * @param nn    the neural network to save
 * @param fp    the file pointer to save to
 */
int
nlk_neuralnet_save(struct nlk_neuralnet_t *nn, FILE *fp)
{
    size_t ii;

    /* write header */
    fprintf(fp, "%zu\n", nn->n_layers);
    for(ii = 0; ii < nn->n_layers; ii++) {
        fprintf(fp, "%d\n", nn->types[ii]);
    }

    /* write layers */
    for(ii = 0; ii < nn->n_layers; ii++) {
        switch(nn->types[ii]) {
            case NLK_LAYER_LOOKUP_TYPE:
                nlk_layer_lookup_save(nn->layers[ii].lk, fp);
                break;
            default:
                NLK_ERROR("invalid layer type", NLK_FAILURE);
                /* unreachable */
        }
    }
    return 0;
}

/**
 * Load Neural Network from a file path
 *
 * @param filepath  the path of the file to load from
 *
 * @return pointer to the neural network structure or NULL
 */
struct nlk_neuralnet_t *
nlk_neuralnet_load_path(char *filepath) 
{
    struct nlk_neuralnet_t *nn = NULL;
    FILE *fp;

    fp = fopen(filepath, "rb");
    if(fp == NULL) {
        NLK_ERROR_NULL("unable to open file", NLK_FAILURE);
    }
    
    nn = nlk_neuralnet_load(fp);
    
    fclose(fp);

    return nn;
}

/**
 * Load Neural Network from a File Pointer
 *
 * @param fp    the file pointer to load from
 *
 * @return pointer to the neural network structure or NULL
 */
struct nlk_neuralnet_t *
nlk_neuralnet_load(FILE *fp) 
{
    struct nlk_neuralnet_t *nn = NULL;
    size_t n_layers;
    size_t ii;
    int ret;

    /* read header */
    ret = fscanf(fp, "%zu\n", &n_layers);
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    nn = nlk_neuralnet_create(n_layers);

    for(ii = 0; ii < nn->n_layers; ii++) {
        ret = fscanf(fp, "%hu\n", &nn->types[ii]);
        if(ret <= 0) {
            goto nlk_neuralnet_load_err_head;
        }
    }

    /* read layers */
    for(ii = 0; ii < nn->n_layers; ii++) {
        switch(nn->types[ii]) {
            case NLK_LAYER_LOOKUP_TYPE:
                nn->layers[ii].lk = nlk_layer_lookup_load(fp);
                if(nn->layers[ii].lk == NULL) {
                    goto nlk_neuralnet_load_err;
                }
                break;
            default:
                NLK_ERROR_NULL("invalid layer type", NLK_FAILURE);
                /* unreachable */
        }
    }

    return nn;

    /* generic header error */
nlk_neuralnet_load_err_head:
    if(nn != NULL) {
        nlk_neuralnet_free(nn);
    }
    NLK_ERROR_NULL("unable to read header information", NLK_FAILURE);
    /* unreachable */

    /* generic layer error */
nlk_neuralnet_load_err:
    nlk_neuralnet_free(nn);
    NLK_ERROR_NULL("unable to read neural network", NLK_FAILURE);
    /* unreachable */
}
