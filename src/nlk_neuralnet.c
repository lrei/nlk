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
#include <stdlib.h>
#include <inttypes.h>

#include "nlk_err.h"
#include "nlk_layer_lookup.h"
#include "nlk_layer_linear.h"

#include "nlk_neuralnet.h"


/**
 * Is this model a Paragraph model?
 */
bool
nlk_neuralnet_is_paragraph_model(const NLK_LM model_type) 
{
    switch(model_type) {
        case NLK_PVDBOW:
            return true;
        case NLK_PVDM:
            return true;
        case NLK_PVDM_CONCAT:
            return true;
        case NLK_PVDM_SUM:
            return true;
        case NLK_CBOW:
            return false;
        case NLK_SKIPGRAM:
            return false;
        case NLK_MODEL_NULL:
            return false;
        default:
            NLK_ERROR("Invalid model type", NLK_EINVAL);
            /* unreachable */
    }

    /* should be unreachable */
    return false;
}

/**
 * Is this model a concatenate model?
 */
bool
nlk_neuralnet_is_concat_model(const NLK_LM model_type)
{
    switch(model_type) {
        case NLK_PVDM_CONCAT:
            return true;
        default:
            return false;
    }
}


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

    nn->words = NULL;
    nn->paragraphs = NULL;
    nn->vocab = NULL;

    if(n_layers > 0) {
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
    } else {
        nn->layers = NULL;
        nn->types = NULL;
    }

    return nn;
}


/**
 * Increase the maximum number of layers in a NN
 *
 * @param nn        the neural network
 * @param add_n     number of layers to expand by
 */
int
nlk_neuralnet_expand(struct nlk_neuralnet_t *nn, size_t add_n)
{
    /* calculate new size */
    size_t n_layers = nn->n_layers + add_n;

    /* realloc layers */
    union nlk_layer_t *layers = (union nlk_layer_t *) 
        realloc(nn->layers, sizeof(union nlk_layer_t *) * n_layers);

    if(layers == NULL) {
        NLK_ERROR("unable to allocate memory for neural net", 
                  NLK_ENOMEM);
        /* unreachable */
    }
    nn->layers = layers;


    /* realloc types */
    unsigned short int *types = (unsigned short int *) 
                                realloc(nn->types, sizeof(unsigned short int));

    if(types == NULL) {
        NLK_ERROR("unable to allocate memory for neural net", 
                  NLK_ENOMEM);
        /* unreachable */
    }
    nn->types = types;


    /* replace */
    nn->layers = layers;
    nn->types = nn->types;
    nn->n_layers = n_layers;

    return NLK_SUCCESS;
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
    nn->pos = nn->pos + 1;
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
    nn->pos = nn->pos + 1;
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

    if(nn->words != NULL) {
        nlk_layer_lookup_free(nn->words);

    }
    if(nn->paragraphs != NULL) {
        nlk_layer_lookup_free(nn->paragraphs);
    }

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
    nn = NULL;
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

    /** @section Write training options 
     */
    /* 1 - model type */
    fprintf(fp, "%d\n", nn->train_opts.model_type); 
    /* 2 - paragraph model? */
    fprintf(fp, "%d\n", nn->train_opts.paragraph); 
    /* 3 - window size */
    fprintf(fp, "%u\n", nn->train_opts.window); 
    /* 4 - sample */
    fprintf(fp, "%f\n", nn->train_opts.sample); 
    /* 5 - learn_rate */
    fprintf(fp, "%f\n", nn->train_opts.learn_rate); 
    /* 6 - hs? */
    fprintf(fp, "%d\n", nn->train_opts.hs); 
    /* 7 - negative */
    fprintf(fp, "%u\n", nn->train_opts.negative); 
    /* 8 - iterations (epochs) */
    fprintf(fp, "%u\n", nn->train_opts.iter); 
    /* 9 - vector_size */
    fprintf(fp, "%u\n", nn->train_opts.vector_size); 
    /* 10 - word_count */
    fprintf(fp, "%"PRIu64"\n", nn->train_opts.word_count); 

    /* write header: number of layers */
    fprintf(fp, "%zu\n", nn->n_layers);
    for(ii = 0; ii < nn->n_layers; ii++) {
        /* layer types */
        fprintf(fp, "%d\n", nn->types[ii]);
    }

    /* write vocabulary */
    nlk_vocab_save(&nn->vocab, fp);

    /* write word lookup */
    nlk_layer_lookup_save(nn->words, fp);

    /* write paragraph lookup */
    if(nn->train_opts.paragraph && nn->paragraphs != NULL) {
        nlk_layer_lookup_save(nn->paragraphs, fp);
    }

    /* write the hierarchical softmax layer */
    if(nn->train_opts.hs) {
        nlk_layer_lookup_save(nn->hs, fp);
    }

    /* write the negative sampling layer */
    if(nn->train_opts.hs) {
        nlk_layer_lookup_save(nn->neg, fp);
    }

    /* write the other layers */
    for(ii = 0; ii < nn->n_layers; ii++) {
        switch(nn->types[ii]) {
            case NLK_LAYER_LOOKUP_TYPE:
                nlk_layer_lookup_save(nn->layers[ii].lk, fp);
                break;
            case NLK_LAYER_LINEAR_TYPE:
                nlk_layer_linear_save(nn->layers[ii].ll, fp);
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
nlk_neuralnet_load_path(char *filepath, bool verbose) 
{
    struct nlk_neuralnet_t *nn = NULL;
    FILE *fp;

    fp = fopen(filepath, "rb");
    if(fp == NULL) {
        NLK_ERROR_NULL("unable to open file", NLK_FAILURE);
    }
    
    nn = nlk_neuralnet_load(fp, verbose);
    
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
nlk_neuralnet_load(FILE *fp, bool verbose) 
{
    struct nlk_neuralnet_t *nn = NULL;
    struct nlk_nn_train_t opts;
    struct nlk_context_opts_t ctx_opts;
    size_t n_layers;
    size_t ii;
    int ret;
    int tmp;

    /**
     * @section read training options and header
     */
    /* 1 - model type */
    ret = fscanf(fp, "%d\n", &tmp); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    opts.model_type = tmp;
    /* 2 - paragraph */
    ret = fscanf(fp, "%d\n", &tmp); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    opts.paragraph = tmp;
    /* 3 - window */
    ret = fscanf(fp, "%u\n", &opts.window); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* 4 - sample */
    ret = fscanf(fp, "%f\n", &opts.sample); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* 5 - learn rate */
    ret = fscanf(fp, "%f\n", &opts.learn_rate); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* 6 - hierarchical softmax */
    ret = fscanf(fp, "%d\n", &tmp); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    opts.hs = tmp;
    /* 7 - negative sampling */
    ret = fscanf(fp, "%u\n", &opts.negative); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* 8 - iterations */
    ret = fscanf(fp, "%u\n", &opts.iter); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* 9 - vector size */
    ret = fscanf(fp, "%u\n", &opts.vector_size); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* 10 - word count */
    ret = fscanf(fp, "%"SCNu64"\n", &opts.word_count); 
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }

    /**
     * @section create neural network and load weights
     */
    /* read header: number of layers */
    ret = fscanf(fp, "%zu\n", &n_layers);
    if(ret <= 0) {
        goto nlk_neuralnet_load_err_head;
    }
    /* create neural network */
    nn = nlk_neuralnet_create(n_layers);

    /* set training options */
    nn->train_opts = opts;

    /* set context options */
    nn->context_opts = ctx_opts;

    if(verbose) {
        printf("Loading Neural Network\nLayers: %zu\n", n_layers);
    }

    /* continue with header: layer types */
    for(ii = 0; ii < nn->n_layers; ii++) {
        ret = fscanf(fp, "%d\n", &tmp);
        if(ret <= 0) {
            goto nlk_neuralnet_load_err_head;
        }
        nn->types[ii] = tmp;
    }

    /* read vocabulary */
    nn->vocab = nlk_vocab_load(fp);

    /* read word lookup */
    nn->words = nlk_layer_lookup_load(fp);
    if(verbose) {
        printf("Loaded Word Table\n");
    }


    /* read paragraph lookup */
    if(nn->train_opts.paragraph) {
        nn->paragraphs = nlk_layer_lookup_load(fp);
        if(verbose) {
            printf("Loaded Paragraph Table\n");
        }
    } else {
        nn->paragraphs = NULL;
    }


    /* read hierarchical softmax */
    if(nn->train_opts.hs) {
       nn->hs = nlk_layer_lookup_load(fp);
        if(verbose) {
            printf("Loaded Hierarchical Softmax Layer\n");
        }
    } else {
        nn->hs = NULL;
    }


    /* read negative sampling layer */
    if(nn->train_opts.negative) {
       nn->neg = nlk_layer_lookup_load(fp);
       nn->neg_table = nlk_vocab_neg_table_create(&nn->vocab, 
                                                  NLK_NEG_TABLE_SIZE, 
                                                  NLK_NEG_TABLE_POW);
        if(verbose) {
            printf("Loaded NEG Layer\n");
        }
    } else {
        nn->hs = NULL;
    }


    /* read the other layers */
    nn->pos = 0;
    for(ii = 0; ii < nn->n_layers; ii++) {
        switch(nn->types[ii]) {
            /* Lookup Layer */
            case NLK_LAYER_LOOKUP_TYPE:
                nn->layers[ii].lk = nlk_layer_lookup_load(fp);
                if(nn->layers[ii].lk == NULL) {
                    goto nlk_neuralnet_load_err;
                }
                if(verbose) {
                    printf("Loaded Lookup Layer %zu: %zu x %zu\n", n_layers,
                            nn->layers[ii].lk->weights->rows,
                            nn->layers[ii].lk->weights->cols);
                }
                break;

            /* Linear Layer */
            case NLK_LAYER_LINEAR_TYPE:
                nn->layers[ii].ll = nlk_layer_linear_load(fp);
                if(nn->layers[ii].ll == NULL) {
                    goto nlk_neuralnet_load_err;
                }
                if(verbose) {
                    printf("Loaded Linear Layer %zu: %zu x %zu bias = %d\n", 
                            n_layers,
                            nn->layers[ii].ll->weights->rows,
                            nn->layers[ii].ll->weights->cols,
                            nn->layers[ii].ll->bias != NULL);
                }
                break;
            default:
                NLK_ERROR_NULL("invalid layer type", NLK_FAILURE);
                /* unreachable */
        } /* end of switch for layer type */

    } /* end of layers */

    /* set position */
    nn->pos =  nn->n_layers;


    /* set context options */
    nlk_lm_context_opts(nn->train_opts.model_type, nn->train_opts.window, 
                        &nn->vocab, &ctx_opts);
    nn->context_opts = ctx_opts;


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


NLK_LM
nlk_lm_model(const char *model_name, const bool concat)
{
    NLK_LM lm_type;

    if(strcasecmp(model_name, "cbow") == 0) { 
        lm_type = NLK_CBOW;
    } else if(strcasecmp(model_name, "sg") == 0) {
        lm_type = NLK_SKIPGRAM;
    } else if(strcasecmp(model_name, "pvdm") == 0) {
        if(concat) {
            lm_type = NLK_PVDM_CONCAT;
        } else {
            lm_type = NLK_PVDM;
        }
    } else if(strcasecmp(model_name, "pvdbow") == 0) {
        lm_type = NLK_PVDBOW;
    } else {
        NLK_ERROR_ABORT("Invalid model type.", NLK_EINVAL);
    }

    return lm_type;
}

/**
 * Default learning rates
 *
 * @note
 * skip/PVDBOW: learn_rate = 0.05; 
 * cbow/PVDM: learn_rate = 0.025; 
 * other: 0.01
 * @endnote
 */
nlk_real
nlk_lm_learn_rate(NLK_LM lm_type)
{
    switch(lm_type) {
        case NLK_CBOW_SUM:
            /* fall through */
        case NLK_PVDM:
            /* fall through */
        case NLK_PVDM_CONCAT:
            /* fall through */
        case NLK_PVDM_SUM:
            /* fall through */
        case NLK_CBOW: 
            return 0.025;
        case NLK_PVDBOW:
            /* fall through */
        case NLK_SKIPGRAM:
            return 0.05;
        case NLK_MODEL_NULL:
            return 0.01;
        default:
            return 0.01;
    }
    return 0.01;
}


void
nlk_lm_context_opts(NLK_LM model, unsigned int window, 
                    struct nlk_vocab_t **vocab, 
                    struct nlk_context_opts_t *opts)
{
    /* defaults */
    opts->before = window;
    opts->after = window;
    opts->b_equals_a = true;
    opts->prepad = false;
    opts->postpad = false;
    opts->paragraph = false;
    opts->prepad_paragraph = false;

    /* random_window in range before=[1, before], after=[1, after] */
    opts->random_windows = true;

    switch(model) {
        case NLK_PVDM_CONCAT:
            /* fixed size windows */
            opts->random_windows = false;
            /* prepad/postpad if smaller */
            opts->prepad = true;
            opts->postpad = true;
            /* FALL THROUGH: all other options are common to PVDM */
        case NLK_PVDM:
            /* predict the next word => after = 0 */
            opts->b_equals_a = false;
            opts->after = 0;
            opts->paragraph = true;
            break;
        case NLK_PVDBOW:
            opts->paragraph = true;
            opts->prepad_paragraph = true;
            break;
        case NLK_CBOW:
        case NLK_SKIPGRAM:
            opts->paragraph = false;
            /* options for CBOW/SKIPGRAM are already set */
            break;
        case NLK_MODEL_NULL:
        default:
            NLK_ERROR_VOID("Invalid model for context generation", NLK_EINVAL);
            /*unreachable */
    }
    opts->start = nlk_vocab_get_start_symbol(vocab)->index;

    opts->max_size = window * 2;
    if(opts->paragraph) {
        opts->max_size += 1;
    }

}
