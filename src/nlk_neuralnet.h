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


/** @file nlk_neuralnet.h
* Neural Net
*/

#ifndef __NLK_NEURALNET_H__
#define __NLK_NEURALNET_H__


#include "nlk_layer_linear.h"


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


/** @enum NLK_LM_TYPE
 * The type of the language model
 */
enum nlk_lm_t { 
    NLK_MODEL_NULL  = 0,
    NLK_CBOW        = 10,  /**< CBOW default: avg() */
    NLK_CBOW_SUM    = 11,  /**< CBOW sum() instead of avg() NOT IMPLEMENTED */
    NLK_SKIPGRAM    = 20,  /**< Skipgram */
    NLK_PVDM        = 30,  /**< PVDM default: avg() */
    NLK_PVDM_CONCAT = 31,  /**< PVDM concat() instead of avg() */
    NLK_PVDM_SUM    = 32,  /**< PVDM sum() instead of avg() NOT IMPLEMENTED */
    NLK_PVDBOW      = 40,  /**< PVDBOW */
};
typedef enum nlk_lm_t NLK_LM;

/** @struct nlk_nn_train_t
 * Training Parameters for algorithms
 */
struct nlk_nn_train_t {
    NLK_LM           model_type;    /**< the model type */
    unsigned int     window;        /**< the window size */
    float            sample;        /**< the sampling rate */
    nlk_real         learn_rate;    /**< the initial learn rate */
    bool             hs;            /**< use hierarchical softmax */
    unsigned int     negative;      /**< number of negative examples */
};
typedef struct nlk_w2v_train_t NLK_W2V_TRAIN;

/** @enum NLK_LAYER_T
 * The type of layer
 */
enum nlk_layer_type { 
    NLK_LAYER_LINEAR_TYPE   = 0,  /**< a regular linear layer */
    NLK_LAYER_LOOKUP_TYPE   = 1   /**< a lookup table */
};
typedef enum nlk_layer_type NLK_LAYER_TYPE;

/** @union NLK_LAYER
 * A Layer
 */
union nlk_layer_t {
    struct nlk_layer_lookup_t *lk;
    NLK_LAYER_LINEAR *ll;
};
typedef union nlk_layer_t NLK_LAYER;

/** @struct nlk_neuralnet_t
 * A Neural Net
 */
struct nlk_neuralnet_t {
    struct nlk_nn_train_t   train_opts; /**< model training options */
    size_t                  n_layers;  /**< the total number of layers */
    size_t                  pos;       /**< keeps track of position for add */
    unsigned short int     *types;     /**< layer type for each layer */
    union nlk_layer_t      *layers;    /**< the array of layers */
};
typedef struct nlk_neuralnet_t NLK_NEURALNET;


/* create */
struct nlk_neuralnet_t *nlk_neuralnet_create(size_t);
/* add layers */
void nlk_neuralnet_add_layer_lookup(struct nlk_neuralnet_t *,
                                    struct nlk_layer_lookup_t *);
void nlk_neuralnet_add_layer_linear(struct nlk_neuralnet_t *,
                                    struct nlk_layer_linear_t *);
/* free */
void nlk_neuralnet_free(struct nlk_neuralnet_t *);

/* save */
int nlk_neuralnet_save(struct nlk_neuralnet_t *, FILE *);
int nlk_neuralnet_save_path(struct nlk_neuralnet_t *, char *);

/* load */
struct nlk_neuralnet_t *nlk_neuralnet_load(FILE *, bool);
struct nlk_neuralnet_t *nlk_neuralnet_load_path(char *, bool);

__END_DECLS
#endif /* __NLK_NEURALNET_H__ */
