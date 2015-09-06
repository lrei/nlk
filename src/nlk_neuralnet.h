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
#include "nlk_window.h"


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
    NLK_LM           model_type;        /**< the model type */
    bool             paragraph;         /**< paragraph model */
    unsigned int     window;            /**< the window size */
    float            sample;            /**< the sampling rate */
    nlk_real         learn_rate;        /**< the initial learn rate */
    bool             hs;                /**< use hierarchical softmax */
    unsigned int     negative;          /**< number of negative examples */
    unsigned int     iter;              /**< number of iterations/epochs */
    unsigned int     vector_size;       /**< word/par vector dimensionality */
    uint64_t         word_count;        /**< total word occurances in corpus */
    uint64_t         paragraph_count;   /**< total paragraphs in corpus */
};
typedef struct nlk_w2v_train_t NLK_W2V_TRAIN;

/** @enum NLK_LAYER_T
 * The type of layer
 */
enum nlk_layer_type_t { 
    NLK_LAYER_LINEAR_TYPE   = 0,  /**< a regular linear layer */
    NLK_LAYER_LOOKUP_TYPE   = 1   /**< a lookup table */
};
typedef enum nlk_layer_type NLK_LAYER_TYPE;

/** @union NLK_LAYER
 * A Layer
 */
union nlk_layer_t {
    struct nlk_layer_lookup_t *lk;  /**< lookup */
    struct nlk_layer_linear_t *ll;  /**< linear */
};
typedef union nlk_layer_t NLK_LAYER;

/** @struct nlk_neuralnet_t
 * A Neural Net structure
 */
struct nlk_neuralnet_t {
    struct nlk_context_opts_t    context_opts;  /**< context generation opts */
    struct nlk_nn_train_t        train_opts;    /**< model training options */
    struct nlk_vocab_t          *vocab;         /**< the vocabulary */
    /**< language specific layers deserve their own shortcuts */
    struct nlk_layer_lookup_t   *words;         /**< word lookup layer */
    struct nlk_layer_lookup_t   *paragraphs;    /**< paragraph lookup layer */
    /**< language model specific layers also get their own shortcuts */
    struct nlk_layer_lookup_t   *hs;            /**< hierarchical softmax */
    struct nlk_layer_lookup_t   *neg;           /**< negative sampling layer */
    size_t                      *neg_table;     /**< negative sampling table */
    /**< other layers go here */
    size_t                       n_layers;      /**< total number of layers */
    size_t                       pos;           /**< add positition */
    unsigned short int          *types;         /**< layer types */
    union nlk_layer_t           *layers;        /**< the other layers */
};
typedef struct nlk_neuralnet_t NLK_NEURALNET;


bool nlk_neuralnet_is_paragraph_model(const NLK_LM); 
bool nlk_neuralnet_is_concat_model(const NLK_LM);


/* create */
struct nlk_neuralnet_t *nlk_neuralnet_create(size_t);

/* expand */
int nlk_neuralnet_expand(struct nlk_neuralnet_t *, size_t);
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

/* misc */
NLK_LM   nlk_lm_model(const char *, const bool);
nlk_real nlk_lm_learn_rate(NLK_LM);
void     nlk_lm_context_opts(NLK_LM, unsigned int, struct nlk_vocab_t **,
                            struct nlk_context_opts_t *);

__END_DECLS
#endif /* __NLK_NEURALNET_H__ */
