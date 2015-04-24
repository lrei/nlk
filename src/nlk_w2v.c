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


/** @file nlk_w2v.c
 * Word2Vec Implementation: CBOW & Skipgram models
 */

#include <time.h>
#include <errno.h>
#include <math.h>
#include <float.h>

#include <omp.h>
#include <cblas.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_random.h"
#include "nlk_vocabulary.h"
#include "nlk_window.h"
#include "nlk_neuralnet.h"
#include "nlk_layer_linear.h"
#include "nlk_tic.h"
#include "nlk_text.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_learn_rate.h"

#include "nlk_w2v.h"



/**
 * Word2Vec style progress display.
 *
 * @param learn_rate        the current learn rate
 * @param word_count_actual total number of words seen so far
 * @param train_words       total number of words in train file
 * @param epochs            total number of epochs
 * @param epoch             the current epoch
 * @param start             the clock at the start of the training
 */
static void
nlk_w2v_display(nlk_real learn_rate, size_t word_count_actual, 
                size_t train_words, int epochs, int epoch, 
                clock_t start)
{
    double progress;
    double speed;
    char display_str[256];

    clock_t now = clock();

    /* calculate */
    progress = (word_count_actual / (double)(epochs * train_words + 1)) * 100;
    speed = word_count_actual / ((double)(now - start + 1) / 
            (double)CLOCKS_PER_SEC * 1000);

    /* create string */
    snprintf(display_str, 256,
            "Alpha: %f  Progress: %.2f%% (%03d/%03d) "
            "Words/Thread/sec: %.2fk Threads: %d/%d", 
            learn_rate, progress, epoch + 1, epochs, speed, 
            omp_get_num_threads(), omp_get_num_procs());

    /* display */
    nlk_tic(display_str, false);
}

//@TODO incomplete comment (args):
/**
 * Create a word2vec neural network
 *
 * @param vocab_size    the vocabulary size
 * @param paragraphs    number of paragraphs (0 if not learning PVs)
 * @param layer_size    the hidden layer size
 * @param hs            use hierarchical softmax
 * @param neg           use negative sampling
 *
 * @return the neural network structure
 */
struct nlk_neuralnet_t *
nlk_w2v_create(NLK_LM model_type, bool concat, unsigned int window, 
               float sample, nlk_real learn_rate, bool hs, unsigned int neg, 
               size_t layer_size, size_t vocab_size, size_t paragraphs, 
               bool verbose) 
{
    struct nlk_neuralnet_t *nn;
    struct nlk_layer_lookup_t *lk1;
    struct nlk_layer_lookup_t *lkhs;
    struct nlk_layer_lookup_t *lkneg;
    int n_layers = 1;
    size_t layer_size2;
    if(concat) {
        layer_size2 = window * 2 * layer_size + layer_size;
    } else {
        layer_size2 = layer_size;
    }

    if(hs) {
        n_layers++;
    }
    if(neg) {
        n_layers++;
    }

    nn = nlk_neuralnet_create(n_layers);
    if(nn == NULL) {
        return NULL;
    }
    if(verbose) {
        printf("Neural Network layers: %d\n", n_layers);
    }

    /* training options */
    nn->train_opts.model_type = model_type;
    nn->train_opts.window = window;
    nn->train_opts.sample = sample;
    nn->train_opts.learn_rate = learn_rate;
    nn->train_opts.hs = hs;
    nn->train_opts.negative = neg;


    /* lookup layer 1 */
    lk1 = nlk_layer_lookup_create(vocab_size + paragraphs, layer_size);
    if(verbose) {
        printf("Layer 1: %zu x %zu\n", lk1->weights->rows, lk1->weights->cols);
    }

    /* random number generator initialization */
    uint64_t seed = 6121984 * clock();
    seed = nlk_random_fmix(seed);
    nlk_random_init_xs1024(seed);

    nlk_layer_lookup_init(lk1);
    nlk_neuralnet_add_layer_lookup(nn, lk1);
    
    /* lookup layer HS */
    if(hs) {
        lkhs = nlk_layer_lookup_create(vocab_size, layer_size2);
        /* [ default initialization: zero ] */
        nlk_neuralnet_add_layer_lookup(nn, lkhs);
        if(verbose) {
            printf("Layer 2 (HS): %zu x %zu\n", 
                    lkhs->weights->rows, lkhs->weights->cols);
        }
    }

    /* lookup layer NEG */
    if(neg) {
        lkneg = nlk_layer_lookup_create(vocab_size, layer_size2);
        /* [ default initialization: zero ] */
        nlk_neuralnet_add_layer_lookup(nn, lkneg);
        if(verbose) {
            printf("Layer %d (NEG): %zu x %zu\n", n_layers,
                   lkneg->weights->rows, lkneg->weights->cols);
        }

    }
 
    return nn;
}

/**
 * Hierarchical Softmax
 * @param lk2hs         the hierarchical softmax layer
 * @param update        wether to update the layer weights or not
 * @param lk1_out       the output of the previous layer, input to this one
 * @param learn_rate    the learning rate
 * @param sigmoid_table the sigmoid table
 * @param center_word   the center/target word
 * @param grad_acc      the accumulated gradient (output)
 *
 */
void
nlk_w2v_hs(NLK_LAYER_LOOKUP *lk2hs, const bool update,
           const NLK_ARRAY *lk1_out, const nlk_real learn_rate, 
           const nlk_real *sigmoid_table, 
           const struct nlk_vocab_t *center_word, NLK_ARRAY *grad_acc)
{
    nlk_real out;
    nlk_real lk2_out;
    nlk_real grad_out;
    size_t point;
    uint8_t code;

    /** @section Hierarchical Softmax Forward
     * the center word (target) gets forwarded through the second lookup 
     * layer. The second lookup layer "maps" points to codes 
     * (through the output softmax)
     */

    /* for each point of center word */
    for(size_t pp = 0; pp < center_word->code_length; pp++) {
        point = center_word->point[pp];
        code = center_word->code[pp];

        /* forward with lookup for point pp */
        nlk_layer_lookup_forward(lk2hs, lk1_out, point, &lk2_out);
        
        /* ignore points with outputs outside of sigm bounds */
        if(lk2_out >= NLK_MAX_EXP) {
            continue;
        } else if(lk2_out <= -NLK_MAX_EXP) {
            continue;
        }
        out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);


        /** @section Hierarchical Softmax Backprop
         * Using the negative log likelihood,
         *
         * log(sigma(z=v'n(w,j))'vwi) = 
         * = (1 - code) * z - log(1 + e^z)
         * d/dx = 1 - code  - sigmoid(z)
         */
        /* error */
        grad_out = code - out;

        /* gradient */
        grad_out = 1.0 - grad_out;

        /* multiply by learning rate */
        grad_out *= learn_rate;
        
        /* layer2hs backprop, accumulate gradient for all points */
        nlk_layer_lookup_backprop_acc(lk2hs, update, lk1_out, point, grad_out, 
                                      grad_acc);

    } /* end of points/codes */
}

/**
 * Negative Sampling
 * 
 * @param lk2neg        the negative sampling layer
 * @param update        wether to update the layer weights or not
 * @param neg_table     the negative sampling table
 * @param negative      the number of negative examples to use
 * @param vocab_size    the size of the vocabulary
 * @param learn_rate    the learning rate
 * @param center_word   the center/target word (i.e. positive example)
 * @param lk1_out       the output of the previous layer, input to this one
 * @param sigmoid_table the sigmoid table
 * @param grad_acc      the accumulated gradient (output)
 *
 */
void
nlk_w2v_neg(NLK_LAYER_LOOKUP *lk2neg, const bool update, 
            const size_t *neg_table, const size_t negative, 
            const size_t vocab_size, const nlk_real learn_rate, 
            const size_t center_word, const NLK_ARRAY *lk1_out, 
            const nlk_real *sigmoid_table, NLK_ARRAY *grad_acc)
{
    nlk_real out;
    size_t target;
    uint64_t random;
    nlk_real grad_out;
    nlk_real lk2_out;


    /** @section Positive Example
     */
    /* forward with lookup for target word */
    nlk_layer_lookup_forward(lk2neg, lk1_out, center_word, &lk2_out);

    /* shortcuts when outside of sigm bounds */
    if(lk2_out >= NLK_MAX_EXP) {
        /* do nothing, no error */
    } else if(lk2_out <= -NLK_MAX_EXP) {
        grad_out = learn_rate;

       /* layer2neg backprop, accumulate gradient for all examples */
        nlk_layer_lookup_backprop_acc(lk2neg, update, lk1_out, center_word, 
                                      grad_out, grad_acc);
    } else {
        /* inside sigmoid bounds */
        out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);

        /** @section Skipgram NEG Sampling Backprop
         * Same gradient formula as in HS but this time label is 1
         * then multiply by learning rate
         */
        grad_out = -out * learn_rate;

       /* layer2neg backprop, accumulate gradient for pos example */
        nlk_layer_lookup_backprop_acc(lk2neg, update, lk1_out, center_word, 
                                      grad_out, grad_acc);
    }


    /** @section Negative Examples *
     */
    for(size_t nn = 0; nn < negative; nn++) {
        random = nlk_random_xs1024(); 
        if(random != 0) {
            target = neg_table[random % NLK_NEG_TABLE_SIZE];
        } else {
            target = nlk_random_xs1024() % (vocab_size - 1) + 1; 
        }
        if(target == center_word) {
            /* ignore if this is the actual word */
            continue;
        }

        /* forward with lookup for target word */
        nlk_layer_lookup_forward(lk2neg, lk1_out, target, &lk2_out);

        /* shortcuts when outside of sigm bounds */
        if(lk2_out >= NLK_MAX_EXP) {
            grad_out = -learn_rate;
        } else if(lk2_out <= -NLK_MAX_EXP) {
            /* no error, do nothing */
            continue;
        } else {
            /* inside sigmoid bounds */
            out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);

            /** @section Skipgram NEG Sampling Backprop
             * Same gradient formula but this time label is 0
             * multiply by learning rate 
             */
            grad_out = (1.0 - out) * learn_rate;
        }

        /* layer2neg backprop, accumulate gradient for all examples */
        nlk_layer_lookup_backprop_acc(lk2neg, update, lk1_out, target, 
                                      grad_out, grad_acc);
    } /* end of negative examples */

}

/**
 * Train CBOW for a series of word contexts
 *
 */
static void
nlk_cbow(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
         const bool hs, NLK_LAYER_LOOKUP *lk2neg, 
         const size_t negative, const size_t *neg_table, 
         const size_t vocab_size, const nlk_real learn_rate, 
         const nlk_real *sigmoid_table, 
         const struct nlk_context_t *context,
         NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
#ifndef NCHECKS
    if(context->size == 0) {
        NLK_ERROR_ABORT("Context size must be > 0", NLK_EBADLEN);
    }
#endif

    nlk_array_zero(grad_acc);


    /** @section CBOW Forward through the first layer
     * The context words get forwarded through the first lookup layer
     * and their vectors are averaged.
     */
    nlk_layer_lookup_forward_lookup_avg(lk1, context->window, 
                                        context->size, lk1_out);

#ifdef CHECK_NANS
        if(nlk_array_has_nan(lk1_out)) {
            NLK_ERROR_VOID("lk1_out has NaNs", NLK_FAILURE);
        }
#endif


    /** @section CBOW Hierarchical Softmax 
     */
    if(hs) {
        nlk_w2v_hs(lk2hs, true, lk1_out, learn_rate, sigmoid_table, 
                   context->center, grad_acc);
    } 

    /** @section CBOW NEG Sampling Forward
     */
    if(negative) {
        nlk_w2v_neg(lk2neg, true, neg_table, negative, vocab_size, learn_rate, 
                    context->center->index, lk1_out, sigmoid_table, grad_acc);
    }

#ifdef CHECK_NANS
        if(nlk_array_has_nan(grad_acc)) {
            NLK_ERROR_VOID("grad_acc has NaNs", NLK_FAILURE);
        }
#endif


    /** @section Backprop into the first lookup layer
     * Learn layer1 weights using the accumulated gradient
     */
    nlk_layer_lookup_backprop_lookup(lk1, context->window, context->size, 
                                     grad_acc);
}

/**
 * Train CBOW_CONCAT (PVDM_CONCAT) for a series of word contexts
 *
 */
static void
nlk_cbow_concat(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
             const bool hs, NLK_LAYER_LOOKUP *lk2neg, 
             const size_t negative, const size_t *neg_table, 
             const size_t vocab_size, const nlk_real learn_rate, 
             const nlk_real *sigmoid_table, 
             const struct nlk_context_t *context,
             NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
#ifndef NCHECKS
    if(context->size == 0) {
        NLK_ERROR_ABORT("Context size must be > 0", NLK_EBADLEN);
    }
#endif

    nlk_array_zero(grad_acc);


    /** @section CBOW Forward through the first layer
     * The context words get forwarded through the first lookup layer
     * and their vectors are averaged.
     */
    nlk_layer_lookup_forward_lookup_concat(lk1, context->window, 
                                           context->size, lk1_out);

    /** @section CBOW Hierarchical Softmax 
     */
    if(hs) {
        nlk_w2v_hs(lk2hs, true, lk1_out, learn_rate, sigmoid_table,
                   context->center, grad_acc);
    } 

    /** @section CBOW NEG Sampling Forward
     */
    if(negative) {
        nlk_w2v_neg(lk2neg, true, neg_table, negative, vocab_size, learn_rate, 
                    context->center->index, lk1_out, sigmoid_table, grad_acc);
    }

#ifdef CHECK_NANS
        if(nlk_array_has_nan(grad_acc)) {
            NLK_ERROR_VOID("grad_acc has NaNs", NLK_FAILURE);
        }
#endif

    /** @section Backprop into the first lookup layer
     * Learn layer1 weights using the accumulated gradient
     */
    nlk_layer_lookup_backprop_lookup_concat(lk1, context->window, 
                                            context->size, grad_acc);
}

/**
 * Train skipgram for a word context
 *
 */
void
nlk_skipgram(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
             const bool hs, NLK_LAYER_LOOKUP *lk2neg, const size_t negative,
             const size_t *neg_table, const size_t vocab_size, 
             const nlk_real learn_rate, const nlk_real *sigmoid_table, 
             const struct nlk_context_t *context,
             NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    /* for each context word jj */
    for(size_t jj = 0; jj < context->size; jj++) {
        nlk_array_zero(grad_acc);


        /** @section Skipgram Lookup Layer1 Forward (common)
         * Each context word gets forwarded through the first lookup layer
         */
        nlk_layer_lookup_forward_lookup_one(lk1, context->window[jj], lk1_out);
        /* or equivalently w/o a copy:
         * (but w2v_train needs to save lk1->data pointer)
         * lk1_out->data = &lk1->weights->data[context->window[jj] 
         *                                      lk1->weights->cols];
         */

        /** @section Skipgram Hierarchical Softmax */
        if(hs) {
            nlk_w2v_hs(lk2hs, true, lk1_out, learn_rate, sigmoid_table, 
                       context->center, grad_acc);
        }
            
        /** @section Skipgram NEG Sampling
         */
        if(negative) {
            nlk_w2v_neg(lk2neg, true, neg_table, negative, vocab_size, 
                        learn_rate, context->center->index, lk1_out, 
                        sigmoid_table, grad_acc);
        }   /* end of NEG sampling specific code */

#ifdef CHECK_NANS
        if(nlk_array_has_nan(grad_acc)) {
            NLK_ERROR_VOID("grad_acc has NaNs", NLK_FAILURE);
        }
#endif

        /** @section Backprop into the first lookup layer
         * Learn layer1 weights using the accumulated gradient
         */
        nlk_layer_lookup_backprop_lookup_one(lk1, context->window[jj], 
                                             grad_acc);

    } /* end of context words */
}

/**
 * Train a word2vec model
 *
 * @param nn                the neural network
 * @param train_file_path   the path of the train file
 * @param vocab             the vocabulary
 * @param total_lines       total lines in train file (used for PVs)
 * @param epochs
 * @param verbose
 *
 * @return total error for last epoch
 */
void
nlk_w2v_train(struct nlk_neuralnet_t *nn, const char *train_file_path, const bool numbered,
              struct nlk_vocab_t **vocab, const size_t total_lines, 
              unsigned int epochs, int verbose)
{
    /*goto_set_num_threads(1);*/

    /* unpack training options */
    const NLK_LM model_type = nn->train_opts.model_type;
    const bool hs = nn->train_opts.hs;
    const unsigned int negative = nn->train_opts.negative;
    const size_t window = nn->train_opts.window;
    const float sample_rate = nn->train_opts.sample;
    nlk_real learn_rate = nn->train_opts.learn_rate;

    /* shortcuts */
    struct nlk_layer_lookup_t *lk1 = nn->layers[0].lk;
    size_t layer_n = 1;
    struct nlk_layer_lookup_t *lkhs = NULL;
    size_t layer_size2 = lk1->weights->cols;
    if(hs) {
        lkhs = nn->layers[layer_n].lk;
        layer_n++;
        layer_size2 = lkhs->weights->cols;
    }
    struct nlk_layer_lookup_t *lkneg = NULL;
    if(negative) {
        lkneg = nn->layers[layer_n].lk;
        layer_size2 = lkneg->weights->cols;
    }


    const size_t vocab_size = nlk_vocab_size(vocab);
    const size_t train_words = nlk_vocab_total(vocab);

    /** @section Shared Initializations
     * All variables declared in this section are shared among threads.
     * Most are read-only but some (e.g. learn_rate & word_count_actual)
     * are updated directly from the threads.
     */

    /** @subsection Input and Context/Window initializations
     * Alllocations and initializations related to the input (text)
     */
    size_t word_count_actual = 0;   /* warning: thread write-shared */
    size_t max_word_size = NLK_LM_MAX_WORD_SIZE;
    size_t max_line_size = NLK_LM_MAX_LINE_SIZE;


    /* Context for PVDBOW (skipgram with paragraph vectors) includes the
     * window center word in the "context". PVDM ds not.
     * Neither of the non paragraph models includes the center word in the 
     * context obviously.
     * Context for PVDM includes the paragraph vector.
     */

    size_t ctx_size = window * 2;   /* max context size */
    size_t context_multiplier = 1;  /* != 1 only for PVDBOW */

    bool learn_par = false;
    /* PVDBOW */
    if(model_type == NLK_PVDBOW) {
        context_multiplier = ctx_size;
        ctx_size = 1;   /* one (word, paragraph) pair at a time */
        learn_par = true;
    /* PVDM */
    } else if(model_type == NLK_PVDM || model_type == NLK_PVDM_CONCAT) { 
        ctx_size += 1; /* space for the paragraph (1st elem of window) */
        learn_par = true;
    }

    struct nlk_context_opts_t ctx_opts;
    nlk_context_model_opts(model_type, window, vocab, &ctx_opts);

    /* determine the train file size */
    FILE *in = fopen(train_file_path, "rb");
    if (in == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }
    fseek(in, 0, SEEK_END);
    size_t train_file_size = ftell(in);
    fclose(in);


    /** @subsection Neural Net initializations
     * Create and initialize neural net and associated variables 
     */
    nlk_real learn_rate_start = learn_rate;

   /* random number generator initialization */
    nlk_random_init_xs1024(nlk_random_seed());

    /* sigmoid table */
    nlk_real *sigmoid_table = nlk_table_sigmoid_create();

    /* neg table for negative sampling */
    size_t *neg_table = NULL;
    if(negative) {
        neg_table = nlk_vocab_neg_table_create(vocab, NLK_NEG_TABLE_SIZE, 
                                               0.75);
    }

    /* time keeping */
    clock_t start = clock();
    nlk_tic_reset();
    nlk_tic(NULL, false);


    /** @section Thread Private initializations 
     * Variables declared in this section are thread private and thus 
     * have thread specific values
     */
#pragma omp parallel
{
    size_t zz;

    int num_threads = omp_get_num_threads();

    /** @subsection
     * word count / progress
     */
    size_t n_examples;
    size_t line_len;
    size_t n_subsampled;
    size_t word_count = 0;
    size_t last_word_count = 0;

    /** @subsection Neural Network thread private initializations
     */
    int local_epoch = 0;          /* current epoch (thread local) */

    /* output of the first layer */
    NLK_ARRAY *lk1_out = nlk_array_create(layer_size2, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size2);
    
    /** @subsection Input Text and Context Window initializations
     */
    
    /* allocate memory for reading from the input file */
    char **text_line = (char **) calloc(max_line_size, sizeof(char *));
    if(text_line == NULL) {
        NLK_ERROR_ABORT("unable to allocate memory for text", NLK_ENOMEM);
        /* unreachable */
    }
    for(zz = 0; zz < max_line_size; zz++) {
        text_line[zz] = calloc(max_word_size, sizeof(char));
        if(text_line[zz] == NULL) {
            NLK_ERROR_ABORT("unable to allocate memory for text", NLK_ENOMEM);
            /* unreachable */
        }
    }

    /* for converting to a vocabularized representation of text */
    struct nlk_vocab_t *vectorized[max_line_size];

    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = (struct nlk_context_t **) 
        malloc(max_line_size * context_multiplier * 
               sizeof(struct nlk_context_t *));
    if(contexts == NULL) {
        NLK_ERROR_ABORT("unable to allocate memory for contexts", NLK_ENOMEM);
        /* unreachable */
    }

    for(zz = 0; zz < max_line_size * context_multiplier; zz++) {
        contexts[zz] = nlk_context_create(ctx_size);
        if(contexts[zz] == NULL) {
            NLK_ERROR_ABORT("unable to allocate memory for contexts", 
                            NLK_ENOMEM);
        }
    }

    /* 
     * The train file is divided into parts, one part for each thread.
     * Open file and move to thread specific starting point
     */
    size_t file_pos = 0;
    size_t end_pos = 0;
    size_t line_start = 0;
    size_t line_cur = 0;
    size_t par_id = 0;
    size_t ex = 0;
    FILE *train = fopen(train_file_path, "rb");

#pragma omp for 
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        /* set train file part position */
        end_pos = nlk_set_file_pos(train, learn_par, train_file_size,
                                   num_threads, thread_id);
        file_pos = ftell(train);

        /* determine line number */
        if(learn_par) {
            line_start = nlk_text_get_line(train);
            line_cur = line_start;
        }

        /** @section Start of Training Loop (Epoch Loop)
         */
        while(local_epoch < epochs) {
             /* update learning rate */
            if (word_count - last_word_count > 10000) {
                word_count_actual += word_count - last_word_count;
                last_word_count = word_count;

                /* display progress */
                if(verbose > 0) {
                    nlk_w2v_display(learn_rate, word_count_actual, 
                                    train_words, epochs, local_epoch, start);
                }
                /* update learning rate */
                learn_rate = nlk_learn_rate_w2v_update(learn_rate,
                                                       learn_rate_start,
                                                       epochs,
                                                       word_count_actual, 
                                                       train_words);
            }

            /** @subsection Read from File and Create Context Windows
             * The actual difference between the word models and the paragraph
             * models is in the context that gets generated here.
             */
            /* read line */
            if(numbered) {
                nlk_read_number_line(train, text_line, &par_id, max_word_size,
                                     max_line_size);
#ifdef DEBUGRINT
                printf("real par_id: %zu\n", par_id);
#endif
                par_id = par_id + vocab_size;
            } else {
                nlk_read_line(train, text_line, max_word_size, max_line_size);
                par_id = line_cur + vocab_size;
            }
#ifdef DEBUGPRINT
            nlk_text_print_numbered_line(text_line, line_cur, thread_id);
#endif
            
            /* vocabularize */
            line_len = nlk_vocab_vocabularize(vocab, train_words, text_line, 
                                              sample_rate, 
                                              NULL, true, vectorized, 
                                              &n_subsampled); 
            
#ifdef DEBUGRINT
            for(size_t vv = 0; vv < line_len; vv++) {
                if(vectorized[vv]->index >= vocab_size) { /* 0 based index */
                    nlk_vocab_print_line(vectorized, line_len, true);
                    NLK_ERROR_ABORT("vectorized is wrong", NLK_EDOM);
                }
            }
#endif

            /* line_len = 1 would be an empty line, nothing to learn */
            if(line_len < 2) {
                continue;
            }

            /* Context Window
             * paragraph id (index in weight matrix) = line number + vocab_size
             */
            n_examples = nlk_context_window(vectorized, line_len, par_id, 
                                            &ctx_opts, contexts);

#ifdef DEBUGRINT
            printf("par_id (ctx): %zu\n", par_id);
            for(size_t vv = 0; vv < n_examples; vv++) {
                nlk_context_print(contexts[vv]);
            }
#endif

            /** @subsection Algorithm Parallel Loop Over Contexts
             */
            /* Skipgram & PVDBOW */
            if(model_type == NLK_SKIPGRAM || model_type == NLK_PVDBOW) {
                for(ex = 0; ex < n_examples; ex++) {
                    nlk_skipgram(lk1, lkhs, hs, lkneg, negative, neg_table, 
                                 vocab_size, learn_rate, sigmoid_table, 
                                 contexts[ex], grad_acc, lk1_out);
                }
            /* CBOW & PVDM */
            } else if(model_type == NLK_CBOW || model_type == NLK_PVDM) {
                for(ex = 0; ex < n_examples; ex++) {
                    nlk_cbow(lk1, lkhs, hs, lkneg, negative, neg_table, 
                             vocab_size, learn_rate, sigmoid_table, 
                             contexts[ex], grad_acc, lk1_out);
                }
            } else if(model_type == NLK_PVDM_CONCAT) {
                for(ex = 0; ex < n_examples; ex++) {
                    nlk_cbow_concat(lk1, lkhs, hs, lkneg, negative, neg_table, 
                                    vocab_size, learn_rate, sigmoid_table, 
                                    contexts[ex], grad_acc, lk1_out);
                }
            } else {
                NLK_ERROR_ABORT("invalid model type", NLK_EINVAL);
                /* unreachable */
            }

            /* update count/location */
            word_count += line_len + n_subsampled;
            line_cur++;
            file_pos = ftell(train);

            /** @subsection Epoch End Check
             * Increment thread private epoch counter if necessary
             */
            /* check for end of file part, decrement local epochs, rewind */
            if(file_pos >= end_pos) {
                /* update count and epoch */
                word_count_actual += word_count - last_word_count;
                local_epoch++;

                /* rewind */
                word_count = 0;
                last_word_count = 0;
                line_cur = line_start;
                nlk_set_file_pos(train, learn_par, train_file_size, 
                                 num_threads, thread_id);
            }
        } /* end of epoch cycle */
    } /* end of threaded algorithm execution */

    /** @subsection Free Thread Private Memory and Close Files
     */
    for(zz = 0; zz < max_line_size * context_multiplier; zz++) {
        nlk_context_free(contexts[zz]);
    }
    for(zz = 0; zz < max_line_size; zz++) {
        free(text_line[zz]);
    }
    free(text_line);
    nlk_array_free(lk1_out);
    nlk_array_free(grad_acc);
    fclose(train);
} /* *** End of Paralell Region *** */

    /** @section End
     */
    nlk_tic_reset();
    free(sigmoid_table);
    if(negative) {
        free(neg_table);
    }
}

/**
 * Export word-vector pairs in word2vec text compatible format
 */
void
nlk_w2v_export_vectors(NLK_ARRAY *weights, NLK_FILE_FORMAT format, 
                       struct nlk_vocab_t **vocab,
                       const size_t start, const size_t end, 
                       char *pv_prepend, char *filepath)
{
    size_t row_index = 0;
    size_t cc = 0;
    size_t real_end = 0;
    size_t pv_start = 0;
    size_t vocab_size = 0;

    if(format != NLK_FILE_W2V_BIN && format != NLK_FILE_W2V_TXT) {
        NLK_ERROR_VOID("unsuported file format", NLK_EINVAL);
        /* unreachable */
    }

    /** @section Prepare
     */
    vocab_size = nlk_vocab_size(vocab);

    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }

    /* handle edge cases */
    if(end > weights->rows) {
        real_end = weights->rows;
    } else {
        real_end = end;
    }
    if(real_end == start) {
        NLK_ERROR_VOID("end == start", NLK_ERANGE);
        /* unreachable */
    } else if(real_end < start) {
        NLK_ERROR_VOID("end < start", NLK_ERANGE);
        /* unreachable */
    }

    /* handle start of pvs */
    if(start > vocab_size) {
        pv_start = start;
    } else {
        pv_start = vocab_size;  /* index of first paragraph */
    }
    
    /** @section Word Weights 
     */
    for(row_index = start; row_index < vocab_size; row_index++) {
        /* output word string */
        fprintf(out, "%s ", nlk_vocab_at_index(vocab, row_index)->word);

        /* output vector */
        if(format == NLK_FILE_W2V_TXT) {
            for(cc = 0; cc < weights->cols; cc++) {
                fprintf(out, "%lf ", 
                        weights->data[row_index * weights->cols + cc]);
            }
        } else if(format == NLK_FILE_W2V_BIN) {
            for(cc = 0; cc < weights->cols; cc++) {
                fwrite(&weights->data[row_index * weights->cols + cc], 
                       sizeof(nlk_real), 1, out);
                }
        }
        fprintf(out, "\n");
    }

    for(row_index = pv_start; row_index < real_end; row_index++) {
        fprintf(out, "%s%zu ", pv_prepend, row_index - vocab_size);
        if(format == NLK_FILE_W2V_TXT) {
            for(cc = 0; cc < weights->cols; cc++) {
                fprintf(out, "%lf ", 
                        weights->data[row_index * weights->cols + cc]);
            }
        } else if(format == NLK_FILE_W2V_BIN) {
            fwrite(&weights->data[row_index * weights->cols + cc], 
                   sizeof(nlk_real), 1, out);
        }
        fprintf(out, "\n");
    }

    fclose(out);
}
