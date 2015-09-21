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
 * Word2Vec Implementation: CBOW, Skipgram, PVDBOW, PVDM models
 */

#include <time.h>
#include <errno.h>
#include <math.h>
#include <float.h>

#include <omp.h>

#include "nlk.h"
#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_random.h"
#include "nlk_vocabulary.h"
#include "nlk_window.h"
#include "nlk_neuralnet.h"
#include "nlk_layer_lookup.h"
#include "nlk_tic.h"
#include "nlk_text.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_learn_rate.h"
#include "nlk_util.h"
#include "nlk.h"

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
            "Words/Thread/sec: %.2fK Threads: %d/%d", 
            learn_rate, progress, epoch + 1, epochs, speed, 
            omp_get_num_threads(), omp_get_num_procs());

    /* display */
    nlk_tic(display_str, false);
}


/**
 * Create a word2vec neural network
 *
 * @param paragraphs    number of paragraphs (0 if not learning PVs)
 *
 * @return the neural network structure
 */
struct nlk_neuralnet_t *
nlk_w2v_create(struct nlk_nn_train_t train_opts, const bool concat,
               struct nlk_vocab_t *vocab, const bool verbose) 
{
    struct nlk_neuralnet_t *nn;
    struct nlk_context_opts_t ctx_opts;
    const size_t vocab_size = nlk_vocab_size(&vocab);
    const size_t paragraph_size = train_opts.paragraph_count;
    const size_t vector_size = train_opts.vector_size;
    size_t layer2_size = vector_size;

    /* handle concatenation models */
    if(concat) {
        /* window * words + pv */
        layer2_size = train_opts.window * vector_size + vector_size;
    } 

    /* create structure */
    nn = nlk_neuralnet_create(0);
    if(nn == NULL) {
        return NULL;
    }

    /* set the training options */
    nn->train_opts = train_opts;

    /* vocabulary */
    nn->vocab = vocab;

    /* context options */
    nlk_lm_context_opts(nn->train_opts.model_type, nn->train_opts.window, 
                        &nn->vocab, &ctx_opts);
    nn->context_opts = ctx_opts;



    /* Word Table */
    nn->words = nlk_layer_lookup_create(vocab_size, vector_size);
    if(verbose) {
        printf("Layer 1 (word lookup): %zu x %zu\n", 
                nn->words->weights->rows, nn->words->weights->cols);
    }
    /* initialize */
    nlk_layer_lookup_init(nn->words);


    /* Paragraph Table */
    if(nn->train_opts.paragraph) {
        nn->paragraphs = nlk_layer_lookup_create(paragraph_size, vector_size);
        if(verbose) {
            printf("Layer 1 (paragraph lookup): %zu x %zu\n", 
                   nn->paragraphs->weights->rows, 
                   nn->paragraphs->weights->cols);
        }
        /* initialize */
        nlk_layer_lookup_init(nn->paragraphs);
    } else {
        nn->paragraphs = NULL;
    }

    /* Hierarchical Softmax Layer */
    if(train_opts.hs) {
        nn->hs = nlk_layer_lookup_create(vocab_size, layer2_size);
        /* [ default initialization: zero ] */
        if(verbose) {
            printf("Layer 2 (HS): %zu x %zu\n", 
                    nn->hs->weights->rows, nn->hs->weights->cols);
        }
    }
    /* zero initialization by default */

    /* Negative Sampling Layer */
    if(nn->train_opts.negative) {
        nn->neg = nlk_layer_lookup_create(vocab_size, layer2_size);
        if(verbose) {
            printf("Layer 2 (NEG): %zu x %zu\n",
                   nn->neg->weights->rows, nn->neg->weights->cols);
        }
    }
    /* zero initialization by default */

    nn->neg_table = NULL;

    return nn;
}


/**
 * Hierarchical Softmax
 * @param nn            the neural network structure
 * @param lk1_out       the output of the previous layer, input to this one
 * @param learn_rate    the learning rate
 * @param center_word   the center/target word
 * @param grad_acc      the accumulated gradient (updated)
 */
static void
nlk_w2v_hs(struct nlk_neuralnet_t *nn, const NLK_ARRAY *lk1_out, 
           const nlk_real learn_rate, const struct nlk_vocab_t *center_word, 
           NLK_ARRAY *grad_acc)
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
        nlk_layer_lookup_forward(nn->hs, lk1_out, point, &lk2_out);
        
        /* ignore points with outputs outside of sigm bounds */
        if(lk2_out >= NLK_MAX_EXP) {
            continue;
        } else if(lk2_out <= -NLK_MAX_EXP) {
            continue;
        }
        out = nlk_sigmoid(lk2_out);


        /** Backprop
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
        nlk_layer_lookup_backprop_acc(nn->hs, lk1_out, point, grad_out, 
                                      grad_acc);

    } /* end of points/codes */
}


/**
 * Negative Sampling
 * 
 * @param nn            the neural network structure
 * @param learn_rate    the learning rate
 * @param center_word   the center/target word (i.e. positive example)
 * @param lk1_out       the output of the previous layer, input to this one
 * @param grad_acc      the accumulated gradient (output)
 *
 */
static void
nlk_w2v_neg(struct nlk_neuralnet_t *nn, const nlk_real learn_rate, 
            const size_t center_word, const NLK_ARRAY *lk1_out, 
            NLK_ARRAY *grad_acc)
{
    nlk_real out;
    size_t target;
    uint64_t random;
    nlk_real grad_out;
    nlk_real lk2_out;

    /** @section Positive Example
     */
    /* forward with lookup for target word */
    nlk_layer_lookup_forward(nn->neg, lk1_out, center_word, &lk2_out);

    /* shortcuts when outside of sigm bounds */
    if(lk2_out >= NLK_MAX_EXP) {
        /* do nothing, no error */
    } else if(lk2_out <= -NLK_MAX_EXP) {
        grad_out = learn_rate;

       /* layer2neg backprop, accumulate gradient for all examples */
        nlk_layer_lookup_backprop_acc(nn->neg, lk1_out, center_word, grad_out, 
                                      grad_acc);
    } else {
        /* inside sigmoid bounds */
        out = nlk_sigmoid(lk2_out);

        /** NEG Sampling Backprop
         * Same gradient formula as in HS but here label (truth) is 1
         */
        grad_out = -out * learn_rate;

       /* Backprop and accumulate gradient for pos example */
        nlk_layer_lookup_backprop_acc(nn->neg, lk1_out, center_word, 
                                      grad_out, grad_acc);
    }


    /** @section Negative Examples
     */
    for(size_t ex = 0; ex < nn->train_opts.negative; ex++) {
        random = nlk_random_xs1024(); 
        if(random != 0) {
            target = nn->neg_table[random % NLK_NEG_TABLE_SIZE];
        } else {
            target = nlk_random_xs1024() % (nn->words->weights->rows - 1) + 1; 
        }
        if(target == center_word) {
            /* ignore if this is the actual word */
            continue;
        }

        /* forward with lookup for target word */
        nlk_layer_lookup_forward(nn->neg, lk1_out, target, &lk2_out);

        /* shortcuts when outside of sigm bounds */
        if(lk2_out >= NLK_MAX_EXP) {
            grad_out = -learn_rate;
        } else if(lk2_out <= -NLK_MAX_EXP) {
            /* no error, do nothing */
            continue;
        } else {
            /* inside sigmoid bounds */
            out = nlk_sigmoid(lk2_out);

            /** @section Skipgram NEG Sampling Backprop
             * Same gradient formula but this time label is 0
             * multiply by learning rate 
             */
            grad_out = (1.0 - out) * learn_rate;
        }


        /* Backprop using accumulate gradient for all examples */
        nlk_layer_lookup_backprop_acc(nn->neg, lk1_out, target, grad_out, 
                                      grad_acc);
    } /* end of negative examples */
}


/**
 * Train CBOW for a series of word contexts
 *
 */
static void
nlk_cbow(struct nlk_neuralnet_t *nn, const nlk_real learn_rate, 
         const struct nlk_context_t *context,
         NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
#ifndef NCHECKS
    if(context->size == 0) {
        NLK_ERROR_ABORT("Context size must be > 0", NLK_EBADLEN);
    }
#endif

    nlk_array_zero(grad_acc);

    /** Word Lookup Forward
     * The context words get forwarded through the first lookup layer
     * and their vectors are averaged.
     */
    nlk_layer_lookup_forward_lookup_avg(nn->words, context->window, 
                                        context->size, lk1_out);

    /* Hierarchical Softmax */
    if(nn->train_opts.hs) {
        nlk_w2v_hs(nn, lk1_out, learn_rate, context->target, grad_acc);
    } 

    /* NEG Sampling  */
    if(nn->train_opts.negative) {
        nlk_w2v_neg(nn, learn_rate, context->target->index, lk1_out, 
                    grad_acc);
    }

    /** Backprop into the words using the accumulated gradient
     */
    nlk_layer_lookup_backprop_lookup(nn->words, context->window, 
                                     context->size, grad_acc);
}


/**
 * Train skipgram for a word context
 */
static void
nlk_skipgram(struct nlk_neuralnet_t *nn, const nlk_real learn_rate, 
             const struct nlk_context_t *context,
             NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    /* for each context word jj */
    for(size_t jj = 0; jj < context->size; jj++) {
        nlk_array_zero(grad_acc);


        /** Word Lookup
         * Each context word gets forwarded through the first lookup layer
         */
        nlk_layer_lookup_forward_lookup_one(nn->words, context->window[jj], 
                                            lk1_out);
        /* @TODO or equivalently w/o a copy:
         * (but w2v_train needs to save lk1->data pointer)
         * lk1_out->data = &lk1->weights->data[context->window[jj] 
         *                                      lk1->weights->cols];
         */

        /* Hierarchical Softmax */
        if(nn->train_opts.hs) {
            nlk_w2v_hs(nn, lk1_out, learn_rate, context->target, 
                       grad_acc);
        }
            
        /* NEG Sampling */
        if(nn->train_opts.negative) {
            nlk_w2v_neg(nn, learn_rate, context->target->index, lk1_out, 
                        grad_acc);
        }

        /** Backprop into the words using the accumulated gradient
         */
        nlk_layer_lookup_backprop_lookup_one(nn->words, context->window[jj], 
                                             grad_acc);

    } /* end of context words */
}


/**
 * Train PVDM (CBOW for PVs) for a context
 */
void
nlk_pvdm(struct nlk_neuralnet_t *nn, struct nlk_layer_lookup_t *par_table,
         const nlk_real learn_rate, const struct nlk_context_t *context, 
         NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{

    /* position of paragraph id */
    const size_t ppos = context->size - 1; /* = number of words */

#ifndef NCHECKS
    if(context->size == 0) {
        NLK_ERROR_ABORT("Context size must be > 0", NLK_EBADLEN);
    }
    /* check that the last context is a paragraph */
    if( ! context->is_paragraph[ppos]) {
        NLK_ERROR_VOID("last context element is not a paragraph", NLK_EINVAL);
    }
#endif


    nlk_array_zero(grad_acc);


    /** PVDM Forward through the first layer 
     * The window[-1] is the paragraph id and should be ignored 
     */
    nlk_layer_lookup_forward_lookup_one(par_table, 
                                        context->window[ppos],
                                        lk1_out);

    /* The context words get forwarded through the first lookup layer
     * and their vectors are averaged together with the PV.
     */
    nlk_layer_lookup_forward_lookup_avg_p(nn->words, context->window, 
                                          ppos, lk1_out);

    /* Hierarchical Softmax */
    if(nn->train_opts.hs) {
        nlk_w2v_hs(nn, lk1_out, learn_rate, context->target, grad_acc);
    } 

    /* NEG Sampling */
    if(nn->train_opts.negative) {
        nlk_w2v_neg(nn, learn_rate, context->target->index, lk1_out, grad_acc);
    }

    /* Backprop into the word vectors: Learn using the accumulated gradient */
    nlk_layer_lookup_backprop_lookup(nn->words, context->window, 
                                     ppos, grad_acc);

    /* Backprop into the PV: Learn PV weights using the accumulated gradient */
    nlk_layer_lookup_backprop_lookup_one(par_table, 
                                         context->window[ppos], 
                                         grad_acc);
}


/**
 * PVDM Concat (CBOW for PVs concatenating words+pvs instead of averaging)
 * The only thing different from the PVDM function is the use of 
 * nlk_layer_lookup_forward_lookup_concat_p instead of avg_p
 */
void
nlk_pvdm_cc(struct nlk_neuralnet_t *nn, struct nlk_layer_lookup_t *par_table,
            const nlk_real learn_rate, const struct nlk_context_t *context,
            NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    /* position of paragraph id */
    const size_t ppos = context->size - 1; /* = number of words */

#ifndef NCHECKS
    if(context->size == 0) {
        NLK_ERROR_ABORT("Context size must be >= 2", NLK_EBADLEN);
    }
    /* check that this (last context) is a paragraph */
    if(!context->is_paragraph[ppos]) {
        NLK_ERROR_VOID("last context element is not a paragraph", NLK_EINVAL);
    }
#endif

    nlk_array_zero(grad_acc);

    /* PVDM Forward through the first layer
     * the first element of lk1_out (position 0) is the PV 
     */
    nlk_layer_lookup_forward_lookup_one(par_table, context->window[ppos],
                                        lk1_out);

    /* The context words get forwarded through the first lookup layer
     * and their vectors are concatenate together with the PV.
     * concat_p starts concatenation into lk1_out at posion 1 (after the PV)
     */
    nlk_layer_lookup_forward_lookup_concat_p(nn->words, context->window, 
                                             ppos, lk1_out);

    /* Hierarchical Softmax */
    if(nn->train_opts.hs) {
        nlk_w2v_hs(nn, lk1_out, learn_rate, context->target, grad_acc);
    } 

    /* NEG Sampling */
    if(nn->train_opts.negative) {
        nlk_w2v_neg(nn, learn_rate, context->target->index, lk1_out, grad_acc);
    }

    /* Backprop into the PV: Learn PV weights using the accumulated gradient. 
     * The PV in the gradient is at position 0
     */
    nlk_layer_lookup_backprop_lookup_concat_one(par_table, 
                                                context->window[ppos], 
                                                0, grad_acc);

    /* Backprop into the word vectors: Learn using the accumulated gradient 
     * words in the gradient start at position 1 (after PV)
     */
    nlk_layer_lookup_backprop_lookup_concat(nn->words, context->window, 
                                            ppos, 1, grad_acc);
}


/**
 * Train PVDBOW (skipgram for PVs)
 * In PVDBOW each "context" is just the target word which is associated with 
 * the PV.
 */
void
nlk_pvdbow(struct nlk_neuralnet_t *nn, struct nlk_layer_lookup_t *par_table, 
           const nlk_real learn_rate, const struct nlk_context_t *context, 
           NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    /* for each context word jj */
    for(size_t jj = 0; jj < context->size; jj++) {
        nlk_array_zero(grad_acc);

        /** PVDM Lookup Words or Paragraphs
         * Each context word gets forwarded through the first lookup layer
         */
        if(context->is_paragraph[jj]) {
            nlk_layer_lookup_forward_lookup_one(par_table, 
                                                context->window[jj], lk1_out);
        } else {
            nlk_layer_lookup_forward_lookup_one(nn->words, 
                                                context->window[jj], lk1_out);
        }

        /* Hierarchical Softmax */
        if(nn->train_opts.hs) {
            nlk_w2v_hs(nn, lk1_out, learn_rate, context->target, grad_acc);
        }
            
        /* NEG Sampling */
        if(nn->train_opts.negative) {
            nlk_w2v_neg(nn, learn_rate, context->target->index, lk1_out, 
                        grad_acc);
        }

        /** Backprop into Words or Paragraphs using the accumulated gradient
         */
        if(context->is_paragraph[jj] == true) { 
           nlk_layer_lookup_backprop_lookup_one(par_table, 
                                                context->window[jj], 
                                                grad_acc);
        } else if(context->is_paragraph[jj] == false) {
            nlk_layer_lookup_backprop_lookup_one(nn->words, 
                                                 context->window[jj], 
                                                 grad_acc);
        }
    } /* end of context words */
}


/**
 * Train or update a word2vec model
 *
 * @param nn                the neural network
 * @param train_file        the path of the train file
 * @param verbose
 *
 * @return total error for last epoch
 */
void
nlk_w2v(struct nlk_neuralnet_t *nn, const char *train_file, const bool verbose)
{
    /*goto_set_num_threads(1);*/

    /* unpack training options */
    NLK_LM model_type = nn->train_opts.model_type;
    nlk_real learn_rate = nn->train_opts.learn_rate;
    unsigned int epochs = nn->train_opts.iter;
    struct nlk_context_opts_t context_opts = nn->context_opts;
    unsigned int ctx_size = context_opts.max_size;
    float sample_rate = nn->train_opts.sample;
    const size_t train_words = nn->train_opts.word_count;
    const size_t train_paragraphs = nn->train_opts.paragraph_count;

    /* shortcuts */
    struct nlk_vocab_t **vocab = &nn->vocab;
    struct nlk_vocab_t *replacement = nlk_vocab_find(vocab, NLK_UNK_SYMBOL);

    size_t layer_size2 = 0;

    if(nn->train_opts.hs) {
        layer_size2 = nn->hs->weights->cols;
    } else if(nn->train_opts.negative) {
        layer_size2 = nn->neg->weights->cols;
    } else {
        NLK_ERROR_ABORT("Hierarchical Softmax or Negative Sampling required",
                        NLK_EINVAL);
        /* unreachable */
    }

    struct nlk_layer_lookup_t *par_table = nn->paragraphs;

    /* for vocabularize we need the real total number of words in our corpus */
    

    /** @section Shared Initializations
     * All variables declared in this section are shared among threads.
     * Most are read-only but some (e.g. learn_rate & word_count_actual)
     * are updated directly from the threads.
     */

    /** @subsection Input and Context/Window initializations
     * Alllocations and initializations related to the input (text)
     */
    size_t word_count_actual = 0;   /* @warn thread write-shared */


    /** @subsection Neural Net initializations
     * Create and initialize neural net and associated variables 
     */
    nlk_real learn_rate_start = learn_rate;

    /* neg table for negative sampling */
    if(nn->train_opts.negative && nn->neg_table == NULL) {
        nn->neg_table = nlk_vocab_neg_table_create(vocab, NLK_NEG_TABLE_SIZE, 
                                                   NLK_NEG_TABLE_POW);
    }

    /* time keeping */
    clock_t start = clock();
    nlk_tic_reset();
    nlk_tic(NULL, false);

    /* threads */
    int num_threads = nlk_get_num_threads();


    /** @section Thread Private initializations 
     * Variables declared in this section are thread private and thus 
     * have thread specific values
     */
#pragma omp parallel shared(word_count_actual)
{
    /** @subsection File Reading
     * The train file is divided into parts, one part for each thread.
     * Open file and move to thread specific starting point
     */
    int train_fd = nlk_open(train_file);
    size_t line_start = 0;      /**< first line for thread */
    size_t line_end = 0;        /**< last line for thread */
    size_t line_cur = 0;        /**< line being read/processed by thread */
    off_t train_file_start = 0; /**< thread specific start */
    char **text_line = nlk_text_line_create();

    char *buffer = malloc(sizeof(char) * NLK_BUFFER_SIZE);
    if(buffer == NULL) {
        NLK_ERROR_ABORT("not enough memory", NLK_ENOMEM);
        /* UNREACHABLE */
    }

    /** @subsection Progress
     */
    size_t word_count = 0;
    size_t last_word_count = 0;
    unsigned int local_epoch = 0;   /* current epoch (thread local) */

    /** @subsection Neural Network Forward/Backward
     */
    /* output of the first layer */
    NLK_ARRAY *layer1_out = nlk_array_create(layer_size2, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size2);
    

    /** @subsection Context
     */
    unsigned int n_examples;
    unsigned int ex;

    /* vocabularized line */
    struct nlk_line_t *line = nlk_line_create(NLK_MAX_LINE_SIZE);

    /* for undersampling words in a line */
    struct nlk_line_t *line_sample = nlk_line_create(NLK_MAX_LINE_SIZE);
 

    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = nlk_context_create_array(ctx_size);


#pragma omp for 
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        /* set train file part position */
        line_cur = nlk_text_get_split_start_line(train_paragraphs, num_threads, 
                                                 thread_id);
        line_start = line_cur;
        line_end = nlk_text_get_split_end_line(train_paragraphs, num_threads, 
                                                  thread_id);
        train_file_start = nlk_text_goto_line(train_fd, line_cur);

        /** @section Start of Training Loop (Epoch Loop)
         */
        while(local_epoch < epochs) {
            /** @subsection Epoch End Check
             * Increment thread private epoch counter if necessary
             */
            /* check for end of file part, decrement local epochs, rewind */
            if(line_cur > line_end) {
                /* update count and epoch */
                word_count_actual += word_count - last_word_count;
                local_epoch++;

                /* rewind */
                word_count = 0;
                last_word_count = 0;
                line_cur = line_start;
                nlk_text_goto_location(train_fd, train_file_start);
                continue;
            }

            /** @subsection Update Counts and Learning Rate, Display
             */

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
                learn_rate = nlk_learn_rate_w2v(learn_rate, learn_rate_start,
                                                epochs, word_count_actual, 
                                                train_words);
            }

            /** @subsection Read from File and Create Context Windows
             * The actual difference between the word models and the paragraph
             * models is in the context that gets generated here.
             */
            nlk_vocab_read_vocabularize(train_fd, vocab, replacement, 
                                        text_line, line, buffer);

            /* check for errors, empty lines, etc */
            if(line->len == 0) {
                line_cur++;
                continue;
            }

            /* subsample  */
            nlk_vocab_line_subsample(line, train_words, sample_rate, 
                                     line_sample);

            /* single word, nothing to do ... */
            if(line_sample->len < 2) {
                line_cur++;
                continue;
            }

            /* Context Window
             */
            n_examples = nlk_context_window(line_sample->varray, 
                                            line_sample->len, 
                                            line_sample->line_id, 
                                            &context_opts, contexts);

            /** @subsection Algorithm Parallel Loop Over Contexts
             */
            switch(model_type) {
                case NLK_SKIPGRAM:
                    for(ex = 0; ex < n_examples; ex++) {
                        nlk_skipgram(nn, learn_rate, contexts[ex], grad_acc, 
                                     layer1_out);
                    }
                    break;
                case NLK_CBOW:
                    for(ex = 0; ex < n_examples; ex++) {
                        nlk_cbow(nn, learn_rate, contexts[ex], grad_acc, 
                                 layer1_out);
                    }
                    break;
                case NLK_PVDBOW:
                    for(ex = 0; ex < n_examples; ex++) {
                        nlk_pvdbow(nn, par_table, learn_rate, 
                                   contexts[ex], grad_acc, layer1_out);
                    }
                    break;
                case NLK_PVDM:
                    for(ex = 0; ex < n_examples; ex++) {
                        nlk_pvdm(nn, par_table, learn_rate, contexts[ex], 
                                 grad_acc, layer1_out);
                    }
                    break;
                case NLK_PVDM_CONCAT:
                    for(ex = 0; ex < n_examples; ex++) {
                        nlk_pvdm_cc(nn, par_table, learn_rate, contexts[ex], 
                                    grad_acc, layer1_out);
                    }
                    break;
                default:
                    NLK_ERROR_ABORT("invalid model type", NLK_EINVAL);
                    /* unreachable */
            }

                
            /* update count/location */
            word_count += line->len;
            line_cur++;

        } /* end of epoch cycle */
    } /* end of threaded algorithm execution */

    /** @subsection Free Thread Private Memory and Close Files
     */
    free(buffer);
    nlk_text_line_free(text_line);
    nlk_context_free_array(contexts);
    nlk_line_free(line_sample);
    nlk_array_free(layer1_out);
    nlk_array_free(grad_acc);


    } /* *** End of Paralell Region *** */
    /** @section End
     */
    free(nn->neg_table);
    nn->neg_table = NULL;
    nlk_tic_reset();
}


/**
 * Export word-vector pairs in word2vec text compatible format
 */
void
nlk_w2v_export_word_vectors(NLK_ARRAY *weights, NLK_FILE_FORMAT format, 
                            struct nlk_vocab_t **vocab, const char *filepath)
{
    size_t cc = 0;
    size_t vocab_size = nlk_vocab_size(vocab);

    if(format != NLK_FILE_W2V_BIN && format != NLK_FILE_W2V_TXT) {
        NLK_ERROR_VOID("unsuported file format", NLK_EINVAL);
        /* unreachable */
    }

    /* open file */
    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }
   
    /* word vectors */
    for(size_t row_index = 0; row_index < vocab_size; row_index++) {
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
    }   /* end of word vectors */

    fclose(out); 
    out = NULL;
}


void
nlk_w2v_export_paragraph_vectors(NLK_ARRAY *weights, NLK_FILE_FORMAT format, 
                                 const char *filepath)
{
    size_t cc;

    /* open file */
    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }
    for(size_t row_index = 0; row_index < weights->rows; row_index++) {
        fprintf(out, "*_%zu ", row_index);
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
    }   /* end of paragraphs */

    fclose(out);
    out = NULL;
}
