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
static inline void
nlk_word2vec_display(nlk_real learn_rate, size_t word_count_actual, 
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
nlk_word2vec_create(size_t vocab_size, size_t paragraphs, size_t layer_size, 
                    bool hs, bool neg)
{
    struct nlk_neuralnet_t *nn;
    struct nlk_layer_lookup_t *lk1;
    struct nlk_layer_lookup_t *lkhs;
    struct nlk_layer_lookup_t *lkneg;
    int n_layers = 1;

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

    /* lookup layer 1 */
    lk1 = nlk_layer_lookup_create(vocab_size + paragraphs, layer_size);

    /* random number generator initialization */
    uint64_t seed = 6121984 * clock();
    seed = nlk_random_fmix(seed);
    nlk_random_init_xs1024(seed);

    nlk_layer_lookup_init(lk1);
    nlk_neuralnet_add_layer_lookup(nn, lk1);
    
    /* lookup layer HS */
    if(hs) {
        lkhs = nlk_layer_lookup_create(vocab_size, layer_size);
        /* [ default initialization: zero ] */
        nlk_neuralnet_add_layer_lookup(nn, lkhs);
    }

    /* lookup layer NEG */
    if(neg) {
        lkneg = nlk_layer_lookup_create(vocab_size, layer_size);
        /* [ default initialization: zero ] */
        nlk_neuralnet_add_layer_lookup(nn, lkneg);
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
static inline nlk_real
hierarchical_softmax(NLK_LAYER_LOOKUP *lk2hs, const bool update,
                     const NLK_ARRAY *lk1_out, const nlk_real learn_rate, 
                     const nlk_real *sigmoid_table,
                     const struct nlk_vocab_t *center_word,
                     NLK_ARRAY *grad_acc)
{
    nlk_real out;
    nlk_real lk2_out;
    nlk_real grad_out;
    size_t point;
    uint8_t code;
    nlk_real err = 0;

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
        err += fabsf(grad_out);

        /* gradient */
        grad_out = 1.0 - grad_out;

        /* multiply by learning rate */
        grad_out *= learn_rate;
        
        /* layer2hs backprop, accumulate gradient for all points */
        nlk_layer_lookup_backprop_acc(lk2hs, update, lk1_out, point, grad_out, 
                                      grad_acc);

    } /* end of points/codes */

    return err;
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
static inline nlk_real
negative_sampling(NLK_LAYER_LOOKUP *lk2neg, const bool update, 
                  const size_t *neg_table,
                  const size_t negative, const size_t vocab_size,
                  const nlk_real learn_rate, 
                  const size_t center_word, 
                  const NLK_ARRAY *lk1_out, const nlk_real *sigmoid_table, 
                  NLK_ARRAY *grad_acc)
{
    nlk_real out;
    size_t target;
    uint64_t random;
    nlk_real grad_out;
    nlk_real lk2_out;
    nlk_real err = 0;


    /** @section Positive Example
     */
    /* forward with lookup for target word */
    nlk_layer_lookup_forward(lk2neg, lk1_out, center_word, &lk2_out);

    /* shortcuts when outside of sigm bounds */
    if(lk2_out >= NLK_MAX_EXP) {
        /* do nothing, no error */
    } else if(lk2_out <= -NLK_MAX_EXP) {
        grad_out = learn_rate;
        err += 1.0;

       /* layer2neg backprop, accumulate gradient for all examples */
        nlk_layer_lookup_backprop_acc(lk2neg, update, lk1_out, center_word, 
                                      grad_out, grad_acc);
    } else {
        /* inside sigmoid bounds */
        out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);
        err =+ fabsf(1.0 - out);

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
            err += 1.0;
        } else if(lk2_out <= -NLK_MAX_EXP) {
            /* no error, do nothing */
            continue;
        } else {
            /* inside sigmoid bounds */
            out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);
            err += err;

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

    return err;
}

/**
 * Train CBOW for a series of word contexts
 *
 */
static inline void
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

    /** @section CBOW Forward through the first layer
     * The context words get forwarded through the first lookup layer
     * and their vectors are averaged.
     */
    nlk_array_zero(lk1_out);
    nlk_layer_lookup_forward_lookup_avg(lk1, context->window, context->size, 
                                        lk1_out);
    nlk_array_zero(grad_acc);

    /** @section CBOW Hierarchical Softmax 
     */
    if(hs) {
        hierarchical_softmax(lk2hs, true, lk1_out, learn_rate, sigmoid_table,
                             context->center, grad_acc);
    } 

    /** @section CBOW NEG Sampling Forward
     */
    if(negative) {
        negative_sampling(lk2neg, true, neg_table, negative, vocab_size, 
                          learn_rate, context->center->index, lk1_out, 
                          sigmoid_table, grad_acc);

    }

    /** @section Backprop into the first lookup layer
     * Learn layer1 weights using the accumulated gradient
     */
    nlk_layer_lookup_backprop_lookup(lk1, context->window, context->size, 
                                     grad_acc);
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

        
        /** @section Skipgram Hierarchical Softmax */
        if(hs) {
            hierarchical_softmax(lk2hs, true, lk1_out, learn_rate, 
                                 sigmoid_table, context->center, grad_acc);
        }
            
        /** @section Skipgram NEG Sampling
         */
        if(negative) {
            negative_sampling(lk2neg, true, neg_table, negative, vocab_size, 
                              learn_rate, context->center->index, lk1_out, 
                              sigmoid_table, grad_acc);
        }   /* end of NEG sampling specific code */

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
 * @param model_type        NLK_SKIPGRAM, NLK_CBOW
 * @param nn                the neural network
 * @param hs                use hierarchical softmax
 * @param negative          number of negative examples to use (NEG sampling)
 * @param train_file_path   the path of the train file
 * @param lower             convert read words to lower case
 * @param vocab             the vocabulary
 * @param total_lines       total lines in train file (used for PVs)
 * @param window            
 * @param sample_rate
 * @param learn_rate
 * @param epochs
 * @param verbose
 *
 * @return total error for last epoch
 */
void
nlk_word2vec(const NLK_LM model_type,  struct nlk_neuralnet_t *nn, 
             const bool hs, const unsigned int negative, 
             const char *train_file_path, const bool lower,
             struct nlk_vocab_t **vocab, const size_t total_lines, 
             const size_t window, const float sample_rate, 
             nlk_real learn_rate, unsigned int epochs, int verbose)
{
    /*goto_set_num_threads(1);*/

    /* shortcuts */
    struct nlk_layer_lookup_t *lk1 = nn->layers[0].lk;
    size_t layer_n = 1;
    struct nlk_layer_lookup_t *lkhs = NULL;
    if(hs) {
        lkhs = nn->layers[layer_n].lk;
        layer_n++;
    }
    struct nlk_layer_lookup_t *lkneg = NULL;
    if(negative) {
        lkneg = nn->layers[layer_n].lk;
    }

    const size_t layer_size = lk1->weights->cols;
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
    if(model_type == NLK_PVDBOW) { /* PVDBOW */
        context_multiplier = ctx_size;
        ctx_size = 1;   /* one (word, paragraph) pair at a time */
        learn_par = true;
    } else if(model_type == NLK_PVDM) { /* PVDM */
        ctx_size += 1; /* space for the paragraph (1st elem of window) */
    }

    struct nlk_context_opts_t ctx_opts;
    nlk_context_model_opts(model_type, window, &ctx_opts);

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
    NLK_ARRAY *lk1_out = nlk_array_create(layer_size, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size);
    
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

    wchar_t *low_tmp = NULL; 
    if(lower) {
        low_tmp = (wchar_t *) malloc(max_word_size * sizeof(wchar_t));
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
                    nlk_word2vec_display(learn_rate, word_count_actual, 
                                         train_words, epochs, local_epoch, 
                                         start);
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
            nlk_read_line(train, text_line, low_tmp, max_word_size,
                          max_line_size);
            
            /* vocabularize */
            line_len = nlk_vocab_vocabularize(vocab, train_words, text_line, 
                                              sample_rate, 
                                              NULL, true, vectorized, 
                                              &n_subsampled); 

            /* line_len = 1 would be an empty line, nothing to learn */
            if(line_len < 2) {
                continue;
            }

            /* Context Window
             * paragraph id (index in weight matrix) = line number + vocab_size
             */
            par_id = line_cur + vocab_size;
            n_examples = nlk_context_window(vectorized, line_len, par_id, 
                                            &ctx_opts, contexts);


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
    if(low_tmp != NULL) {
        free(low_tmp);
    }
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
 * Generate a PV using PVDM
 */
static inline nlk_real
nlk_pv_dm_gen(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
              const bool hs, NLK_LAYER_LOOKUP *lk2neg, const size_t negative,
              const size_t *neg_table, const size_t vocab_size, 
              const nlk_real learn_rate, const nlk_real *sigmoid_table, 
              const struct nlk_context_t *context, NLK_ARRAY *grad_acc, 
              NLK_ARRAY *lk1_out, NLK_ARRAY *pv)
{
#ifndef NCHECKS
    if(context->size == 0) {
        NLK_ERROR_ABORT("Context size must be > 0", NLK_EBADLEN);
    }
#endif

    nlk_real err = 0;

    /* PVDM Forward through the first layer */
    nlk_array_zero(lk1_out);
    nlk_array_zero(grad_acc);

    /* The context words get forwarded through the first lookup layer
     * and their vectors are averaged together with the PV.
     * The window[0] is the paragraph id and should be ignored as the PV
     * is provided independently of the rest of the lookup layer
     */
    nlk_layer_lookup_forward_lookup_avg_p(lk1, &context->window[1], 
                                          context->size - 1, pv, lk1_out);

    /* Hierarchical Softmax */
    if(hs) {
        err += hierarchical_softmax(lk2hs, false, lk1_out, learn_rate, 
                                    sigmoid_table, context->center, grad_acc);
    } 

    /* NEG Sampling */
    if(negative) {
        err += negative_sampling(lk2neg, false, neg_table, negative, 
                                 vocab_size, learn_rate, 
                                 context->center->index, lk1_out, 
                                 sigmoid_table, grad_acc);
    }

    /* Backprop into the PV: Learn PV weights using the accumulated gradient */
    nlk_array_add(grad_acc, pv);

    return err;
}

/**
 * Generate a PV using PVDBOW
 *
 * In PVDBOW each "context" is just the target word which is associated with 
 * the PV.
 */
static inline nlk_real
nlk_pv_dbow_gen(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
                const bool hs, NLK_LAYER_LOOKUP *lk2neg, const size_t negative,
                const size_t *neg_table, const size_t vocab_size, 
                const nlk_real learn_rate, const nlk_real *sigmoid_table, 
                const struct nlk_context_t *context, NLK_ARRAY *grad_acc, 
                NLK_ARRAY *lk1_out, NLK_ARRAY *pv)
{
    nlk_real err = 0;
    nlk_array_zero(grad_acc);

    /* Hierarchical Softmax */
    if(hs) {
        err += hierarchical_softmax(lk2hs, false, pv, learn_rate, 
                                    sigmoid_table, context->center, grad_acc);
    }
        
    /* NEG Sampling */
    if(negative) {
        err += negative_sampling(lk2neg, false, neg_table, negative, 
                                 vocab_size, learn_rate, 
                                 context->center->index, lk1_out, 
                                 sigmoid_table, grad_acc);
    }

    /* Backprop into the PV */
    nlk_array_add(grad_acc, pv);


    return err;
}


static void
nlk_pv_display_one(nlk_real err, nlk_real prev_err, size_t iter) {
    char display_str[64];
    nlk_real diff = err - prev_err;

    snprintf(display_str, 64,
            "Thread: %d,  Iter: %zu, Diff: %.5f; err: %.5f", 
            omp_get_thread_num(), iter, diff, err);
    nlk_tic(display_str, true);

}

/**
 * Learn a Paragraph vector for a single line
 */
nlk_real
nlk_pv_gen_one(const NLK_LM model_type, struct nlk_neuralnet_t *nn, 
               const bool hs, unsigned int negative, nlk_real learn_rate, 
               const nlk_real tol, struct nlk_vocab_t **vocab, 
               const size_t vocab_size, char **paragraph, 
               const size_t *neg_table, const nlk_real *sigmoid_table, 
               struct nlk_context_t **contexts, NLK_CONTEXT_OPTS *ctx_opts, 
               NLK_ARRAY *pv, NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    nlk_real err = 1e20;
    nlk_real prev_err = 0;
    nlk_real start_learn_rate = learn_rate;
    size_t n_examples;
    size_t n_subsampled;
    size_t line_len;
    struct nlk_vocab_t *vectorized[NLK_LM_MAX_LINE_SIZE];


    struct nlk_layer_lookup_t *lk1 = nn->layers[0].lk;
    size_t layer_n = 1;
    struct nlk_layer_lookup_t *lkhs = NULL;
    if(hs) {
        lkhs = nn->layers[layer_n].lk;
        layer_n++;
    }
    struct nlk_layer_lookup_t *lkneg = NULL;
    if(negative) {
        lkneg = nn->layers[layer_n].lk;
    }


    /* vocabularize */
    line_len = nlk_vocab_vocabularize(vocab, 0, paragraph, 0, NULL, false, 
                                      vectorized, &n_subsampled); 

    /* context window */
    n_examples = nlk_context_window(vectorized, line_len, 0, 
                                    ctx_opts, contexts);

    size_t n_iter = 1;
    if(model_type == NLK_PVDBOW) {
        do {
            err = 0;
            for(size_t ex = 0; ex < n_examples; ex++) {
                err += nlk_pv_dbow_gen(lk1, lkhs, hs, lkneg, negative, 
                                       neg_table, vocab_size, learn_rate, 
                                       sigmoid_table, contexts[ex], grad_acc, 
                                       lk1_out, pv);

                learn_rate = nlk_learn_rate_dec_update(learn_rate,
                                                       start_learn_rate, 
                                                       0.99);
            }
            n_iter++;
        } while(err > tol);
    } else if(model_type == NLK_PVDM) {
        do {
            prev_err = err;
            err = 0;
            for(size_t ex = 0; ex < n_examples; ex++) {
                err += nlk_pv_dm_gen(lk1, lkhs, hs, lkneg, negative, neg_table, 
                                     vocab_size, learn_rate, sigmoid_table, 
                                     contexts[ex], grad_acc, lk1_out, pv);

                learn_rate = nlk_learn_rate_dec_update(learn_rate,
                                                       start_learn_rate, 
                                                       0.99);
            }
            if(n_iter % 100000 == 0) {
                nlk_pv_display_one(err, prev_err, n_iter);
            }
            n_iter++;
        } while(err > tol);
    } else {
        NLK_ERROR_ABORT("invalid model type", NLK_EINVAL);
        /* unreachable */
    }

    return err;
}

/**
 * Display progress of paragraph vector generation
 *
 * @param generated number of PVs generated so far
 * @param total     total number of PVs to generate
 */
static void
nlk_pv_display(const size_t generated, const size_t total, nlk_real last_err)
{
    char display_str[64];
    float progress = (generated / (float) total) * 100;

    snprintf(display_str, 64,
            "Progress: %.2f%% (%zu/%zu) last_err: %.5f Threads: %d\t", 
            progress, generated, total, last_err, omp_get_num_threads());
    nlk_tic(display_str, true);
}


/**
 * Learn a series of paragraph vectors from a file
 *
 */
NLK_ARRAY *
nlk_pv(NLK_LM model_type, struct nlk_neuralnet_t *nn, const bool hs, 
       const unsigned int negative, const char *par_file_path, 
       const bool lower, struct nlk_vocab_t **vocab, const size_t window, 
       nlk_real learn_rate, nlk_real tol, int verbose)
{
    /* shortcuts */
    struct nlk_layer_lookup_t *lk1 = nn->layers[0].lk;
    size_t layer_size = lk1->weights->cols;
    size_t vocab_size = nlk_vocab_size(vocab);

    /* context */
    size_t max_word_size = NLK_LM_MAX_WORD_SIZE;
    size_t max_line_size = NLK_LM_MAX_LINE_SIZE;

    size_t ctx_size = window * 2;   /* max context size */
    size_t context_multiplier = 1;  /* != 1 only for PVDBOW */

    bool learn_par = false;
    if(model_type == NLK_PVDBOW) { /* PVDBOW */
        context_multiplier = ctx_size;
        ctx_size = 1;   /* one (word, paragraph) pair at a time */
        learn_par = true;
    } else if(model_type == NLK_PVDM) { /* PVDM */
        ctx_size += 1; /* space for the paragraph (1st elem of window) */
    }

    struct nlk_context_opts_t ctx_opts;
    nlk_context_model_opts(model_type, window, &ctx_opts);


    /* open file */
    errno = 0;
    FILE *in = fopen(par_file_path, "rb");
    if(in == NULL) {
        NLK_ERROR_ABORT(strerror(errno), errno);
        /* unreachable */
    }

    /* count lines */
    size_t total_lines = nlk_text_count_lines(in);
    if(verbose) {
        printf("Generating Paragraphs: %zu\n", total_lines);
    }

    /* file size */
    size_t par_file_size;
    fseek(in, 0, SEEK_END);
    par_file_size = ftell(in);
    fclose(in);

    /* random number generator initialization */
    uint64_t seed = 6121984 * clock();
    seed = nlk_random_fmix(seed);
    nlk_random_init_xs1024(seed);

    /* create and init PVs */
    NLK_ARRAY *par_vectors = nlk_array_create(total_lines, layer_size);
    nlk_layer_lookup_init_array(par_vectors);

    /* sigmoid table */
    nlk_real *sigmoid_table = nlk_table_sigmoid_create();

    /* neg table for negative sampling */
    size_t *neg_table = NULL;
    if(negative) {
        neg_table = nlk_vocab_neg_table_create(vocab, NLK_NEG_TABLE_SIZE, 
                                               0.75);
    }

    size_t generated = 0;

#pragma omp parallel
{
    int num_threads = omp_get_num_threads();

    /* current paragraph vector (empty shell) */
    NLK_ARRAY pv;
    pv.cols = 1;
    pv.rows = layer_size;


    /* output of the first layer */
    NLK_ARRAY *lk1_out = nlk_array_create(layer_size, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size);
    
    /* Input Text and Context Window initializations */
    
    /* allocate memory for reading from the input file */
    char **text_line = (char **) calloc(max_line_size, sizeof(char *));
    if(text_line == NULL) {
        NLK_ERROR_ABORT("unable to allocate memory for text", NLK_ENOMEM);
        /* unreachable */
    }
    for(size_t zz = 0; zz < max_line_size; zz++) {
        text_line[zz] = calloc(max_word_size, sizeof(char));
        if(text_line[zz] == NULL) {
            NLK_ERROR_ABORT("unable to allocate memory for text", NLK_ENOMEM);
            /* unreachable */
        }
    }
    wchar_t *low_tmp = NULL; 
    if(lower) {
        low_tmp = (wchar_t *) malloc(max_word_size * sizeof(wchar_t));
    }


    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = (struct nlk_context_t **) 
        malloc(max_line_size * context_multiplier * 
               sizeof(struct nlk_context_t *));
    if(contexts == NULL) {
        NLK_ERROR_ABORT("unable to allocate memory for contexts", NLK_ENOMEM);
        /* unreachable */
    }

    for(size_t zz = 0; zz < max_line_size * context_multiplier; zz++) {
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
    FILE *par_file = fopen(par_file_path, "rb");
    nlk_real err = 0;

#pragma omp for 
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        /* set train file part position */
        end_pos = nlk_set_file_pos(par_file, learn_par, par_file_size,
                                   num_threads, thread_id);

        /* determine line number */
        if(learn_par) {
            line_start = nlk_text_get_line(par_file);
            line_cur = line_start;
        }

        while(true) {
           /* read line */
            nlk_read_line(par_file, text_line, low_tmp, max_word_size,
                          max_line_size);

            /* select paragraph */
            pv.data = &par_vectors->data[line_cur * layer_size];


            /* generate */
            err = nlk_pv_gen_one(model_type, nn, hs, negative, learn_rate, tol,
                           vocab, vocab_size, text_line, neg_table, 
                           sigmoid_table, contexts, &ctx_opts, &pv, grad_acc, 
                           lk1_out);

            generated += 1;
            line_cur++;
            file_pos = ftell(par_file);
            nlk_pv_display(generated, total_lines, err);
            if(file_pos >= end_pos) {
                break;
            }
        }     
    } /* end of threaded algorithm execution */

    /** @subsection Free Thread Private Memory and Close Files
     */
    for(size_t zz = 0; zz < max_line_size * context_multiplier; zz++) {
        nlk_context_free(contexts[zz]);
    }
    for(size_t zz = 0; zz < max_line_size; zz++) {
        free(text_line[zz]);
    }
    free(text_line);
    if(low_tmp != NULL) {
        free(low_tmp);
    }
    nlk_array_free(lk1_out);
    nlk_array_free(grad_acc);
    fclose(par_file);
} /* *** End of Paralell Region *** */

    /** @section End
     */
    nlk_tic_reset();
    free(sigmoid_table);
    if(negative) {
        free(neg_table);
    }

    return par_vectors;
}
