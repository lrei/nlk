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


/** @file nlk_skipgram.c
 * Word2Vec Implementation: CBOW & kipgram models
 */

#include <time.h>
#include <errno.h>
#include <math.h>
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

#include "nlk_w2v.h"


/**
 * Learn rate update function from word2vec
 *
 * @param learn_rate        current learn rate
 * @param start_learn_rate  starting learn rate
 * @param epochs            total number of epochs
 * @param word_count_actual total number of words seen so far
 * @param train_words       total number of words in train file
 *
 * @return  new learn rate
 */
nlk_real
nlk_word2vec_learn_rate_update(nlk_real learn_rate, nlk_real start_learn_rate,
                               size_t epochs, size_t word_count_actual, 
                               size_t train_words)
{
    learn_rate = start_learn_rate * (1 - word_count_actual / 
                                     (nlk_real) (epochs * train_words + 1));

    if(learn_rate < start_learn_rate * 0.0001) {
        learn_rate = start_learn_rate * 0.0001;
    }

    return learn_rate;
}

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
nlk_word2vec_display(nlk_real learn_rate, size_t word_count_actual, 
                     size_t train_words, int epochs, int epoch, 
                     clock_t start)
{
    double progress;
    double speed;
    char display_str[256];

    clock_t now = clock();

    progress = word_count_actual / 
               (double)(epochs * train_words + 1) * 100;
    speed = word_count_actual / ((double)(now - start + 1) / 
            (double)CLOCKS_PER_SEC * 1000),
    snprintf(display_str, 256,
            "Alpha: %f  Progress: %.2f%% (%3d/%d) "
            "Words/Thread/sec: %.2fk Threads: %d", 
            learn_rate, progress, (epochs - epoch), epochs, speed, 
            omp_get_num_threads());
    nlk_tic(display_str, false);
}

/**
 * Create a word2vec neural network
 *
 * @param vocab_size    the vocabulary size
 * @param layer_size    the hidden layer size
 *
 * @return the neural network structure
 */
struct nlk_neuralnet_t *
nlk_word2vec_create(size_t vocab_size, size_t layer_size, bool hs, bool neg)
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
    lk1 = nlk_layer_lookup_create(vocab_size, layer_size);
    tinymt32_t rng;
    tinymt32_init(&rng,  447290 + clock());
    nlk_layer_lookup_init(lk1, &rng);
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
 * Train skipgram for a series of word contexts
 *
 */
void
nlk_skipgram_for_context(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
                         bool hs, NLK_LAYER_LOOKUP *lk2neg, size_t negative,
                         size_t *neg_table, nlk_real learn_rate, 
                         nlk_real *sigmoid_table, 
                         nlk_Context *context,
                         NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    nlk_real lk2_out;
    nlk_real out;
    nlk_real grad_out;

    nlk_Vocab *center_word = context->center;
    nlk_Vocab *word;
    size_t point;
    uint8_t code;
    size_t target;
    uint64_t random;
    unsigned short int label;


    /* for each context word jj */
    for(size_t jj = 0; jj < context->size; jj++) {
        word = context->window[jj];
        nlk_array_zero(grad_acc);


        /** @section Skipgram Lookup Layer1 Forward (common)
         * Each context word gets forwarded through the first lookup layer
         */
        nlk_layer_lookup_forward_one(lk1, word->index, lk1_out);


        
        /** @section Skipgram Hierarchical Softmax Forward
         * the center word (target) gets forwarded through the second lookup 
         * layer. If using hierarchical softmax, the second lookup layer
         * "maps" points to codes (through the output softmax)
         */
        if(hs) {
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


                /** @section Skipgram Hierarchical Softmax Backprop
                 * Conveniently, using the negative log likelihood,
                 * the gradient simplifies to the same formulation/code
                 * as the binary neg likelihood:
                 *
                 * log(sigma(z=v'n(w,j))'vwi) = 
                 * = (1 - code) * z - log(1 + e^z)
                 * d/dx = 1 - code  - sigmoid(z)
                 */

                nlk_bin_nl_sgradient(out, code, &grad_out);

                /* multiply by learning rate */
                grad_out *= learn_rate;
                
                /* layer2hs backprop, accumulate gradient for all points */
                nlk_layer_lookup_backprop_acc(lk2hs, lk1_out, point, grad_out, 
                                              grad_acc);

            } /* end of points/codes */
        } /* end of hierarchical softmax specific code */

        /** @section Skipgram NEG Sampling Forward 
         */
        if(negative) {
            for(size_t nn = 0; nn < negative + 1; nn++) {
                if(nn == 0) { /* postive example */
                    label = 1;
                    target = center_word->index;
                } else { /* negative examples */
                    label = 0;
                    random = nlk_random_xs1024(); 
                    target = neg_table[random % NLK_NEG_TABLE_SIZE];
                    if(target == 0) {
                        random = nlk_random_xs1024(); 
                        target = neg_table[random % NLK_NEG_TABLE_SIZE];
                    }
                    if(target == center_word->index) {
                        /* ignore if this is the actual word */
                        continue;
                    }
                }
                /* forward with lookup for target word */
                nlk_layer_lookup_forward(lk2neg, lk1_out, target, &lk2_out);

                /* outside of sigm bounds round to 1 or 0 respectively */
                if(lk2_out >= NLK_MAX_EXP) {
                    out = 1;
                } else if(lk2_out <= -NLK_MAX_EXP) {
                    out = 0;
                }
                out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);

                /** @section Skipgram NEG Sampling Backprop
                 * Same gradient formula (see above) but this time label can
                 * be 0 (neg) or 1 (target)
                 */
                nlk_bin_nl_sgradient(out, label, &grad_out);

                /* multiply by learning rate */
                grad_out *= learn_rate;

                /* layer2neg backprop, accumulate gradient for all examples */
                nlk_layer_lookup_backprop_acc(lk2neg, lk1_out, target, grad_out, 
                                              grad_acc);

            } /* end of examples */
        }   /* end of NEG sampling specific code */

        /** @section Backprop into the first lookup layer
         * Learn layer1 weights using the accumulated gradient
         */
        nlk_layer_lookup_backprop_lookup(lk1, &word->index, 1, grad_acc);

    } /* end of context words */
}

/** @fn nlk_cbow_for_context
 * Train CBOW for a series of word contexts
 *
 */
void
nlk_cbow_for_context(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
                     bool hs, NLK_LAYER_LOOKUP *lk2neg, size_t negative, 
                     size_t *neg_table, nlk_real learn_rate, 
                     nlk_real *sigmoid_table, nlk_Context *context,
                     size_t *ctx_ids, NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
    nlk_real lk2_out;
    nlk_real out;
    nlk_real grad_out;
    nlk_Vocab *center_word = context->center;
    size_t point;
    uint8_t code;
    size_t target;
    uint64_t random;
    unsigned short int label;
    size_t jj;
    size_t pp;

    if(context->size == 0) {
        return;
    }

    nlk_array_zero(grad_acc);
    for(jj = 0; jj < context->size; jj++) {
        ctx_ids[jj] = context->window[jj]->index;
    }

    /** @section CBOW Forward through the first layer
     * The context words get forwarded through the first lookup layer
     * and their vectors are averaged.
     */
    nlk_array_zero(lk1_out);
    nlk_layer_lookup_forward_avg(lk1, ctx_ids, context->size, lk1_out);


    /** @section 
     * CBOW Hierarchical Softmax Forward
     * The center word gets forwarded through the second lookup 
     * layer. Using hierarchical softmax, the second lookup layer
     * "maps" points to codes (through the output softmax)
     */
    if(hs) {
        /* for each point of center word */ 
        for(pp = 0; pp < center_word->code_length; pp++) {
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
            out =  nlk_sigmoid_lookup(sigmoid_table, lk2_out);

            /*
             * CBOW Hierarchical Softmax Backprop
             * log(sigma(z=v'n(w,j))'vwi) = 
             * = (1 - code) * z - log(1 + e^z)
             * d/dx = 1 - code  - sigmoid(z)
             */
            nlk_bin_nl_sgradient(out, code, &grad_out);

            /* multiply by learning rate */
            grad_out *= learn_rate;
            
            /* layer2 backprop, accumulate gradient for all points */
            nlk_layer_lookup_backprop_acc(lk2hs, lk1_out, point, grad_out, 
                                          grad_acc);
        } /* end of context points/codes */
    } 

    /** @section CBOW NEG Sampling Forward
     */
    if(negative) {
      for(size_t nn = 0; nn < negative + 1; nn++) {
            if(nn == 0) { /* postive example */
                label = 1;
                target = center_word->index;
            } else { /* negative examples */
                label = 0;
                random = nlk_random_xs1024(); 
                target = neg_table[random % NLK_NEG_TABLE_SIZE];
                if(target == 0) {
                    random = nlk_random_xs1024(); 
                    target = neg_table[random % NLK_NEG_TABLE_SIZE];
                }
                if(target == center_word->index) {
                    /* ignore if this is the actual word */
                    continue;
                }
            }
            /* forward with lookup for target word */
            nlk_layer_lookup_forward(lk2neg, lk1_out, target, &lk2_out);

            /* outside of sigm bounds round to 1 or 0 respectively */
            if(lk2_out >= NLK_MAX_EXP) {
                out = 1;
            } else if(lk2_out <= -NLK_MAX_EXP) {
                out = 0;
            }
            out = nlk_sigmoid_lookup(sigmoid_table, lk2_out);

            /** @section Skipgram NEG Sampling Backprop
             * Same gradient formula (see above) but this time label can
             * be 0 (neg) or 1 (target)
             */
            nlk_bin_nl_sgradient(out, label, &grad_out);

            /* multiply by learning rate */
            grad_out *= learn_rate;

            /* layer2neg backprop, accumulate gradient for all examples */
            nlk_layer_lookup_backprop_acc(lk2neg, lk1_out, target, grad_out, 
                                          grad_acc);

            } /* end of examples */
    } /* end of NEG Sampling specific code */

    /** @section Backprop into the first lookup layer
     * Learn layer1 weights using the accumulated gradient
     */
    nlk_layer_lookup_backprop_lookup(lk1, ctx_ids, context->size, 
                                     grad_acc);
}



/**
 * Train a word2vec model
 *
 * @param model_type        NLK_SKIPGRAM, NLK_CBOW
 * @param nn                the neural network
 * @param learn_par         learn paragraph vectors
 * @param train_file_path   the path of the train file
 * @param vocab             the vocabulary
 * @param window
 * @param sample_rate
 * @param learn_rate
 * @param epochs
 * @param verbose
 *
 * @return total error for last epoch
 */
void
nlk_word2vec(NLK_LM model_type,  struct nlk_neuralnet_t *nn, bool hs,
             size_t negative, bool learn_par, char *train_file_path,
             nlk_Vocab **vocab, size_t window, float sample_rate, 
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

    size_t layer_size = lk1->weights->cols;

    /** @section Shared Initializations
     * All variables declared in this section are shared among threads.
     * Most are read-only but some (e.g. learn_rate & word_count_actual)
     * are updated directly from the threads.
     */

    /** @subsection Input and Window initializations
     * Alllocations and initializations related to the input (text)
     */
    size_t word_count_actual = 0;   /* warning: thread write-shared */
    size_t max_word_size = NLK_LM_MAX_WORD_SIZE;
    size_t max_line_size = NLK_LM_MAX_LINE_SIZE;
    size_t ctx_size = window * 2;   /* max context size */

    /* Context for PVDBOW (skipgram with paragraph vectors) includes the
     * window center word in the "context". PVDM ds not.
     * Neither of the non paragraph models includes the center word in the 
     * context obviously.
     * Context for PVDM includes the paragraph vector.
     */
    size_t context_multiplier = 1;
    bool include_self = false;
    bool center_par = false;
    if(learn_par) {
        if(model_type == NLK_SKIPGRAM) { /* PVDBOW */
            context_multiplier *= (ctx_size + 1);
            include_self = true;
            center_par = true;
        } else { /* PVDM */
            ctx_size += 1; /* space for the paragraph (1st elem of window) */
            center_par = false;
        }
    }
    /* determine the train file size */
    size_t train_file_size;
    FILE *in = fopen(train_file_path, "rb");
    if (in == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }
    fseek(in, 0, SEEK_END);
    train_file_size = ftell(in);
    fclose(in);


    size_t train_words = nlk_vocab_total(vocab);


    /** @subsection Neural Net initializations
     * Create and initialize neural net and associated variables 
     */
    nlk_real learn_rate_start = learn_rate;

    /* sigmoid table */
    nlk_real *sigmoid_table = nlk_table_sigmoid_create();

    /* neg table for negative sampling */
    size_t *neg_table = NULL;
    if(negative) {
        neg_table = nlk_vocab_neg_table_create(vocab, NLK_NEG_TABLE_SIZE, 
                                               0.75);
    }

    clock_t start = clock();
    nlk_tic_reset();
    nlk_tic(NULL, false);

    /* random number generator initialization: used for NEG */
    uint64_t seed = 6121984 * clock();
    seed = nlk_random_fmix(seed);
    nlk_random_init_xs1024(seed);


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
    size_t line_count = 0;

    /** @subsection Neural Network thread private initializations
     */
    int local_epoch = epochs;          /* current epoch (thread local) */

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
    /* variables for generating vocab items for the paragraph */
    nlk_Vocab *vocab_par = NULL;
    char *word = (char *) malloc(max_word_size * sizeof(char));
    wchar_t *low_tmp = (wchar_t *) malloc(max_word_size * sizeof(wchar_t));
    char *tmp = NULL;
    if(learn_par) {
        tmp = (char *) malloc((max_word_size * max_line_size + max_line_size) *
                              sizeof(char));
    }

    /* for converting to a vocabularized representation of text */
    nlk_Vocab *vectorized[max_line_size];

    /* for converting a sentence to a series of training contexts */
    nlk_Context **contexts = (nlk_Context **) malloc(max_line_size *
                                                     context_multiplier *
                                                     sizeof(nlk_Context *));
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

    /* for context indexes in cbow */
    size_t *ctx_ids = NULL;
    if(model_type == NLK_CBOW) {
        ctx_ids = (size_t *) malloc(ctx_size * sizeof(size_t));
        if(ctx_ids == NULL) {
            NLK_ERROR_ABORT("unable to allocate memory for context word "
                            " ids", NLK_ENOMEM);
            /* unreachable */
        }
    }

    /* 
     * The train file is divided into parts, one part for each thread.
     * Open file and move to thread specific starting point
     */
    size_t count = 0;
    size_t file_pos_last = 0;
    size_t file_pos = 0;
    size_t end_pos = 0;
    FILE *train = fopen(train_file_path, "rb");

#pragma omp for 
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        end_pos = nlk_set_file_pos(train, learn_par, train_file_size,
                                   num_threads, thread_id);
        file_pos = ftell(train);
        file_pos_last = file_pos;
        /** @section Start of Training Loop (Epoch Loop)
         */
        while(local_epoch > 0) {
             /* update learning rate */
            if (word_count - last_word_count > 10000) {
                word_count_actual += word_count - last_word_count;
                last_word_count = word_count;

                count += file_pos - file_pos_last;
                file_pos_last = file_pos;

                /* display progress */
                if(verbose > 0) {
                    nlk_word2vec_display(learn_rate, word_count_actual, 
                                         train_words, epochs, local_epoch, 
                                         start);
                }
                /* update learning rate */
                learn_rate = nlk_word2vec_learn_rate_update(learn_rate,
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
            line_count++;
            
            /* vocabularize */
            line_len = nlk_vocab_vocabularize(vocab, train_words, text_line, sample_rate, 
                                              NULL, false, vectorized, 
                                              &n_subsampled, vocab_par, tmp, 
                                              word); 

            /* window */
            n_examples = nlk_context_window(vectorized, line_len, include_self, 
                                            window, window, true, 
                                            vocab_par, center_par, contexts);


            /** @subsection Algorithm Parallel Loop Over Contexts
             * CBOW, Skipgram, PVDM, PVDBOW
             */
            for(size_t ex = 0; ex < n_examples; ex++) {
                nlk_Context *context = contexts[ex];

                if(model_type == NLK_SKIPGRAM) {
                    nlk_skipgram_for_context(lk1, lkhs, hs, lkneg,
                                                negative, neg_table, 
                                                learn_rate, sigmoid_table, 
                                                context, grad_acc, lk1_out);
                } else if(model_type == NLK_CBOW) {
                    nlk_cbow_for_context(lk1, lkhs, hs, lkneg, negative,
                                             neg_table, learn_rate, 
                                             sigmoid_table, context, ctx_ids, 
                                             grad_acc, lk1_out);
                } else {
                    NLK_ERROR_ABORT("invalid model type", NLK_EINVAL);
                    /* unreachable */
                }
            }
            word_count += n_examples + n_subsampled;

            /** @subsection Epoch End Check
             * Decrement thread private epoch counter if necessary
             */
            file_pos = ftell(train);
            /* check for end of file part, decrement local epochs, rewind */
            if(file_pos >= end_pos) {
                /* update */
                word_count_actual += word_count - last_word_count;
                local_epoch--;

                /* rewind */
                word_count = 0;
                line_count = 0;
                last_word_count = 0;
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
    if(ctx_ids != NULL) {
        free(ctx_ids);
    }
    if(word != NULL) {
        free(word);
    }
    if(tmp != NULL) {
        free(tmp);
    }
    nlk_array_free(lk1_out);
    nlk_array_free(grad_acc);
    fclose(train);
} /* *** End of Paralell Region *** */

    /** @section End
     */
    nlk_tic_reset();
    free(neg_table);

}


nlk_real
nlk_learn_pv(NLK_LM type, char *train_file, nlk_Vocab **vocab, 
             NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2, nlk_real learn_rate, 
             size_t before, size_t after, nlk_real tol, size_t max_iter,
             char *save_file, nlk_Format save_format)
{

    return 0;
}
