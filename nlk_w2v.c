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
#include <pthread.h>
#include <omp.h>
#include <cblas.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_window.h"
#include "nlk_layer_linear.h"
#include "nlk_tic.h"
#include "nlk_text.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"

#include "nlk_w2v.h"


/* 
 * global variables for skipgram  & cbow
 */
nlk_Lm model_type;
/* text/context */
nlk_Vocab **vocab;
float sample_rate;      /* sampling rate to subsample frequent words */
size_t after;           /* max words after current word */
size_t before;          /* max words before current word */
size_t ctx_size;        /* max context size */
char *train_file_path;  /* train file path */
size_t train_file_size; /* size of the train file */
size_t max_word_size;
size_t max_line_size;
/* learn rate update, epochs */
size_t train_words;
size_t word_count_actual;
nlk_real learn_rate_start;  /* starting learning rate */
nlk_real learn_rate;        /* current learning rate */
int epochs;                 /* total epochs */
/* layers */
size_t layer_size;
nlk_Layer_Lookup *lk1;      /* first layer */
nlk_Layer_Lookup *lk2;      /* the second layer */
nlk_Table *sigmoid_table;   /* for the softmax */
/* threading, verbose */
int num_threads;
int verbose;
clock_t start;


/** @fn nlk_w2v_learn_rate_update(nlk_real learn_rate, nlk_real start_learn_rate,
 *                                size_t epochs, size_t word_count_actual, 
 *                                size_t train_words)
 * Learn rate update function from word2vec
 */
nlk_real
nlk_w2v_learn_rate_update(nlk_real learn_rate, nlk_real start_learn_rate,
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


void
nlk_word2vec(nlk_Lm _model_type, char *filepath, nlk_Vocab **_vocab, 
             size_t _before, size_t _after, float _sample_rate, 
             size_t _layer_size, nlk_real _learn_rate,  unsigned int _epochs, 
             unsigned int _num_threads, int _verbose, char *save_file_path,
             nlk_Format save_format)
{
    size_t ii;
    word_count_actual = 0;

    model_type = _model_type;

    /*
     * text input related allocations/initializations
     */
    vocab = _vocab;
    train_words = nlk_vocab_total(vocab);
    train_file_path = filepath;
    max_word_size = NLK_LM_MAX_WORD_SIZE;
    max_line_size = NLK_LM_MAX_LINE_SIZE;

    /* context window variables */
    after = _after;              /* max words after current word */
    before = _before;            /* max words before current word */
    ctx_size = after + before;   /* max context size */
    sample_rate = _sample_rate;  /* subsample */

    /* 
     * create / initialize neural net and associated variables 
     */
    size_t vocab_size = nlk_vocab_size(vocab);

    layer_size       = _layer_size;        /* size of the hidden layer */
    learn_rate       = _learn_rate;        /* current learning rate */
    learn_rate_start = _learn_rate;        /* initial learning rate */
    epochs           = _epochs;            /* total epochs */

    nlk_set_seed(6121984);                 /* set PRNG seed */

    /* the first layer: vanilla loopup table - word vectors*/
    lk1 = nlk_layer_lookup_create(vocab_size, layer_size);
    nlk_layer_lookup_init(lk1);

    /* the second layer, hierarchical softmax - hs point vectors */
    lk2 = nlk_layer_lookup_create(vocab_size, layer_size);
  
    /* sigmoid table */
    sigmoid_table = nlk_table_sigmoid_create(NLK_SIGMOID_TABLE_SIZE, 
                                             NLK_MAX_EXP);
    verbose = _verbose;

    /* 
     * train file 
     */
    FILE *in = fopen(train_file_path, "rb");
    if (in == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }
    /* determine file size */
    fseek(in, 0, SEEK_END);
    train_file_size = ftell(in);
    fclose(in);


    /* 
     * do train
     */
    start = clock();
    nlk_tic_reset();
    nlk_tic(NULL, false);

    /* start threads */
    if(_num_threads == 0) {
        num_threads = omp_get_num_procs();
    } else {
        num_threads = _num_threads;
    }
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for(ii = 0; ii < num_threads; ii++) {
        pthread_create(&pt[ii], NULL, nlk_word2vec_thread, (void *)ii);
    }
    /* wait for threads to finish */
    for(ii = 0; ii < num_threads; ii++) {
        pthread_join(pt[ii], NULL);
    }
       
    nlk_tic_reset();

    /* save vectors */
    if(save_file_path != NULL) {
        nlk_layer_lookup_save(save_file_path, save_format, vocab, lk1);
        nlk_layer_lookup_save("vectors.bin", NLK_FILE_W2V_BIN, vocab, lk1);
    }

    /* free neural net memory */
    nlk_layer_lookup_free(lk1);
    nlk_layer_lookup_free(lk2);
    nlk_table_free(sigmoid_table);
    free(pt);
}


void *
nlk_word2vec_thread(void *thread_id)
{
    size_t zz;
    /* word count / progress */
    int local_epoch = epochs;          /* current epoch (thread local) */
    size_t n_examples;
    size_t line_len;
    size_t n_subsampled;
    size_t word_count = 0;
    size_t last_word_count = 0;
    clock_t now;
    double progress;
    double speed;
    char display_str[256];

    size_t max_line_size = NLK_LM_MAX_LINE_SIZE;
    size_t max_word_size = NLK_LM_MAX_WORD_SIZE;

    nlk_Array *lk1_out;
    if(model_type == NLK_CBOW) {
        lk1_out = nlk_array_create(ctx_size, layer_size);
    } else {
        lk1_out = nlk_array_create(1, layer_size);
    }
    nlk_Array *cc_out = nlk_array_create(layer_size, 1);

    nlk_Array *lk2_grad = nlk_array_create(1, layer_size);
    nlk_Array *grad_acc = nlk_array_create(1, layer_size);
    nlk_Array *lk2_temp = nlk_array_create(layer_size, 1);

    nlk_Table *rand_pool = nlk_random_pool_create(NLK_RAND_POOL_SIZE, 1000);

    /* allocate memory for reading from the input file */
    char **text_line = (char **) calloc(max_line_size, sizeof(char *));
    if(text_line == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for text", NLK_ENOMEM);
        /* unreachable */
    }
    for(zz = 0; zz < max_line_size; zz++) {
        text_line[zz] = calloc(max_word_size, sizeof(char));
        if(text_line[zz] == NULL) {
            NLK_ERROR_NULL("unable to allocate memory for text", NLK_ENOMEM);
            /* unreachable */
        }
    }
    /* for converting to a vectorized representation of text */
    nlk_Vocab *vectorized[max_line_size];

    /* for converting a sentence to a series of training contexts */
    nlk_Context **contexts = (nlk_Context **) calloc(max_line_size,
                                                     sizeof(nlk_Context *));
    if(contexts == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for contexts", NLK_ENOMEM);
        /* unreachable */
    }
    for(zz = 0; zz < max_line_size; zz++) {
        contexts[zz] = nlk_context_create(ctx_size);
    }

    /* for context indexes in cbow */
    size_t *ctx_ids = (size_t *) calloc(ctx_size, sizeof(size_t));
    if(ctx_ids == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for context word "
                        " ids", NLK_ENOMEM);
        /* unreachable */
    }
   

    /* 
     * The train file is divided into parts, one part for each thread.
     * Open file and move to thread specific starting point
     */
    FILE *train = fopen(train_file_path, "rb");
    fseek(train, train_file_size / (size_t)num_threads * (size_t)thread_id,
          SEEK_SET);

    while(local_epoch > 0) {
         /* update learning rate */
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;

            /* display progress */
            if(verbose > 0) {
                now = clock();

                progress = word_count_actual / 
                           (double)(epochs * train_words + 1) * 100;
                speed = word_count_actual / ((double)(now - start + 1) / 
                        (double)CLOCKS_PER_SEC * 1000),
                snprintf(display_str, 256,
                        "Alpha: %f  Progress: %.2f%% (%d) "
                        "Words/Thread/sec: %.2fk ", 
                        learn_rate, progress, local_epoch, speed);
                nlk_tic(display_str, false);
            }
            /* update learning rate */
            learn_rate = nlk_w2v_learn_rate_update(learn_rate,
                                                   learn_rate_start,
                                                   epochs,
                                                   word_count_actual, 
                                                   train_words);
        }

        /* read line */
        nlk_read_line(train, text_line, max_word_size, max_line_size);
        
        /* vocabularize */
        line_len = nlk_vocab_vocabularize(vocab, text_line, sample_rate, 
                                          rand_pool, NULL, false,
                                          vectorized, &n_subsampled); 

        /* window */
        n_examples = nlk_context_window(vectorized, line_len, false, before, 
                                        after, true, rand_pool, contexts);

        if(model_type == NLK_SKIPGRAM) {
            nlk_skipgram_for_contexts(contexts, n_examples, grad_acc, lk1_out, 
                                      cc_out, lk2_grad, lk2_temp);
        } else if(model_type == NLK_CBOW) {
            nlk_cbow_for_contexts(contexts, n_examples, ctx_ids, grad_acc, 
                                 lk1_out, cc_out, lk2_grad, lk2_temp);
        } else {
            NLK_ERROR_NULL("invalid model type", NLK_EINVAL);
            /* unreachable */
        }

        word_count += n_examples + n_subsampled;

        /* check for end of file, decrement local epochs, rewind */
        if(feof(train) || (word_count > train_words / num_threads)) {
            word_count_actual += word_count - last_word_count;
            local_epoch--;
            /* rewind */
            word_count = 0;
            last_word_count = 0;
            fseek(train, train_file_size /
                         (size_t)num_threads * (size_t)thread_id, SEEK_SET);
        }
    } /* end of thread */

    /* free memory and close files */
    for(zz = 0; zz < max_line_size; zz++) {
        nlk_context_free(contexts[zz]);
    }
    for(zz = 0; zz < max_line_size; zz++) {
        free(text_line[zz]);
    }
    free(ctx_ids);
    nlk_array_free(lk1_out);
    nlk_array_free(cc_out);
    nlk_array_free(lk2_grad);
    nlk_array_free(grad_acc);
    nlk_array_free(lk2_temp);
    nlk_table_free(rand_pool);
    fclose(train);
}

/** @fn nlk_skipgram_for_contexts(nlk_Context **contexts, size_t n_examples,
 *                                nlk_Array *grad_acc, nlk_Array *lk1_out, 
 *                                nlk_Array *cc_out, nlk_Array *lk2_grad,
 *                                nlk_Array *lk2_temp)
 * Train skipgram for a series of word contexts
 *
 */
void
nlk_skipgram_for_contexts(nlk_Context **contexts, size_t n_examples,
                          nlk_Array *grad_acc, nlk_Array *lk1_out, 
                          nlk_Array *cc_out, nlk_Array *lk2_grad,
                          nlk_Array *lk2_temp)
{
    /* 
     * loop through each context word 
     * each example is one word + the context surrounding it
     */
    for(size_t zz = 0; zz < n_examples; zz++) {
        /* loop variables */
        nlk_real lk2_out;
        nlk_real out;
        nlk_real grad_out;
        nlk_Context *context = contexts[zz];
        nlk_Vocab *center_word = context->word;
        nlk_Vocab *word;
        size_t point;
        uint8_t code;
        size_t ii;
        size_t jj;
        size_t pp;


        /* for each context word jj */
        for(jj = 0; jj < context->size; jj++) {
            word = context->context_words[jj];
            nlk_array_zero(grad_acc);

            /* 
             * Forward
             * Each context word gets forwarded through the first lookup layer 
             * while the center word gets forwarded through the second lookup 
             * layer. If using hierarchical softmax, the second lookup layer
             * "maps" points to codes (through the output softmax)
             */
            nlk_layer_lookup_forward_lookup(lk1, &word->index, 1,
                                            lk1_out);

            /*  [1, layer_size] -> [layer_size, 1] */
            nlk_transfer_concat_forward(lk1_out, cc_out);

            
            /* hierarchical softmax: for each point of center word */
            for(pp = 0; pp < center_word->code_length; pp++) {
                point = center_word->point[pp];
                code = center_word->code[pp];

                /* forward with lookup for point pp */
                nlk_layer_lookup_forward(lk2, cc_out, point, &lk2_out);
                
                /* ignore points with outputs outside of sigm bounds */
                if(lk2_out >= sigmoid_table->max) {
                    continue;
                } else if(lk2_out <= sigmoid_table->min) {
                    continue;
                }
                out =  nlk_sigmoid_table(sigmoid_table, lk2_out);

                /*
                 * Backprop
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
                
                /* layer2 backprop, accumulate gradient for all points */
                nlk_layer_lookup_backprop_acc(lk2, cc_out, point, grad_out, 
                                              lk2_grad, grad_acc, lk2_temp);

            } /* end of points/codes */
            /* learn layer1 weights using the accumulated gradient */
            nlk_layer_lookup_backprop_lookup(lk1, &word->index, 1,
                                             grad_acc);

       } /* end of context words for center_word */
    } /* contexts (contexts for line) */
}

/** @fn nlk_cbow_for_contexts(nlk_Context **contexts, size_t n_examples,
 *                            nlk_Array *grad_acc, nlk_Array *lk1_out, 
 *                            nlk_Array *cc_out, nlk_Array *lk2_grad,
 *                            nlk_Array *lk2_temp)
 * Train CBOW for a series of word contexts
 *
 */
void
nlk_cbow_for_contexts(nlk_Context **contexts, size_t n_examples,
                      size_t *ctx_ids, nlk_Array *grad_acc, nlk_Array *lk1_out, 
                      nlk_Array *avg_out, nlk_Array *lk2_grad,
                      nlk_Array *lk2_temp)
{
    size_t context_indices[NLK_LM_MAX_LINE_SIZE];
    /* 
     * loop through each context word 
     * each example is one word + the context surrounding it
     */
    for(size_t zz = 0; zz < n_examples; zz++) {
        /* loop variables */
        nlk_real lk2_out;
        nlk_real out;
        nlk_real grad_out;
        nlk_Context *context = contexts[zz];
        nlk_Vocab *center_word = context->word;
        nlk_Vocab *word;
        size_t point;
        uint8_t code;
        size_t ii;
        size_t jj;
        size_t pp;

        if(context->size == 0) {
            continue;
        }

        nlk_array_zero(grad_acc);
        for(jj = 0; jj < context->size; jj++) {
            ctx_ids[jj] = context->context_words[jj]->index;
        }

        /* 
         * Forward
         * The context words get forwarded through the first lookup layer
         * and their vectors are averaged.
         * The center word gets forwarded through the second lookup 
         * layer. Using hierarchical softmax, the second lookup layer
         * "maps" points to codes (through the output softmax)
         */
        nlk_layer_lookup_forward_lookup(lk1, ctx_ids, context->size, lk1_out);

        nlk_average(lk1_out, context->size, avg_out);
        
        /* hierarchical softmax: for each point of center word */
        for(pp = 0; pp < center_word->code_length; pp++) {
            point = center_word->point[pp];
            code = center_word->code[pp];

            /* forward with lookup for point pp */
            nlk_layer_lookup_forward(lk2, avg_out, point, &lk2_out);
            
            /* ignore points with outputs outside of sigm bounds */
            if(lk2_out >= sigmoid_table->max) {
                continue;
            } else if(lk2_out <= sigmoid_table->min) {
                continue;
            }
            out =  nlk_sigmoid_table(sigmoid_table, lk2_out);

            /*
             * Backprop
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
            
            /* layer2 backprop, accumulate gradient for all points */
            nlk_layer_lookup_backprop_acc(lk2, avg_out, point, grad_out, 
                                          lk2_grad, grad_acc, lk2_temp);

        } /* end of points/codes */
        /* learn layer1 weights using the accumulated gradient */
        nlk_layer_lookup_backprop_lookup(lk1, ctx_ids, context->size, 
                                         grad_acc);

    } /* contexts (examples) */
}



