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


/** @file nlk_pv.c
 * Paragraph Vector Specific functions
 * @note: code relative to training a PV model is in nlk_w2v.c
 */


#include <errno.h>

#include <omp.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_tic.h"
#include "nlk_text.h"
#include "nlk_random.h"
#include "nlk_w2v.h"
#include "nlk_learn_rate.h"

#include "nlk_pv.h"


/**
 * Generate a PV using PVDM
 */
static inline void
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

    nlk_array_zero(lk1_out);
    nlk_array_zero(grad_acc);

    /* PVDM Forward through the first layer */


    /* The context words get forwarded through the first lookup layer
     * and their vectors are averaged together with the PV.
     * The window[0] is the paragraph id and should be ignored as the PV
     * is provided independently of the rest of the lookup layer
     */
    nlk_layer_lookup_forward_lookup_avg_p(lk1, &context->window[1], 
                                          context->size - 1, pv, lk1_out);

    /* Hierarchical Softmax */
    if(hs) {
        hierarchical_softmax(lk2hs, false, lk1_out, learn_rate, 
                             sigmoid_table, context->center, grad_acc);
    } 

    /* NEG Sampling */
    if(negative) {
        negative_sampling(lk2neg, false, neg_table, negative, 
                          vocab_size, learn_rate, 
                          context->center->index, lk1_out, 
                          sigmoid_table, grad_acc);
    }

    /* Backprop into the PV: Learn PV weights using the accumulated gradient */
    nlk_array_add_carray(grad_acc, pv->data);
}

/**
 * Generate a PV using PVDBOW
 *
 * In PVDBOW each "context" is just the target word which is associated with 
 * the PV.
 */
static inline void
nlk_pv_dbow_gen(NLK_LAYER_LOOKUP *lk1, NLK_LAYER_LOOKUP *lk2hs,
                const bool hs, NLK_LAYER_LOOKUP *lk2neg, const size_t negative,
                const size_t *neg_table, const size_t vocab_size, 
                const nlk_real learn_rate, const nlk_real *sigmoid_table, 
                const struct nlk_context_t *context, NLK_ARRAY *grad_acc, 
                NLK_ARRAY *lk1_out, NLK_ARRAY *pv)
{
    nlk_array_zero(grad_acc);

    /* Hierarchical Softmax */
    if(hs) {
        hierarchical_softmax(lk2hs, false, pv, learn_rate, 
                             sigmoid_table, context->center, grad_acc);
    }
        
    /* NEG Sampling */
    if(negative) {
        negative_sampling(lk2neg, false, neg_table, negative, 
                           vocab_size, learn_rate, 
                           context->center->index, lk1_out, 
                           sigmoid_table, grad_acc);
    }

    /* Backprop into the PV */
    nlk_array_add_carray(grad_acc, pv->data);
}


/**
 * Learn a Paragraph vector for a single line
 */
void
nlk_pv_gen_one(const NLK_LM model_type, struct nlk_neuralnet_t *nn, 
               const bool hs, const unsigned int negative, nlk_real learn_rate, 
               const unsigned int steps, struct nlk_vocab_t **vocab, 
               const size_t vocab_size, char **paragraph, 
               const size_t *neg_table, const nlk_real *sigmoid_table, 
               struct nlk_context_t **contexts, NLK_CONTEXT_OPTS *ctx_opts, 
               NLK_ARRAY *pv, NLK_ARRAY *grad_acc, NLK_ARRAY *lk1_out)
{
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


    /* vocabularize: no subsampling */
    line_len = nlk_vocab_vocabularize(vocab, 0, paragraph, 0, NULL, false, 
                                      vectorized, &n_subsampled); 

    /* context window */
    n_examples = nlk_context_window(vectorized, line_len, 0, 
                                    ctx_opts, contexts);

    unsigned int step;
    if(model_type == NLK_PVDBOW) {
        for(step = 0; step < steps; step++) {
            for(size_t ex = 0; ex < n_examples; ex++) {
                nlk_pv_dbow_gen(lk1, lkhs, hs, lkneg, negative, 
                                neg_table, vocab_size, learn_rate, 
                                sigmoid_table, contexts[ex], grad_acc, 
                                lk1_out, pv);
            }

            learn_rate = nlk_learn_rate_step_dec_update(learn_rate, step, 
                                                        steps);
        }
    } else if(model_type == NLK_PVDM) {
        for(step = 0; step < steps; step++) {
            for(size_t ex = 0; ex < n_examples; ex++) {
                nlk_pv_dm_gen(lk1, lkhs, hs, lkneg, negative, neg_table, 
                              vocab_size, learn_rate, sigmoid_table, 
                              contexts[ex], grad_acc, lk1_out, pv);

            }
            learn_rate = nlk_learn_rate_step_dec_update(learn_rate, step, 
                                                        steps);
        }
    } else {
        NLK_ERROR_ABORT("invalid model type", NLK_EINVAL);
        /* unreachable */
    }
}

/**
 * Display progress of paragraph vector generation
 *
 * @param generated number of PVs generated so far
 * @param total     total number of PVs to generate
 */
static void
nlk_pv_display(const size_t generated, const size_t total)
{
    char display_str[64];
    float progress = (generated / (float) total) * 100;

    snprintf(display_str, 64,
            "Progress: %.2f%% (%zu/%zu) Threads: %d\t", 
            progress, generated, total, omp_get_num_threads());
    nlk_tic(display_str, false);
}


/**
 * Learn a series of paragraph vectors from a file
 *
 */
NLK_ARRAY *
nlk_pv(NLK_LM model_type, struct nlk_neuralnet_t *nn, const bool hs, 
       const unsigned int negative, const char *par_file_path, 
       const bool lower, struct nlk_vocab_t **vocab, const size_t window, 
       nlk_real learn_rate, const unsigned int steps, int verbose)
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

    bool learn_par = true;
    if(model_type == NLK_PVDBOW) { /* PVDBOW */
        context_multiplier = ctx_size;
        ctx_size = 1;   /* one (word, paragraph) pair at a time */
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

#pragma omp parallel shared(generated) 
{
    int num_threads = omp_get_num_threads();

    /** @subsection NN training initializations
     */
    /* current paragraph vector (empty shell) */
    NLK_ARRAY pv;
    pv.cols = 1;
    pv.rows = layer_size;


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

#pragma omp for 
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        /* set train file part position */
        end_pos = nlk_set_file_pos(par_file, true, par_file_size,
                                   num_threads, thread_id);

        /* determine line number */
        line_start = nlk_text_get_line(par_file);
        line_cur = line_start;

        while(true) {
           /* read line */
            nlk_read_line(par_file, text_line, low_tmp, max_word_size,
                          max_line_size);

#ifdef DEBUGRINT
            nlk_text_print_numbered_line(text_line, line_cur, thread_id);
#endif
            
            /* select paragraph */
            pv.data = &par_vectors->data[line_cur * layer_size];


            /* generate */
            nlk_pv_gen_one(model_type, nn, hs, negative, learn_rate, steps,
                           vocab, vocab_size, text_line, neg_table, 
                           sigmoid_table, contexts, &ctx_opts, &pv, grad_acc, 
                           lk1_out);

            generated++;
            line_cur++;
            file_pos = ftell(par_file);
            nlk_pv_display(generated, total_lines);
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
