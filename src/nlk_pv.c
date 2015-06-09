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
 * Paragraph Vector Inference functions
 * @note: code relative to training a PV model is in nlk_w2v.c
 */


#include <errno.h>
#include <stdio.h>

#include <omp.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_tic.h"
#include "nlk_text.h"
#include "nlk_corpus.h"
#include "nlk_random.h"
#include "nlk_w2v.h"
#include "nlk_learn_rate.h"

#include "nlk_pv.h"



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
 * Sets Paragraph Vector Inference Mode (ie. PV generation)
 * In this mode, gradients are only propagated into the Paragraphs Table and
 * not into the Words Table or the second (HS/NEG) layers.
 *
 * @param nn    the neural network structure
 */
void
nlk_pv_inference_mode(struct nlk_neuralnet_t *nn)
{
    /* Prevent Layer Weights From Changing: Words & HS/Neg */
    nn->words->update = false;
    if(nn->train_opts.hs) {
        nn->hs->update = false;
    }
    if(nn->train_opts.negative) {
        nn->neg->update = false;
    }
}


/**
 * Disables Paragraph Vector Inference Mode (i.e. PV generation)
 * In learn mode, gradients are backpropagate to all layers.
 *
 * @param nn    the neural network structure
 */
void
nlk_pv_learn_mode(struct nlk_neuralnet_t *nn)
{
    /* Allow Layer Weights to Change: Words & HS/Neg */
    nn->words->update = true;
    if(nn->train_opts.hs) {
        nn->hs->update = true;
    }
    if(nn->train_opts.negative) {
        nn->neg->update = true;
    }
}


/**
 * Infer Paragraph Vector for a line
 * This is function is the function that does the inference part for 
 * nlk_pv_gen and nlk_pv_gen_string.
 *
 * @param nn            the neural network structure
 * @param line          the line structure holding the input       
 * @param epochs        the number of epochs or iterations to generate a PV
 * @param paragraphs    the paragraph table that will be updated (output)
 * @param line_sample   will hold sampled line for each epoch
 * @param contexts      will hold the contexts generated for each epoch
 * @param grad_acc      for accumulating grandients
 * @param layer1_out    for holding the output of the first layer
 */
inline static void
nlk_pv_gen_line(struct nlk_neuralnet_t *nn, 
                struct nlk_line_t *line,
                const unsigned int epochs,
                struct nlk_layer_lookup_t *paragraphs,
                struct nlk_line_t *line_sample,
                struct nlk_context_t **contexts, NLK_ARRAY *grad_acc,
                NLK_ARRAY *layer1_out)
{
    NLK_LM model_type = nn->train_opts.model_type;
    nlk_real learn_rate = nn->train_opts.learn_rate;
    const nlk_real learn_rate_start = nn->train_opts.learn_rate;
    uint64_t train_words = nn->train_opts.word_count;
    const float sample_rate = nn->train_opts.sample; 
 

    /* Generate PV for this line */
    unsigned int n_examples;
    unsigned int ex;
    uint64_t line_words;
    uint64_t word_count_actual = 0;

    
    line_words = line->len;

    /** @section Generate Contexts Update Vector Loop
     */
    unsigned int local_epoch = 0;
    for(local_epoch = 0; local_epoch < epochs; local_epoch++) {
        word_count_actual += line_words;

         /* subsample  line */
        nlk_vocab_line_subsample(line, train_words, sample_rate, 
                                 line_sample);

        /* single word, nothing to do ... */
        if(line_sample->len < 2) {
            continue;
        }

        /* generate contexts  */
        n_examples = nlk_context_window(line_sample->varray, 
                                        line_sample->len, 
                                        line_sample->line_id, 
                                        &nn->context_opts, 
                                        contexts);


        /** @subsection Update the Paragraph Vector with these contexts
         */
        switch(model_type) {
            case NLK_PVDBOW:
                for(ex = 0; ex < n_examples; ex++) {
                    nlk_pvdbow(nn, paragraphs, learn_rate, 
                               contexts[ex], grad_acc, layer1_out);
                }
                break;

            case NLK_PVDM:
                for(ex = 0; ex < n_examples; ex++) {
                    nlk_pvdm(nn, paragraphs, learn_rate, contexts[ex], 
                             grad_acc, layer1_out);
                }
                break;

            case NLK_PVDM_CONCAT:
                for(ex = 0; ex < n_examples; ex++) {
                    nlk_pvdm_cc(nn, paragraphs, learn_rate, contexts[ex], 
                                grad_acc, layer1_out);
                }
                break;
            case NLK_MODEL_NULL:
            case NLK_CBOW:
            case NLK_CBOW_SUM:
            case NLK_SKIPGRAM:
            default:
                NLK_ERROR_ABORT("invalid model type", NLK_EINVAL);
                /* unreachable */
        } /* end of model switch */
        learn_rate = nlk_learn_rate_w2v(learn_rate, learn_rate_start,
                                        epochs, word_count_actual, 
                                        line_words);
    } /* end of contexts: pv has been generated */ 
}


/**
 * Paragraph Vector Inference
 * Creates a new Paragraph Table and uses the neural network and the corpus
 * to generate a PV for each line in the corpus.
 *
 * @param nn            the neural network structure
 * @param corpus        the input corpus
 * @param epochs        the number of epochs or iterations to generate a PV
 * @param verbose       siplay progress and print other stuff that is useful
 *
 * @return a paragraph table (lookup layer) with the PVs generated
 */
struct nlk_layer_lookup_t *
nlk_pv_gen(struct nlk_neuralnet_t *nn, const struct nlk_corpus_t *corpus, 
           const unsigned int epochs, const bool verbose)
{
    if(verbose) {
        nlk_tic("Generating paragraph vectors", false);
        printf(" (%u iterations)\n", epochs);
    }

    /* create new paragraph table */
    struct nlk_layer_lookup_t *paragraphs;
    paragraphs = nlk_layer_lookup_create(corpus->len, nn->words->weights->cols);
    nlk_layer_lookup_init(paragraphs);

    /* unpack options */
    unsigned int ctx_size = nn->context_opts.max_size;
    unsigned int layer_size2 = 0;

    if(nn->train_opts.hs) {
        layer_size2 = nn->hs->weights->cols;
    } else if(nn->train_opts.negative) {
        layer_size2 = nn->neg->weights->cols;
    }

    /* lines shortcut */
    struct nlk_line_t *lines = corpus->lines;

    /* prevent weights from changing for words and hs/neg */
    nlk_pv_inference_mode(nn);

    /* progress */
    size_t generated = 0;
    const size_t total = corpus->len;


    /** @section Parallel Generation of PVs
     */
#pragma omp parallel shared(generated)
{
    struct nlk_line_t *line = NULL;

    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = NULL;
    contexts = nlk_context_create_array(ctx_size);

    /* for undersampling words in a line */
    struct nlk_line_t *line_sample = nlk_line_create(NLK_MAX_LINE_SIZE);

    /* output of the first layer */
    NLK_ARRAY *layer1_out = nlk_array_create(layer_size2, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size2);

    /* variables for handling splitting the corpus among threads */
    int num_threads = omp_get_num_threads();
    size_t line_cur;
    size_t end_line;


#pragma omp for
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {

        /* get the thread's part of the corpus */
        line_cur = nlk_text_get_split_start_line(total, num_threads, 
                                                 thread_id);
        end_line = nlk_text_get_split_end_line(total, num_threads, 
                                               thread_id);
    
        /* for each line, generate it's PV */
        while(line_cur < end_line) {
            /* get the next line */
            line = &lines[line_cur];
            
            /* generate the paragraph vector */
            nlk_pv_gen_line(nn, line, epochs, paragraphs, line_sample,
                            contexts, grad_acc, layer1_out);
       
            /* go to next line */
            line_cur++;
            generated++;

            /* display progress */
            if(verbose) {
                nlk_pv_display(generated, total);
            }
        } /* end of PV */
    } /* end of all PVs (thread loop) */

    if(verbose) {
            nlk_pv_display(generated, total);
    }

    /* free thread memory */
    nlk_context_free_array(contexts);
    nlk_line_free(line_sample);
    nlk_array_free(layer1_out);
    nlk_array_free(grad_acc);

} /* end of parallel section */

    if(verbose) {
        printf("\n");
    }

    nlk_pv_learn_mode(nn);

    return paragraphs;
}


/**
 * Infer Paragraph Vector for a string 
 *
 * @param nn    the neural netowrk
 * @param str   the string
 * @param epochs    epochs (iterations) to use in generating the PVs
 *
 * @return a paragraph table (lookup layer) with the PV generated (id = 0)
 * 
 * @warning terribly inneficient function that allocates and frees all memory
 *          each time it is called.
 */
struct nlk_layer_lookup_t *
nlk_pv_gen_string(struct nlk_neuralnet_t *nn, char *str,
                  const unsigned int epochs)
{
    /** @section Initialization
     */
    struct nlk_layer_lookup_t *paragraphs = NULL;

    paragraphs = nlk_layer_lookup_create(1, nn->words->weights->cols);

    nlk_layer_lookup_init(paragraphs);

    /* unpack options */
    unsigned int ctx_size = nn->context_opts.max_size;
    unsigned int layer_size2 = 0;

    if(nn->train_opts.hs) {
        layer_size2 = nn->hs->weights->cols;
    } else if(nn->train_opts.negative) {
        layer_size2 = nn->neg->weights->cols;
    }

    /* representation of the line as an array of pointer to strings */
    char **tline = nlk_text_line_create();

    struct nlk_line_t *line = nlk_line_create(NLK_MAX_LINE_SIZE);
    line->line_id = 0;

    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = NULL;
    contexts = nlk_context_create_array(ctx_size);

    /* for undersampling words in a line */
    struct nlk_line_t *line_sample = nlk_line_create(NLK_MAX_LINE_SIZE);

    /* output of the first layer */
    NLK_ARRAY *layer1_out = nlk_array_create(layer_size2, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size2);


    /** @section Vocabularize & Generate Paragraph Vector
     */
    const size_t len = strlen(str);

    /* 1 - convert to **text_line */
    nlk_text_line_read(str, len, tline);

    /* 2 - vocabularity */
    line->len = nlk_vocab_vocabularize(&nn->vocab,  tline, NULL, line->varray); 
 

    /* 3 - generate the paragraph vector */
    nlk_pv_gen_line(nn, line, epochs, paragraphs, line_sample,
                    contexts, grad_acc, layer1_out);
 

    /**@section Free memory 
     */
    nlk_text_line_free(tline);
    nlk_line_free(line);
    nlk_context_free_array(contexts);
    nlk_line_free(line_sample);
    nlk_array_free(layer1_out);
    nlk_array_free(grad_acc);

    return paragraphs;
}
