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
#include <stdio.h>

#include <omp.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_tic.h"
#include "nlk_text.h"
#include "nlk_corpus.h"
#include "nlk_random.h"
#include "nlk_layer_lookup.h"
#include "nlk_layer_linear.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_w2v.h"
#include "nlk_learn_rate.h"
#include "nlk_util.h"
#include "nlk_class.h"

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



void
nlk_pv_freeze_nn(struct nlk_neuralnet_t *nn)
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


void
nlk_pv_unfreeze_nn(struct nlk_neuralnet_t *nn)
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



struct nlk_layer_lookup_t *
nlk_pv_gen(struct nlk_neuralnet_t *nn, const struct nlk_corpus_t *corpus, 
           const unsigned int epochs, const bool verbose)
{
    if(verbose) {
        nlk_tic("Generating paragraph vectors", false);
        printf(" (%u iterations)\n", epochs);
    }

    /* create new paragraph table */
    struct nlk_layer_lookup_t *par_table;
    par_table = nlk_layer_lookup_create(corpus->len, nn->words->weights->cols);
    nlk_layer_lookup_init(par_table);

    /* unpack options */
    NLK_LM model_type = nn->train_opts.model_type;
    const nlk_real learn_rate_start = nn->train_opts.learn_rate;
    uint64_t train_words = nn->train_opts.word_count;
    const float sample_rate = nn->train_opts.sample; 
    unsigned int ctx_size = nn->context_opts.max_size;
    unsigned int layer_size2 = 0;

    if(nn->train_opts.hs) {
        layer_size2 = nn->hs->weights->cols;
    } else if(nn->train_opts.negative) {
        layer_size2 = nn->neg->weights->cols;
    }

    struct nlk_line_t *lines = corpus->lines;

    /* prevent weights from changing for words and hs/neg */
    nlk_pv_freeze_nn(nn);

    /* progress */
    size_t generated = 0;
    const size_t total = corpus->len;


    /** @section Parallel Generation of PVs
     */
#pragma omp parallel shared(generated)
{
    unsigned int local_epoch;
    struct nlk_line_t *line = NULL;

    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = NULL;
    contexts = nlk_context_create_array(ctx_size);

    /* for undersampling words in a line */
    struct nlk_line_t *line_sample = nlk_line_create(NLK_LM_MAX_LINE_SIZE);

    /* output of the first layer */
    NLK_ARRAY *layer1_out = nlk_array_create(layer_size2, 1);

    /* for storing gradients */
    NLK_ARRAY *grad_acc = nlk_array_create(1, layer_size2);

    /* split */
    int num_threads = omp_get_num_threads();
    size_t line_cur;
    size_t end_line;


#pragma omp for
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {

        line_cur = nlk_text_get_split_start_line(total, num_threads, 
                                                 thread_id);
        end_line = nlk_text_get_split_end_line(total, num_threads, 
                                               thread_id);


        while(line_cur < end_line) {
            /* Generate PV for this line */
            nlk_real learn_rate = learn_rate_start;
            unsigned int n_examples;
            unsigned int ex;
            uint64_t line_words;
            uint64_t word_count_actual = 0;

            line = &lines[line_cur];
            line_words = line->len;

            /** @section Generate Contexts Update Vector Loop
             */
            local_epoch = 0;
            for(local_epoch = 0; local_epoch < epochs; local_epoch++) {
                word_count_actual += line_words;

                 /* subsample  line */
                nlk_vocab_line_subsample(line, train_words, sample_rate, 
                                         line_sample);

                /* single word, nothing to do ... */
                if(line_sample->len < 2) {
                    continue;
                }

                /* generate context  */
                n_examples = nlk_context_window(line_sample->varray, 
                                                line_sample->len, 
                                                line_sample->line_id, 
                                                &nn->context_opts, 
                                                contexts);


                /** @subsection update vector with this context
                 */
                switch(model_type) {
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

            /* go to next line, display */
            line_cur++;
            generated++;

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

    nlk_pv_unfreeze_nn(nn);
    return par_table;
}


unsigned int *
nlk_pv_classify(struct nlk_neuralnet_t *nn, 
                struct nlk_layer_lookup_t *par_table, size_t *ids, size_t n,
                const bool verbose)
{
    /** @section Init
     */
    bool ids_mine = false; /* true if function created the ids array */
    /* if no ids are passed, classify the entire table */
    if(n == 0) {
        n = par_table->weights->rows;
        ids = nlk_range(n);
        ids_mine = true;
    }

    if(verbose) {
        nlk_tic("Classifying ", false);
        printf("%zu\n", n);
    }

    unsigned int *pred = NULL;
    pred = (unsigned int *) malloc(n * sizeof(unsigned int));
    if(pred == NULL) {
        NLK_ERROR_NULL("unable to allocate memory", NLK_ENOMEM);
        /* unreachable */
    }

    /* PV size */
    const size_t pv_size = par_table->weights->cols;

    /* softmax layer */
    struct nlk_layer_linear_t *linear = nn->layers[nn->n_layers - 1].ll;
    unsigned int n_classes = linear->weights->rows;

    /** @section Classify
     */
#pragma omp parallel
{
    /** @subsection Parallel Initialization
     */
    /* paragraph id */
    size_t pid = 0;
    /* paragraph vector */
    NLK_ARRAY *pv = nlk_array_create(pv_size, 1);    
    /* output of the linear layer */
    NLK_ARRAY *linear_out = nlk_array_create(n_classes, 1);
    /* output of the softmax transfer (and thus the network) */
    NLK_ARRAY *out = nlk_array_create(n_classes, 1);

    /** @subsection Parallel Classify
     */
#pragma omp parallel for
    /* for each pv */
    for(size_t tid = 0; tid < n; tid++) {
            pid = ids[tid];
            /* forward step 1: get paragraph vector (lookup) */
            nlk_layer_lookup_forward_lookup_one(par_table, pid, pv);
            /* forward step 2: linear layer */
            nlk_layer_linear_forward(linear, pv, linear_out);
            /* forward step 3: softmax transfer */
            nlk_log_softmax_forward(linear_out, out);

            pred[tid] = nlk_array_max_i(out);
    }

} /* end of parallel region */

    if(ids_mine) {
        free(ids);
        ids = NULL;
    }

    return pred;
}


float
nlk_pv_class_train(struct nlk_neuralnet_t *nn, struct nlk_dataset_t *dset, 
                   const unsigned int iter, nlk_real learn_rate, 
                   const nlk_real learn_rate_decay, const bool verbose)
{
    float accuracy = 0;

    /** @section Shortcuts
     */
    /* paragraph lookup table */
    struct nlk_layer_lookup_t *par_table = nn->paragraphs; 

    /* PV size */
    const size_t pv_size = par_table->weights->cols;

    /* softmax layer */
    struct nlk_layer_linear_t *linear = nn->layers[nn->n_layers - 1].ll;
    const unsigned int n_classes = dset->n_classes;
    /* n_classes should be equal to linear->weights->rows */

    /* dataset */
    const size_t size = dset->size;


    /** @section Train Classes
     */
    size_t pid; /* the parapraph id */
    /* paragraph vector */
    NLK_ARRAY *pv = nlk_array_create(pv_size, 1);    
    /* output of the linear layer */
    NLK_ARRAY *linear_out = nlk_array_create(n_classes, 1);
    /* output of the softmax transfer (and thus the network) */
    NLK_ARRAY *out = nlk_array_create(n_classes, 1);

    /* gradient at output */
    NLK_ARRAY *grad_out = nlk_array_create(n_classes, 1);
    /* gradient at softmax input = gradient at linear output */
    NLK_ARRAY *grad_sm_in = nlk_array_create(n_classes, 1);
    /* gradient at linear layer input */
    NLK_ARRAY *grad_in = nlk_array_create(pv_size, 1);


    /** @subsection Train Cycle
     */
    for(unsigned int local_iter = 1; local_iter <= iter; local_iter++) {
        accuracy = 0;
        size_t correct = 0;
        unsigned int pred = 0;

        /* shuffle the data */
        nlk_dataset_shuffle(dset);

        /* for each pv */
        for(size_t tid = 0; tid < size; tid++) {
            /** @subsection Forward
             */
            /* get paragraph id */
            pid = dset->ids[tid];

            /* forward step 1: get paragraph vector (lookup) */
            nlk_layer_lookup_forward_lookup_one(par_table, pid, pv);

            /* forward step 2: linear layer */
            nlk_layer_linear_forward(linear, pv, linear_out);

            /* forward step 3: softmax transfer */
            nlk_log_softmax_forward(linear_out, out);


            /* check result */
            pred = nlk_array_max_i(out);
            if(pred == dset->classes[tid]) {
                correct++;
            }


            /** @subsection Backpropation 
             */
            /* Backprop step 1: Negative Log Likelihood */
            nlk_nll_backprop(out, dset->classes[tid], grad_out);

            /* learning rate */
            nlk_array_scale(learn_rate, grad_out);

            /* Backprop step 2: softmax transfer */
            nlk_log_softmax_backprop(out, grad_out, grad_sm_in); 

            /* Backprop step3: linear layer */
            nlk_layer_linear_backprop(linear, pv, grad_sm_in, grad_in);

        } /* end of paragraphs */

        accuracy = correct / (float) dset->size;
        if(verbose) {
            printf("[%d/%d] accuracy = %f (%zu / %zu) alpha = %f\n", 
                    local_iter, iter, accuracy, correct, dset->size, 
                    learn_rate);
        }

        /* update learning rate */
        learn_rate = nlk_learn_rate_decay(learn_rate, learn_rate_decay);
    } /* end of iterations */

    /** @subsection Cleanup 
     */
    nlk_array_free(pv);
    nlk_array_free(linear_out);
    nlk_array_free(out);
    nlk_array_free(grad_out);
    nlk_array_free(grad_sm_in);
    nlk_array_free(grad_in);

    return accuracy;
}

/**
 * Creates and Trains PV classifier
 * Creates a LogSoftMax Layer and ads it to the network then trains it.
 *
 * @param iter          the number of supervised iterations
 *
 * @return accuracy on the train set
 */
float
nlk_pv_classifier(struct nlk_neuralnet_t *nn, struct nlk_dataset_t *dset, 
                  const unsigned int iter, nlk_real learn_rate,
                  const nlk_real learn_rate_decay, const bool verbose)
{
    double accuracy = 0;

    /**@section Shortcuts and Initializations 
     */
    
    /**@subsection Create the softmax layer
     * softmax layer = linear layer followed by a softmax transfer
     */
    /* embedding size of the paragraphs */
    const size_t pv_size = nn->paragraphs->weights->cols;

    /* create */
    struct nlk_layer_linear_t *linear = NULL;
    linear = nlk_layer_linear_create(dset->n_classes, pv_size,  true);

    /* init */
    nlk_layer_linear_init_sigmoid(linear); /* not a sigmoid but meh */

    /* add to neural network */
    nlk_neuralnet_expand(nn, 1);
    nlk_neuralnet_add_layer_linear(nn, linear);

   
    /**@section Train
     */
    nlk_pv_class_train(nn, dset, iter, learn_rate, learn_rate_decay, verbose);

    /* test */
    unsigned int *pred = NULL;
    pred = nlk_pv_classify(nn, nn->paragraphs, dset->ids, 
                           dset->size, verbose);
    accuracy = nlk_class_score_accuracy(pred, dset->classes, 
                                        dset->size);
    free(pred);

    if(verbose) {
        printf("\naccuracy classifying train set: %f (%zu)\n", 
                accuracy, dset->size); 
        printf("Finished training\n");
    }

    return accuracy;
}

/**
 * Convenience function
 */
float
nlk_pv_classify_test(struct nlk_neuralnet_t *nn, const char *test_path, 
                     const bool verbose)
{
    float ac = 0;
    unsigned int *pred;
    struct nlk_dataset_t *test_set = NULL;
    test_set = nlk_dataset_load_path(test_path);
    if(test_set == NULL) {
        NLK_ERROR("invalid test set", NLK_FAILURE);
        /* unreachable */
    }

    pred = nlk_pv_classify(nn, nn->paragraphs, test_set->ids, test_set->size,
                           verbose);
    ac = nlk_class_score_accuracy(pred, test_set->classes, test_set->size);
    if(verbose) {
        printf("\nTEST SCORE = %f\n", ac);
    }
    free(pred);
    nlk_dataset_free(test_set);

    return ac;
}
