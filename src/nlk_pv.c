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
 *
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
*/



void
nlk_pv_gen(struct nlk_neuralnet_t *nn, struct nlk_corpus_t *corpus)
{
    struct nlk_layer_lookup_t *par_table;
    par_table = nlk_layer_lookup_create(corpus->len, nn->words->weights->cols);


}



unsigned int *
nlk_pv_classify(struct nlk_neuralnet_t *nn, 
                struct nlk_layer_lookup_t *par_table, size_t *ids, size_t n)
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
                           dset->size);
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
 * @warning this function should probably be moved to eval or something
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

    pred = nlk_pv_classify(nn, nn->paragraphs, test_set->ids, test_set->size);
    ac = nlk_class_score_accuracy(pred, test_set->classes, test_set->size);
    if(verbose) {
        printf("\nTEST SCORE = %f\n", ac);
    }
    free(pred);
    nlk_dataset_free(test_set);

    return ac;

}
