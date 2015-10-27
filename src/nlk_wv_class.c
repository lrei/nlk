/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
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


/** @file nlk_wv_class.c
 * Word Vector Classification
 */


#include <errno.h>
#include <stdio.h>

#include <omp.h>

#include "nlk_tic.h"
#include "nlk_neuralnet.h"
#include "nlk_layer_lookup.h"
#include "nlk_layer_linear.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_learn_rate.h"
#include "nlk_dataset.h"
#include "nlk_util.h"
#include "nlk_text.h"
#include "nlk_vocabulary.h"

#include "nlk_wv_class.h"


/**
 * Create a Window Level Classifier Neural Network based on SENNA
 * See NLP (Almost) from Scratch
 *
 * @param wv    a word lookup layer
 */
struct nlk_neuralnet_t *
nlk_wv_class_create_senna(struct nlk_nn_train_t train_opts,
                          struct nlk_vocab_t *vocab,
                          struct nlk_layer_lookup_t *wv, 
                          const size_t n_classes,
                          const bool verbose)
{
    struct nlk_neuralnet_t *nn = NULL;

    struct nlk_context_opts_t ctx_opts;

    size_t hidden_size = train_opts.vector_size;

    /* create structure */
    nn = nlk_neuralnet_create(2);
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
    nn->words = wv;

    /* hardtanh layer (linear layer with bias) 
     * the input is the concatenation of word vector for the window
     */
    size_t layer2_size =  wv->weights->cols * (nn->train_opts.window * 2 + 1);

    if(verbose) {
        printf("Linear Layer (1): %zu x %zu\n", hidden_size, layer2_size); 
    }
    struct nlk_layer_linear_t *l1 = nlk_layer_linear_create(hidden_size,
                                                            layer2_size,
                                                            true);
    nlk_layer_linear_init_senna(l1);
    nlk_neuralnet_add_layer_linear(nn, l1);

    /* softmax layer */
    struct nlk_layer_linear_t *l2 = nlk_layer_linear_create(n_classes,
                                                            hidden_size,
                                                            true);
    if(verbose) {
        printf("Linear Layer (2): %zu x %zu\n", n_classes, hidden_size); 
    }
    nlk_layer_linear_init_senna(l2);
    nlk_neuralnet_add_layer_linear(nn, l2);


    return nn;
}


/**
 * Forward propagation for the SENNA imlementation
 */
inline static void
nlk_wv_class_senna_forward(struct nlk_neuralnet_t *nn,
                           struct nlk_context_t *context,
                           NLK_ARRAY *lk1_out, NLK_ARRAY *ll1_out,
                           NLK_ARRAY *hth_out, NLK_ARRAY *ll2_out,
                           NLK_ARRAY *out)
{
    /* shortcuts */
    struct nlk_layer_linear_t *ll1 = nn->layers[0].ll;
    struct nlk_layer_linear_t *ll2 = nn->layers[1].ll;
 
    /* lookup words */
    nlk_layer_lookup_forward_lookup_concat(nn->words, 
                                           context->window,
                                           context->size,
                                           lk1_out);
    /* hardtanh */
    nlk_layer_linear_forward(ll1, lk1_out, ll1_out);
    nlk_hardtanh_forward(ll1_out, hth_out);

    /* softmax */
    nlk_layer_linear_forward(ll2, hth_out, ll2_out);
    nlk_log_softmax_forward(ll2_out, out);


}


/**
 * Train a SENNA model
 * @param nn        the neural network
 * @param train    the supervised corpus for training
 * #param verbose   print progress
 */
void
nlk_wv_class_senna_train(struct nlk_neuralnet_t *nn, 
                         struct nlk_supervised_corpus_t *train,
                         int verbose)
{
    /**@section Allocations and Shortcuts
     */
    /* vocabulary/context */
    unsigned int n_sentences = train->n_sentences;
    struct nlk_vocab_t **vocab = &nn->vocab;
    struct nlk_context_opts_t context_opts = nn->context_opts;

    struct nlk_vocab_t *replacement = nlk_vocab_find(vocab, NLK_UNK_SYMBOL);
    struct nlk_vocab_t **varray;
    unsigned int max_sentence_size = 0;
    unsigned int len = 0;
    unsigned int n_examples = 0;
    float accuracy = 0;
    
    max_sentence_size = nlk_supervised_corpus_max_sentence_size(train);
    varray = (struct nlk_vocab_t **) malloc(sizeof(struct nlk_vocab_t *) 
                                            * max_sentence_size);
    nlk_assert_silent(varray != NULL);
    
    unsigned int ctx_size = context_opts.max_size;
    struct nlk_context_t **contexts = nlk_context_create_array(ctx_size);
    struct nlk_context_t *context;

    /** neural network 
     */
    const nlk_real learn_rate = nn->train_opts.learn_rate;
    unsigned int epochs = nn->train_opts.iter;
    struct nlk_layer_linear_t *ll1 = nn->layers[0].ll;
    struct nlk_layer_linear_t *ll2 = nn->layers[1].ll;
    const size_t concat_size = ll1->weights->cols;
    const size_t hidden_size = ll1->weights->rows;
    const size_t n_classes = ll2->weights->rows;

    /* output of the first layer (lookup layer) */
    NLK_ARRAY *lk1_out = nlk_array_create(concat_size, 1);
    /* output of the second layer (hardtanh linear layer - hidden) - ll1 */
    NLK_ARRAY *ll1_out = nlk_array_create(hidden_size, 1);
    /* output of the hardtanh transfer */
    NLK_ARRAY *hth_out = nlk_array_create(hidden_size, 1);
    /* output of the 3rd layer (softmax linear layer) - ll2 */
    NLK_ARRAY *ll2_out = nlk_array_create(n_classes, 1);
    /* output of the softmax transfer function */
    NLK_ARRAY *out = nlk_array_create(n_classes, 1);

    /* gradient at output */
    NLK_ARRAY *grad_out = nlk_array_create(n_classes, 1);
    /* gradient at softmax transfer input */
    NLK_ARRAY *grad_sm = nlk_array_create(n_classes, 1);
    /* gradient at the softmax linear layer input */
    NLK_ARRAY *grad_ll2 = nlk_array_create(hidden_size, 1);
    /* gradient at hardtanh input */
    NLK_ARRAY *grad_hth = nlk_array_create(hidden_size, 1);
    /* gradient at the hardtanh linear layer input */
    NLK_ARRAY *grad_ll1 = nlk_array_create(concat_size, 1);

    /* learn rate for ll2 */
    const nlk_real learn_rate_ll2 = learn_rate / hidden_size;
    const nlk_real learn_rate_ll1 = learn_rate / concat_size;


    /** cycle variables 
     */
    size_t *sentence_indices = nlk_range(n_sentences);
    unsigned int epoch = 0;
    unsigned int si = 0;
    unsigned int ci = 0;
    size_t pred = 0;
    size_t correct = 0;

    /**@TODO generate contexts only once */


    /* for each epoch */
    for(epoch = 0; epoch < epochs; epoch++) {
        correct = 0;

        /* randomize sentence order */
        nlk_shuffle_indices(sentence_indices, n_sentences);

        /* for each sentence */
        for(unsigned int ii = 0; ii < n_sentences; ii++) {
            si = sentence_indices[ii];

            /* Vocabularize */
            len = nlk_vocab_vocabularize(vocab, train->words[si], 
                                         replacement, varray); 
            /* Generate Context Window */
            n_examples = nlk_context_window(varray, len, 0, &context_opts, 
                                            contexts);

            nlk_assert_debug(n_examples == train->n_words[si],
                             "wrong number of examples generated");

            for(ci = 0; ci < n_examples; ci++) {
                context = contexts[ci];
                /** @section Forward propagation
                 */
                nlk_wv_class_senna_forward(nn, context, lk1_out, ll1_out,
                                           hth_out, ll2_out, out);
                /* check result */
                pred = nlk_array_max_i(out);
                if(pred == train->classes[si][ci]) {
                    correct++;
                }

                /** @section Backpropagation
                 */
                /* Negative Log Likelihood */
                nlk_nll_backprop(out, train->classes[si][ci], grad_out);

                /** Sofmax layer backprop
                 */
                /* Softmax Transfer */
                nlk_log_softmax_backprop(out, grad_out, grad_sm); 

                /* update gradient at input of the softmax linear layer */
                nlk_layer_linear_update_gradient(ll2, grad_sm, grad_ll2);

                /* apply learning rate */
                nlk_array_scale(learn_rate_ll2, grad_sm);

                /* update parameters of the softmax linear layer */
                nlk_layer_linear_update_parameters(ll2, hth_out, grad_sm);

                /** Hardtanh Layer backprop
                 */
                /* hardtanh transfer */
                nlk_hardtanh_backprop(hth_out, grad_ll2, grad_hth);

                /* to update the gradient at input of this layer is unnecessary
                 * because we are not propagating back into the lookup layer
                nlk_layer_linear_update_gradient(ll1, grad_sm, grad_ll1);
                */

                /* apply learning rate */
                nlk_array_scale(learn_rate_ll1, grad_hth);

                /* update parameters of the softmax linear layer */
                nlk_layer_linear_update_parameters(ll1, lk1_out, grad_hth);

            }
        }

        if(verbose) {
            accuracy = correct / (float) train->size;
            printf("[%d/%d] accuracy = %f (%zu / %zu)\n", 
                    epoch, epochs, accuracy, correct, train->size);
        }

    } /* End of epoch cycle */

    /** @section Free Memory 
     */
    free(varray);
    free(sentence_indices);
    nlk_context_free_array(contexts);
    nlk_array_free(lk1_out);
    nlk_array_free(ll1_out);
    nlk_array_free(hth_out);
    nlk_array_free(ll2_out);
    nlk_array_free(out);
    nlk_array_free(grad_out);
    nlk_array_free(grad_sm);
    nlk_array_free(grad_ll2);
    nlk_array_free(grad_hth);
    nlk_array_free(grad_ll1);

    return;

error:
    NLK_ERROR_ABORT("Unable to allocate memory", NLK_ENOMEM);
    /* UNREACHABLE */
}


unsigned int **
nlk_wv_class_senna_classify(struct nlk_neuralnet_t *nn,
                            struct nlk_supervised_corpus_t *test,
                            int verbose)
{

    /** @section Init
     */
   if(verbose) {
        nlk_tic("Classifying ", false);
        printf("%zu\n", test->size);
    }

    unsigned int **pred = NULL;
    pred = (unsigned int **) malloc(test->n_sentences * sizeof(unsigned int *));
    for(size_t si = 0; si < test->n_sentences; si++) {
        pred[si] = (unsigned int *) calloc(test->n_words[si],
                                           sizeof(unsigned int));

    }
    nlk_assert_silent(pred != NULL);

    /* vocabulary/context */
    unsigned int n_sentences = test->n_sentences;
    struct nlk_vocab_t **vocab = &nn->vocab;
    struct nlk_context_opts_t context_opts = nn->context_opts;

    struct nlk_vocab_t *replacement = nlk_vocab_find(vocab, NLK_UNK_SYMBOL);
    unsigned int max_sentence_size = 0;
    unsigned int len = 0;
    unsigned int n_examples = 0;
    
    max_sentence_size = nlk_supervised_corpus_max_sentence_size(test);

    /* neural network */
    struct nlk_layer_linear_t *ll1 = nn->layers[0].ll;
    struct nlk_layer_linear_t *ll2 = nn->layers[1].ll;
    const size_t concat_size = ll1->weights->cols;
    const size_t hidden_size = ll1->weights->rows;
    const size_t n_classes = ll2->weights->rows;


//#pragma omp parallel
{
    struct nlk_vocab_t **varray;
    varray = (struct nlk_vocab_t **) malloc(sizeof(struct nlk_vocab_t *) 
                                            * max_sentence_size);
    /* @TODO check memory */
    
    unsigned int ctx_size = context_opts.max_size;
    struct nlk_context_t **contexts = nlk_context_create_array(ctx_size);
    struct nlk_context_t *context;

    /* output of the first layer (lookup layer) */
    NLK_ARRAY *lk1_out = nlk_array_create(concat_size, 1);
    /* output of the second layer (hardtanh linear layer - hidden) - ll1 */
    NLK_ARRAY *ll1_out = nlk_array_create(hidden_size, 1);
    /* output of the hardtanh transfer */
    NLK_ARRAY *hth_out = nlk_array_create(hidden_size, 1);
    /* output of the 3rd layer (softmax linear layer) - ll2 */
    NLK_ARRAY *ll2_out = nlk_array_create(n_classes, 1);
    /* output of the softmax transfer function */
    NLK_ARRAY *out = nlk_array_create(n_classes, 1);


    /** @section Parallel Classify
     */
    /* for each sentence */
//#pragma omp for
    for(size_t si = 0; si < n_sentences; si++) {
        /* Vocabularize */
        len = nlk_vocab_vocabularize(vocab,  test->words[si], 
                                     replacement, varray); 
        /* Generate Context Window */
        n_examples = nlk_context_window(varray, len, 0, &context_opts, 
                                        contexts);

        for(unsigned int ci = 0; ci < n_examples; ci++) {
            context = contexts[ci];
            /** @section Forward propagation
             */
            nlk_wv_class_senna_forward(nn, context, lk1_out, ll1_out,
                                       hth_out, ll2_out, out);
            /* for each word */
            pred[si][ci] = nlk_array_max_i(out);
        }
    }

    /** @section Free Memory 
     */
    free(varray);
    nlk_context_free_array(contexts);
    nlk_array_free(lk1_out);
    nlk_array_free(ll1_out);
    nlk_array_free(hth_out);
    nlk_array_free(ll2_out);
    nlk_array_free(out);

} /* end of parallel region */

    return pred;
    
error:
    NLK_ERROR_NULL("unable to allocate memory", NLK_ENOMEM);
    /* unreachable */

}


/**
 * Convenience function - outputs to file: 
 * [word]\t[TRUE_CLASS]\t[PRED_CLASS]\n
 */
void
nlk_wv_class_senna_test_out(struct nlk_neuralnet_t *nn,
                            struct nlk_supervised_corpus_t *test,
                            FILE *output)
{
    unsigned int **pred;
    if(test == NULL) {
        NLK_ERROR_VOID("invalid test set", NLK_FAILURE);
        /* unreachable */
    }

    pred = nlk_wv_class_senna_classify(nn, test, false);

    /* write */
    for(size_t si = 0; si < test->n_sentences; si++) {
        for(size_t wi = 0; wi < test->n_words[si]; wi++) {
            char *pred_label = nlk_vocab_at_index(&test->label_map,
                                                  pred[si][wi])->word; 
            char *true_label = nlk_vocab_at_index(&test->label_map,
                                                  test->classes[si][wi])->word;
            char *word = test->words[si][wi];
            fprintf(output, "%s %s %s\n", word, true_label, pred_label); 
        }
        fprintf(output, "\n");
    }
    fflush(output);


    /*nlk_free_double((void **)pred, test->size);*/

}


/**
 * A convenience function for evaluating results
 */
float
nlk_wv_class_senna_test_eval(struct nlk_neuralnet_t *nn,
                        struct nlk_supervised_corpus_t *test,
                        const int verbose)
{
    float ac = 0;
    float f1 = 0;
    float prec = 0;
    float rec = 0;
    unsigned int **pred;
    if(test == NULL) {
        NLK_ERROR("invalid test set", NLK_FAILURE);
        /* unreachable */
    }

    pred = nlk_wv_class_senna_classify(nn, test, verbose);

    unsigned int *pred_flat;
    pred_flat = (unsigned int *) malloc(sizeof(unsigned int) * test->size);
    nlk_flatten(pred, test->n_sentences, test->n_words, pred_flat);

    unsigned int *classes;
    classes = (unsigned int *) malloc(sizeof(unsigned int) * test->size);
    nlk_flatten(test->classes, test->n_sentences, test->n_words, classes);


    ac = nlk_class_score_accuracy(pred_flat, classes, test->size);

    f1 = nlk_class_f1pr_score_micro(pred_flat, classes,
                                    test->size, test->n_classes,
                                    &prec, &rec);
    if(verbose) {
        printf("\nTEST SCORE:\n"
               "accuracy =\t%.4f\n"
               "precision =\t%.4f\n"
               "recall =\t%.4f\n"
               "f1 = %.4f\n\n", 
               ac, prec, rec, f1);
    }

    free(pred_flat);
    free(classes);
    /*nlk_free_double((void **)pred, test->size);*/

    return ac;

}
