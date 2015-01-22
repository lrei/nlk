#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <locale.h>
#include <getopt.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_text.h"
#include "nlk_window.h"
#include "nlk_neuralnet.h"
#include "nlk_layer_linear.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_tic.h"
#include "nlk_eval.h"

#include "nlk_w2v.h"


static void print_usage()
{
    printf("usage:\n");
}


int main(int argc, char **argv)
{
    nlk_Vocab *vocab; 
    size_t vocab_size = 0;
    size_t vocab_words = 0;
    size_t vocab_total = 0;

    NLK_LM lm_type;

    /** @section Parse Command Line Options Defintions
     */
    int c;
    static int verbose          = 0;
    static int hs               = 0;    /* use hierarchical softmax */
    static int cbow             = 0;    /* use CBOW instead of Skipgram */
    static int binary           = 0;    /* export in binary file */
    static int lower_case       = 0;    /* convert chars to lower case */
    static int learn_par        = 0;    /* learn paragraph vectors */
    char *train_file            = NULL; /* train model on this file */
    char *vocab_load_file       = NULL; /* load vocab from this file */
    char *vocab_save_file       = NULL; /* save vocab to this file */
    char *nn_load_file          = NULL; /* load neuralnet from this file */
    char *nn_save_file          = NULL; /* save neuralnet to this file */
    char *word_vectors_file     = NULL; /* save word vectors to this file */
    char *locale                = NULL; /* set the locale */
    char *questions_file        = NULL; /* word2vec word-analogy tests */
    char *paraphrases_file      = NULL; /* MSR paraphrases style file */
    size_t vector_size          = 100;  /* word vector size */    
    int window                  = 8;    /* window, words before and after */    
    int negative                = 0;    /* number of negative examples */
    int epochs                  = 5;    /* number of iterations/epochs */
    int min_count               = 5;    /* number of iterations/epochs */
    nlk_real learn_rate         = 0;    /* learning rate (start) */
    float sample_rate           = 1e-3; /* random undersample of freq words */
    size_t limit_vocab          = 0;    /* only for question answering */

    while(1) {
        static struct option long_options[] = {
            /* These options set a flag. */
            {"verbose",             no_argument,       &verbose,        1  },
            {"hs",                  no_argument,       &hs,             1  },
            {"cbow",                no_argument,       &cbow,           1  },
            {"binary",              no_argument,       &binary,         1  },
            {"lower-case",          no_argument,       &lower_case,     1  },
            {"learn-sentences",     no_argument,       &learn_par,      1  },
            /* These options don’t set a flag */
            {"train",               required_argument, 0,               't'},
            {"locale",              required_argument, 0,               'a'},
            {"load-vocab",          required_argument, 0,               'r'},
            {"save-vocab",          required_argument, 0,               'c'},
            {"save-net",            required_argument, 0,               's'},
            {"load-net",            required_argument, 0,               'l'},
            {"size",                required_argument, 0,               'v'},
            {"window",              required_argument, 0,               'w'},
            {"negative",            required_argument, 0,               'n'},
            {"alpha",               required_argument, 0,               'a'},
            {"sample",              required_argument, 0,               'u'},
            {"iter",                required_argument, 0,               'i'},
            {"min-count",           required_argument, 0,               'm'},
            {"export",              required_argument, 0,               'x'},
            {"eval-questions",      required_argument, 0,               'q'},
            {"limit-vocab",         required_argument, 0,               'k'},
            {"eval-paraphrases",    required_argument, 0,               'p'},
            {0,                     0,                 0,               0  }
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "t:r:c:v:w:a:u:i:m:x:s:l:a:q:p:k:n:",
                       long_options, &option_index);

        /* Detect the end of the options. */
        if(c == -1) {
            break;
        }

        switch(c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if(long_options[option_index].flag != 0) {
                    break;
                }
                break;
            case 't':
                train_file = optarg;
                break;
            case 'r':
                vocab_load_file = optarg;
                break;
            case 'c':
                vocab_save_file = optarg;
                break;
            case 's':
                nn_save_file = optarg;
                break;
            case 'l':
                nn_load_file = optarg;
                break;
            case 'v':
                vector_size = atoi(optarg);
                break;
            case 'w':
                window = atoi(optarg);
                break;
            case 'n':
                negative = atoi(optarg);
                break;
            case 'a':
                learn_rate = atof(optarg);
                break;
            case 'u':
                sample_rate = atof(optarg);
                break;
            case 'i':
                epochs = atoi(optarg);
                break;
            case 'm':
                min_count = atoi(optarg);
                break;
            case 'x':
                word_vectors_file = optarg;
                break;
            case 'q':
                questions_file = optarg;
                break;
            case 'k':
                limit_vocab = atoi(optarg);
                break;
            case 'p':
                paraphrases_file = optarg;
                break;
            case '?':
                print_usage();
                exit(0);
            default:
                print_usage();
                exit(1);
        }
    }
    /** @subsection Process some options */
#ifndef NCHECKS
    if(verbose) {
        printf("Checks enabled!\n");
    }
#endif
    if(locale != NULL) {
        setlocale(LC_ALL, locale);
    } else {
        setlocale(LC_ALL, "");
    }
   
    /** @section Vocabulary
     */
    bool vocab_changes = false;

    if(vocab_load_file == NULL) {
        if(verbose) {
            nlk_tic(NULL, false);
        }
        /* @TODO fix for learn para */
        vocab = nlk_vocab_create(train_file, NLK_LM_MAX_WORD_SIZE, 0, true,
                                 true);
        vocab_changes = true;
    } else {
        vocab = nlk_vocab_load(vocab_load_file, NLK_LM_MAX_WORD_SIZE);
        nlk_vocab_sort(&vocab);
        nlk_vocab_encode_huffman(&vocab);
    }

    vocab_size = nlk_vocab_size(&vocab);
    vocab_words = nlk_vocab_words_size(&vocab);
    vocab_total = nlk_vocab_total(&vocab);

    if(verbose) {
        printf("\nVocabulary Stats:\nitems: %zu, words: %zu total count: %zu\n", 
                vocab_size, vocab_words, vocab_total);
    }
    
    if(min_count > 0) {
        vocab_changes = true;
        nlk_vocab_reduce(&vocab, min_count);
        vocab_size = nlk_vocab_size(&vocab);
        vocab_words = nlk_vocab_words_size(&vocab);
        vocab_total = nlk_vocab_total(&vocab);
    
        if(verbose) {
            printf("Reduced to:\nitems: %zu, words: %zu total count: %zu\n", 
                    vocab_size, vocab_words, vocab_total);
        }
    }

    /* sort and huffman encode if newly created or reduced */
    if(vocab_changes) {
        nlk_vocab_sort(&vocab);
        nlk_vocab_encode_huffman(&vocab);
        if(verbose) {
            printf("Vocabulary sorted & huffman encoded\n");
        }

        if(nn_load_file != NULL) {
            printf("WARNING: Vocabulary might have changed for previously " 
                   "trained network\n");
        }
    }

    if(vocab_save_file != NULL) {
        nlk_vocab_save(vocab_save_file, &vocab);
        if(verbose) {
            printf("Vocabulary saved to: %s\n", vocab_save_file);
        }
    }

    /** @ section Neural Network
     */
    struct nlk_neuralnet_t *nn;

    if(nn_load_file != NULL) {
        nn = nlk_neuralnet_load_path(nn_load_file);
        if(nn == NULL) {
            printf("Unable to load neural network from %s\n", nn_load_file);
            exit(1);
        }
        if(verbose) {
            printf("Neural Network loaded from %s\n", nn_load_file);
        }
    } else {
        nn = nlk_word2vec_create(vocab_size, vector_size, hs, negative);
        if(verbose) {
            printf("Neural Network created v=%zu x size=%zu\n",
                   vocab_size, vector_size);
        }
    }

    nlk_real learn_err;

    /* set learn_rate if necessary */
    if(cbow && learn_rate == 0) {
        learn_rate = 0.025; 
    } else if(learn_rate == 0) {
        learn_rate = 0.05;
    }

    /* set model type */
    char *model_type;
    if(cbow) {
        lm_type = NLK_CBOW;
        if(learn_par) {
            model_type = "PVDM";
        } else {
            model_type = "CBOW";
        }
    } else {
        lm_type = NLK_SKIPGRAM;
        if(learn_par) {
            model_type = "PVDBOW";
        } else {
            model_type = "SKIPGRAM";
        }
    }

    /* train */
    if(train_file != NULL) {
        //nlk_set_error_handler_off();
        printf("training %s...\n", model_type);
        learn_err = nlk_word2vec(lm_type, nn, hs, negative, learn_par, 
                                 train_file, &vocab, window, sample_rate, 
                                 learn_rate, epochs, verbose);
        if(verbose) {
            printf("\nerror for final epoch = %f\n", learn_err);
        }
    }
    nlk_tic_reset();
    
    /* save */
    if(nn_save_file != NULL) {
        if(nlk_neuralnet_save_path(nn, nn_save_file) != 0) {
            printf("Unble to save neural network to %s\n", nn_save_file);
        }
    }

    /** @TODO exports */
    if(word_vectors_file != NULL) {

    }


    /** @section Evaluation 
     */
    
    NLK_LAYER_LOOKUP *lk1 = nn->layers[0].lk;
    nlk_real accuracy = 0;

    if(questions_file != NULL) {
        nlk_tic("evaluating", true);
        nlk_eval_on_questions(questions_file, &vocab, lk1->weights, 
                              limit_vocab, lower_case, &accuracy);
        printf("accuracy = %f%%\n", accuracy * 100);
    }
    

       
    return 0;
}

/*
     nlk_real tol = 0.01;
    nlk_real ret = 1;
    size_t iter = 0;
 
        nlk_tic("tested", true);

    return 0;
    char *pv_test_file = NULL;

    // pass over test
    freeze = true;
    learn_rate = 0.01;
    learn_par = true;
    // extend the vocabulary with paragraphs
    size_t max_iter = 1000;

    nlk_vocab_extend(&vocab, pv_test_file, NLK_LM_MAX_WORD_SIZE,
                     NLK_LM_MAX_LINE_SIZE, lower_words);

    // increase size of layers
    size_t new_table_size = nlk_vocab_size(&vocab);
    nlk_layer_lookup_resize(lk1, new_table_size);

    NLK_LAYER_LOOKUP *lk2 = nlk_layer_lookup_load_path("vectors.2.nlk.bin");

    nlk_tic_reset();
    nlk_tic("Learning PVs", true);

    while(ret > tol && iter < max_iter) {
        ret = nlk_word2vec(lm_type, lk1, lk2, learn_par, freeze, 
                           pv_test_file, &vocab, window, window, 0, 
                           vector_size, learn_rate, 10, verbose, NULL, NULL, 
                           NLK_FILE_BIN);
        iter++;
    }
    nlk_tic("learned PVs", true);
    printf("ret = %f\n", ret);

    nlk_tic_reset();
    nlk_tic("eval PVs", true);
    nlk_eval_on_paraphrases(pv_test_file, &vocab, lk1->weights,
                            lower_words, &accuracy);
    

    //printf("\nii=%zu\n", ii);
 *
 */
