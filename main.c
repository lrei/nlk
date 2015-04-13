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


#define PROGRAM_NAME "nlktool"
#define PROGRAM_VERSION "1.0.0"
#define AUTHOR "Luis Rei <me@luisrei.com>"
#define LICENSE "MIT"


static void print_version()
{
    printf("(N)eural (L)anguage (K)it Tool- v %s\n"
           "Copyright (C) 2014-2015 %s\n"
           "http://luisrei.com\n"
           "License: %s\n\n",
           PROGRAM_VERSION, AUTHOR, LICENSE);
}

static void print_usage()
{
    print_version();
    printf ("Usage: %s [OPTION]... [FILE]...\n", PROGRAM_NAME);
    fputs("\
Neural Network Natural Language Processing Tool.\n\
\n\
  -a, --append              append to the given FILEs, do not overwrite\n\
  -i, --ignore-interrupts   ignore interrupt signals\n\
", stdout);
}


int main(int argc, char **argv)
{
    struct nlk_vocab_t *vocab; 
    size_t vocab_size = 0;
    size_t vocab_words = 0;
    size_t vocab_total = 0;
    size_t total_lines = 0;
    NLK_LAYER_LOOKUP *lk1 = NULL;

    NLK_LM lm_type;

    /** @section Parse Command Line Options Defintions
     */
    int c;
    static int verbose          = 0;
    static int hs               = 0;    /* use hierarchical softmax */
    static int binary           = 0;    /* export in binary file */
    static int lower_case       = 0;    /* convert chars to lower case */
    static int learn_par        = 0;    /* learn paragraph vectors */
    static int save_sents       = 0;    /* save NN with the sentence vectors */
    char *model_name            = NULL; /* model type */
    char *train_file            = NULL; /* train model on this file */
    char *vocab_load_file       = NULL; /* load vocab from this file */
    char *vocab_save_file       = NULL; /* save vocab to this file */
    char *nn_load_file          = NULL; /* load neuralnet from this file */
    char *nn_save_file          = NULL; /* save neuralnet to this file */
    char *word_vectors_file     = NULL; /* export word vectors to this file */
    char *vectors_output_file   = NULL; /* save word vectors to this file */
    char *locale                = NULL; /* set the locale */
    char *questions_file        = NULL; /* word2vec word-analogy tests */
    char *paraphrases_file      = NULL; /* MSR paraphrases style file */
    char *paragraphs_file       = NULL; /* Line delimited paragraphs file */
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
            {"binary",              no_argument,       &binary,         1  },
            {"lower-case",          no_argument,       &lower_case,     1  },
            {"save-sentences",      no_argument,       &save_sents,     1  },
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
            {"min-count",           required_argument, 0,               'y'},
            {"export-vectors",      required_argument, 0,               'x'},
            {"eval-questions",      required_argument, 0,               'q'},
            {"limit-vocab",         required_argument, 0,               'k'},
            {"eval-paraphrases",    required_argument, 0,               'p'},
            {"model",               required_argument, 0,               'm'},
            {"ouput-vectors",       required_argument, 0,               'o'},
            {"gen-par-vectors",     required_argument, 0,               'g'},
            {0,                     0,                 0,               0  }
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "t:r:c:v:w:a:u:i:m:x:s:l:a:q:p:k:n:y:g:",
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
            case 'm':
                model_name = optarg;
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
            case 'y':
                min_count = atoi(optarg);
                break;
            case 'x':
                word_vectors_file = optarg;
                break;
            case 'o':
                vectors_output_file = optarg;
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
            case 'g':
                paragraphs_file = optarg;
                printf("-g NOT IMPLEMENTED");
                exit(1);
                break;
            case '?':
                print_usage();
                exit(0);
            default:
                print_usage();
                exit(1);
        }
    }

    /* check for no options */
    if(optind == 0) {
        print_usage();
        exit(1);
    }

    /** @subsection Process some options */
#ifndef NCHECKS
    if(verbose) {
        printf("Checks enabled!\n");
        printf("Sizeof nlk_real = %lu bytes\n", sizeof(nlk_real));
    }
#endif
    if(locale != NULL) {
        setlocale(LC_ALL, locale);
    } else {
        setlocale(LC_ALL, "");
    }

    /** @section Model Type 
     * Model Type and associated options (e.g. default learn rate)
     */
    char *model_type;
    if(model_name == NULL || strcasecmp(model_name, "cbow") == 0) { 
        /* default, CBOW */
        lm_type = NLK_CBOW;
        model_type = "CBOW";
        learn_par = false;
        if(learn_rate == 0) {
            learn_rate = 0.025; 
        }
 
    } else if(strcasecmp(model_name, "sg") == 0) {
        lm_type = NLK_SKIPGRAM;
        model_type = "SKIPGRAM";
        learn_par = false;
        if(learn_rate == 0) {
            learn_rate = 0.05; 
        }
    } else if(strcasecmp(model_name, "pvdm") == 0) {
        lm_type = NLK_PVDM;
        model_type = "PVDM";
        learn_par = true;
        if(learn_rate == 0) {
            learn_rate = 0.05; 
        }
    } else if(strcasecmp(model_name, "pvdbow") == 0) {
        lm_type = NLK_PVDBOW;
        model_type = "PVDBOW";
        learn_par = true;
        if(learn_rate == 0) {
            learn_rate = 0.025; 
        }
    } else {
        print_usage();
        NLK_ERROR_ABORT("Invalid model type.", NLK_FAILURE);
    }
   
    /** @section Vocabulary
     */
    bool vocab_changes = false;

    if(vocab_load_file == NULL) {
        if(verbose) {
            nlk_tic(NULL, false);
        }
        vocab = nlk_vocab_create(train_file, NLK_LM_MAX_WORD_SIZE, 
                                 lower_case, verbose);
        nlk_vocab_sort(&vocab);

        vocab_changes = true;
        nlk_tic_reset();

        if(verbose) { /* since tic did not print newlines */
            printf("\n");
        }
    } else {
        vocab = nlk_vocab_load(vocab_load_file, NLK_LM_MAX_WORD_SIZE);
        nlk_vocab_sort(&vocab);
        nlk_vocab_encode_huffman(&vocab);
    }

    vocab_size = nlk_vocab_size(&vocab);
    vocab_words = nlk_vocab_words_size(&vocab);
    vocab_total = nlk_vocab_total(&vocab);

    if(verbose) {
        printf("Vocabulary:\nitems: %zu, words: %zu (total count: %zu)\n", 
                vocab_size, vocab_words, vocab_total);
    }
    
    if(min_count > 0) {
        vocab_changes = true;
        nlk_vocab_reduce(&vocab, min_count);
        vocab_size = nlk_vocab_size(&vocab);
        vocab_words = nlk_vocab_words_size(&vocab);
        vocab_total = nlk_vocab_total(&vocab);
    
        if(verbose) {
            printf("Reduced to:\nitems: %zu, words: %zu (total count: %zu)\n", 
                    vocab_size, vocab_words, vocab_total);
        }
    }

    /* sort and huffman encode if newly created or reduced */
    if(vocab_changes) {
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

    /* paragraphs */
    if(learn_par && train_file != NULL) {
        errno = 0;
        FILE *lc = fopen(train_file, "rb");
        if(lc == NULL) {
            NLK_ERROR_ABORT(strerror(errno), errno);
            /* unreachable */
        }
        total_lines = nlk_text_count_lines(lc);
        fclose(lc);
        if(verbose) {
            printf("Lines: %zu\n", total_lines);
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
        nn = nlk_word2vec_create(vocab_size, total_lines, vector_size, hs,
                                 negative);
        if(verbose) {
            printf("Neural Network created v=%zu (+%zu) x size=%zu\n"
                   "hs = %d, neg = %d\n", vocab_size, total_lines, vector_size, 
                   hs, negative);
        }
    }

    /* train */
    if(train_file != NULL) {
        //nlk_set_error_handler_off();
        printf("training %s...\n", model_type);
        nlk_word2vec(lm_type, nn, hs, negative, 
                     train_file, lower_case, &vocab, total_lines, window, 
                     sample_rate, learn_rate, epochs, verbose);
    }
    nlk_tic_reset();
    if(verbose) { 
        printf("\n");
    }
    
    /* save */
    if(learn_par && !save_sents && train_file != NULL) {
        if(verbose) {
            printf("Removing sentence vectors from NN\n");
        }
        /* remove PVs from lookup */
        vocab_size = nlk_vocab_size(&vocab);

        lk1 = nn->layers[0].lk;
        nlk_layer_lookup_resize(lk1, vocab_size);

    }
    if(nn_save_file != NULL) {
        if(nlk_neuralnet_save_path(nn, nn_save_file) != 0) {
            printf("Unable to save neural network to %s\n", nn_save_file);
        }
    }

    /** @section Vectors
     */

    /* save word vectors */
    if(vectors_output_file != NULL) {
        nlk_layer_lookup_save_path(lk1, vectors_output_file);
    }

    /** @section Evaluation 
     */
    nlk_real accuracy = 0;

    if(questions_file != NULL) {
        nlk_tic_reset();
        nlk_tic(NULL, false);
        nlk_tic("evaluating word-analogy", true);
        nlk_eval_on_questions(questions_file, &vocab, lk1->weights, 
                              limit_vocab, lower_case, &accuracy);
        printf("accuracy = %f%%\n", accuracy * 100);
    }
    
    accuracy = 0;
    if(paraphrases_file != NULL) {
        nlk_tic_reset();
        nlk_tic(NULL, false);
        nlk_tic("evaluating paraphrases", true);
        nlk_eval_on_paraphrases(lm_type, nn, hs, negative, window,
                                paraphrases_file, &vocab, verbose,
                                &accuracy);
        printf("accuracy = %f%%\n", accuracy * 100);
        
    }

       
    return 0;
}
