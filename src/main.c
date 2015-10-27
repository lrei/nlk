#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <locale.h>
#include <getopt.h>

#include <omp.h>

#include "nlk.h"
#include "nlk_err.h"
#include "nlk_math.h"
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_text.h"
#include "nlk_window.h"
#include "nlk_neuralnet.h"
#include "nlk_layer_lookup.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_tic.h"
#include "nlk_eval.h"
#include "nlk_w2v.h"
#include "nlk_pv.h"
#include "nlk_pv_class.h"
#include "nlk_wv_class.h"
#include "nlk_dataset.h"
#include "nlk_util.h"



/* program info */
#define PROGRAM_NAME "nlktool"
#define PROGRAM_FULLNAME "(N)eural (L)anguage (K)it Tool"
#define PROGRAM_VERSION "0.0.1"
#define AUTHOR "Luis Rei\n<me@luisrei.com>\nhttp://luisrei.com"
#define URL "http://github.com/lrei/nlk"
#define LICENSE "MIT"


/* command line options */
enum cmd_opts_t { 
    /* general options */
    CMD_OPTS_THREADS = 1,   /**< specify language model */
    /* unsupervised train/nn options */
    CMD_OPTS_MODEL,         /**< specify language model */
    CMD_OPTS_TRAIN,         /**< train from file */
    CMD_OPTS_EPOCHS,        /**< number of epochs */
    CMD_OPTS_LEARN_RATE,    /**< starting learning rate */
    CMD_OPTS_LEARN_DECAY,   /**< learning rate decay */
    CMD_OPTS_NEGATIVE,      /**< number of negative examples to use */
    CMD_OPTS_VECTOR_SIZE,   /**< word/pv size */
    CMD_OPTS_WINDOW,        /**< context window  size = [window, window]*/
    CMD_OPTS_SAMPLE,        /**< undersample rate for words */
    /* supervised sentence labelling options */
    CMD_OPTS_TRAIN_SENT,    /**< CONLL format file */
    CMD_OPTS_EVAL_SENT,     /**< CONLL format file */
    CMD_OPTS_TEST_SENT,     /**< CONLL format file */
    CMD_OPTS_OUT_SENT,     /**< CONLL format file */
    /* supervised document classification options */
    CMD_OPTS_CLASS,         /**< file specifying an id-class mapping */
    CMD_OPTS_CLASS_TEST,    /**< first id used as test */
    CMD_OPTS_CLASS_FILE,    /**< first id used as test */
    CMD_OPTS_OUT_CLASS,     /**< output classification to this file */
    /* vocabulary */
    CMD_OPTS_SAVE_VOCAB,    /**< save vocab to file */
    CMD_OPTS_LOAD_VOCAB,    /**< load vocab from file */
    CMD_OPTS_IMPORT_VOCAB,  /**< import vocab from file */
    CMD_OPTS_MIN_COUNT,     /**< minimum token frequency */
    CMD_OPTS_REPLACEMENT,   /**< replace low freq tokens with special token */
    /* serialization/output */
    CMD_OPTS_SAVE_NET,      /**< save neural net to file */
    CMD_OPTS_LOAD_NET,      /**< load neural net from file */
    CMD_OPTS_OUT_WORDS,     /**< save/export word vectors */
    CMD_OPTS_IN_WORDS,      /**< import word vectors */
    CMD_OPTS_OUT_PVS,       /**< save/export PVs */
    CMD_OPTS_OUT_FORMAT,    /**< output format */
    CMD_OPTS_PREFIX_PVS,    /**< prefix paragraph ids with string in export */
    /* PV generation (inference/test) */
    CMD_OPTS_GEN_PVS,       /**< generate paragraph vectors for file */
    CMD_OPTS_GEN_SAVE,      /**< save generated PVs according to FORMAT */
    /* evaluation */
    CMD_OPTS_EVAL_QUESTIONS,    /**< eval model on question answering */
    CMD_OPTS_EVAL_PARAPHRASES,  /**< eval model on paraphrase corpus */
    CMD_OPTS_EVAL_PVS,          /**< eval pre-generated PVs as paraphrases */
    CMD_OPTS_EVAL_LIMIT,        /**< limit evaluation to first n elements */
};


static void print_version()
{
    printf("%s - v %s\n"
           "%s\n\n"
           "Copyright (C) 2014-2015\n%s\n"
           "License: %s\n\n",
           PROGRAM_FULLNAME, PROGRAM_VERSION, URL, AUTHOR, LICENSE);
}

static void 
print_usage()
{
    printf ("Usage:\n%s --help\n\n", PROGRAM_NAME);
}
static void print_help()
{
    print_version();
    fputs("\
Neural Network Language Tool.\n\
\n\
General Options:\n\
  --threads [INT]       number of threads to use (default: 0 - all procs)\n\
\n\
Training/Classification:\n\
  --model [STRING]          language model: CBOW, SG, PVDM, PVDBOW\n\
  --concat                  use concatenate variant (valid if --model PVDM)\n\
  --corpus [FILE]           train model with this (text) file\n\
  --line-ids                line's start with ids (paragraph ids)\n\
  --train                   train unsupervised model (--model)\n\
  --iter [INT]              number of train epochs (default: 20)\n\
  --alpha [FLOAT]           the initial learning rate\n\
  --decay [FLOAT]           the learning rate decay\n\
  --hs                      use hierarchical softmax\n\
  --negative [INT]          the number of negative sampling examples\n\
  --size [INT]              the size of word/paragraph vectors\n\
  --window [INT]            the size of the context window\n\
  --sample [FLOAT]          the word undersampling rate\n\
\n\
Supervised Sentence-Word Classification Options:\n\
  --train-sent-word [FILE]      train classifier with CONLL format file\n\
  --test-sent-word [FILE]       evaluate classifier with CONLL format file\n\
  --classify-sent-word [FILE]   classify a file in CONLL format w/o labels\n\
  --output-sent-word [FILE]     output the results (classes) to this file\n\
\n\
Supervised Document Classification Options:\n\
  --classes [FILE]          train classifier with this id-class map file\n\
  --test [FILE]             test classifier with this id-class map file\n\
  --test-file [FILE]        test classifier with this file [NOT]\n\
  --test-classes [FILE]     test classifier with this id-class map file[NI!]\n\
  --classify [FILE]         classify this file\n\
  --output-class [FILE]     output classification results to this file\n\
\n\
Vocabulary:\n\
  --min-count  [INT]    minimum token count\n\
  --load-vocab [FILE]   load the vocabulary to file (includes counts)\n\
  --with-replacement    replace tokens below the minimum with special token \n\
  --save-vocab [FILE]   save the vocabulary to file\n\
  --import-vocab [FILE] import vocabulary from file\n\
\n\
Serialization/Export:\n\
  --save-net [FILE]     save the neural network\n\
  --load-net [FILE]     load an existing neural network\n\
  --remove-pvs          remove paragraph vectors in saved network\n\
  --output [FILE]       output word+paragraph vectors\n\
  --output-words [FILE] output word vectors\n\
  --output-pvs [FILE]   output paragraph vectors\n\
  --par-prefix [STR]    prefix paragraph ids when exporting\n\
  --import-words [FILE] import word vectors from file\n\
\n\
Paragraph Vector Inference:\n\
  --gen-pvs [FILE]      generate paragraph vectors for this file\n\
  --gen-output [FILE]   output generated paragraph vectors\n\
\n\
Evaluation:\n\
  --questions [FILE]    evaluate on question corpus\n\
  --paraphrases [FILE]  evaluate on question corpus\n\
  --eval-pvs [FILE]     evaluate similarity of pre-generated PVs\n\
  --eval-limit [INT]    limit evaluation to the first vectors\n\
\n\
Miscellaneous:\n\
  --format [STR]        format for the output options: w2vtxt, w2vbin, nlk,\n\
                        nlktext (for PVs only)\n\
  --verbose             print status related messages during execution\n\
  --help                print this message and quit\n\
  --version             print program version information and quit\n\
\n\n\
", stdout);

    /**@TODO printf("Example:\n"); */
}

void
nlk_export_words(struct nlk_layer_lookup_t *table, struct nlk_vocab_t **vocab, 
           NLK_FILE_FORMAT format, const char *path, const bool verbose)
{
    if(verbose) {
        printf("Saving word vectors: %s\n", path);
    }
    if(format == NLK_FILE_W2V_BIN || format == NLK_FILE_W2V_TXT) {
        nlk_w2v_export_word_vectors(table->weights, format, 
                                    vocab, path);
    } else {
        nlk_layer_lookup_save_path(table, path);
    }
}

void
nlk_export_pvs(struct nlk_layer_lookup_t *table, NLK_FILE_FORMAT format, 
               const char *path, const bool verbose)
{
    if(verbose) {
        printf("Saving paragraph vectors: %s\n", path);
    }
    if(format == NLK_FILE_W2V_BIN || format == NLK_FILE_W2V_TXT) {
        nlk_w2v_export_paragraph_vectors(table->weights, format, path);
    } else {
        nlk_layer_lookup_save_path(table, path);
    }
}


int 
main(int argc, char **argv)
{
    /** @section Variable Declaration and Initialization
     */

    /** @subsection General Options
     */
    int num_threads             = 0;    /**< number of threads to use */

    /** @subsection Vocabulary Options
     */
    struct nlk_vocab_t *vocab;          /**< the vocabulary structure */
    char *vocab_save_file       = NULL; /**< save vocab to this file */
    int min_count               = 0;    /**< minimum word frequency (count) */
    static int replace          = 0;    /**< replace low freq tokens */
    char *import_vocab_file     = NULL; /**< import vocabulary from file */
    char *load_vocab_file       = NULL; /**< load vocabulary from file */

    /** @subsection Model Options
     */
    NLK_LM lm_type              = NLK_MODEL_NULL; /* the model type */
    char *model_name            = NULL; /**< model type (string)  */
    static int concat           = 0;    /***< concatenate (e.g. PVDM_CONCAT) */

    /** @subsection Training Options
     */
    char *corpus_file           = NULL; /**< train model on this file */
    static int line_ids         = 0;    /**< lines begin with paragraph ids */
    static int hs               = 0;    /**< use hierarchical softmax */
    static int train            = 0;    /**< unsupervised train */
    size_t vector_size          = 100;  /**< word vector size */    
    int window                  = 8;    /**< window, words before and after */    
    int negative                = 0;    /**< number of negative examples */
    int iter                    = 20;   /**< number of iterations/epochs */
    nlk_real learn_rate         = 0;    /**< learning rate (start) */
    nlk_real learn_rate_decay   = 0;    /**< learning rate decay */
    float sample_rate           = 1e-3; /**< random undersample of freq words */

    /** @subsection sentence labelling
     */
    char *train_sent_file       = NULL; /**< train word classifier */
    char *test_sent_file        = NULL; /**< test word classifier */
    char *out_sent_file         = NULL; /**< output file for word classifier */
    char *eval_sent_file        = NULL; /**< eval word classifier */

    /** @subsection document classification
     */
    char *class_train_file      = NULL; /**< train class map file */
    char *class_test_file       = NULL; /**< test class map file */
    char *classify_file         = NULL; /**< a file to classify */
    char *class_out_file        = NULL; /**< classification output file */

    /** @subsection Serialization &  Export
     */
    static int remove_pvs       = 0;    /**< save NN without paragraphs */
    char *nn_load_file          = NULL; /**< load neuralnet from this file */
    char *nn_save_file          = NULL; /**< save neuralnet to this file */
    char *output_words_file     = NULL; /**< export word vectors to file */
    char *import_words_file     = NULL; /**< import word vectors from file */
    char *output_pvs_file       = NULL; /**< save PVs to this file */
    char *format_name           = NULL; /**< format option as a string */
    NLK_FILE_FORMAT format      = NLK_FILE_BIN;
    
    /** @subsection Paragraph Vector Generation (Inferance/Test) Variables
     */
    char *gen_paragraphs_file   = NULL; /**< generate PVs from this file */
    char *pvs_save_file         = NULL; /**< save generated PVs to this file */

    /** @subsection Evaluation
     */
    char *questions_file        = NULL; /**< word2vec word-analogy tests */
    char *paraphrases_file      = NULL; /**< paraphrases file */
    char *pvs_input_file        = NULL; /**< read PVs from this file (eval) */
    size_t eval_limit           = 0;    /**< limit eval to first n elements */

    /** @subsection Miscellaneous
     */
    static int show_help        = 0;    /**< show help */
    static int show_version     = 0;    /**< show version information */
    static int verbose          = 0;    /**< print status during execution */
    int c                       = 0;    /**< used by getop */
    int option_index            = 0;    /**< getopt option index */



    /** @section Define and Parse Command Line Options Defintions
     */
    while(1) {
        static struct option long_options[] = {
            /* These options set a flag. */
            {"concat",          no_argument,       &concat,         1  },
            {"with-replacement",no_argument,       &replace,        1  },
            {"hs",              no_argument,       &hs,             1  },
            {"train",           no_argument,       &train,          1  },
            {"line-ids",        no_argument,       &line_ids,       1  },
            {"remove-pvs",      no_argument,       &remove_pvs,     1  },
            {"help",            no_argument,       &show_help,      1  },
            {"version",         no_argument,       &show_version,   1  },
            {"verbose",         no_argument,       &verbose,        1  },
            /* 
             * These options don’t set a flag 
             */
            /* general options */
            {"threads",         required_argument, 0, CMD_OPTS_THREADS       },
            /* train/nn/context options */
            {"model",           required_argument, 0, CMD_OPTS_MODEL         },
            {"corpus",          required_argument, 0, CMD_OPTS_TRAIN         },
            {"iter",            required_argument, 0, CMD_OPTS_EPOCHS        },
            {"alpha",           required_argument, 0, CMD_OPTS_LEARN_RATE    },
            {"negative",        required_argument, 0, CMD_OPTS_NEGATIVE      },
            {"size",            required_argument, 0, CMD_OPTS_VECTOR_SIZE   },
            {"window",          required_argument, 0, CMD_OPTS_WINDOW        },
            {"sample",          required_argument, 0, CMD_OPTS_SAMPLE        },
            /* supervised sentence labelling */
            {"train-sent-word", required_argument, 0, CMD_OPTS_TRAIN_SENT    },
            {"test-sent-word",  required_argument, 0, CMD_OPTS_TEST_SENT     },
            {"output-sent-word",required_argument, 0, CMD_OPTS_OUT_SENT     },
            {"eval-sent-word",  required_argument, 0, CMD_OPTS_EVAL_SENT     },
            /* supervised document classification */
            {"classes",         required_argument, 0, CMD_OPTS_CLASS         },
            {"test",            required_argument, 0, CMD_OPTS_CLASS_TEST    },
            {"output-classes",  required_argument, 0, CMD_OPTS_OUT_CLASS     },
            {"classify",        required_argument, 0, CMD_OPTS_CLASS_FILE    },
            {"decay",           required_argument, 0, CMD_OPTS_LEARN_DECAY   },
            /* vocabulary */
            {"save-vocab",      required_argument, 0, CMD_OPTS_SAVE_VOCAB    },
            {"load-vocab",      required_argument, 0, CMD_OPTS_LOAD_VOCAB    },
            {"min-count",       required_argument, 0, CMD_OPTS_MIN_COUNT     },
            {"import-vocab",    required_argument, 0, CMD_OPTS_IMPORT_VOCAB  },
            /* serialization */
            {"save-net",        required_argument, 0, CMD_OPTS_SAVE_NET      },
            {"load-net",        required_argument, 0, CMD_OPTS_LOAD_NET      },
            {"output-words",    required_argument, 0, CMD_OPTS_OUT_WORDS     },
            {"import-words",    required_argument, 0, CMD_OPTS_IN_WORDS  },
            {"output-pvs",      required_argument, 0, CMD_OPTS_OUT_PVS       },
            {"format",          required_argument, 0, CMD_OPTS_OUT_FORMAT    },
            {"par-prefix",      required_argument, 0, CMD_OPTS_PREFIX_PVS    },
            /* PV generation (inferance/test) */
            {"gen-pvs",         required_argument, 0, CMD_OPTS_GEN_PVS       },
            {"gen-output",      required_argument, 0, CMD_OPTS_GEN_SAVE      },
            /* intrinsic evaluation */
            {"questions",  required_argument, 0, CMD_OPTS_EVAL_QUESTIONS     },
            {"paraphrases",required_argument, 0, CMD_OPTS_EVAL_PARAPHRASES   },
            {"eval-pvs",   required_argument, 0, CMD_OPTS_EVAL_PVS           },
            {"eval-limit", required_argument, 0, CMD_OPTS_EVAL_LIMIT         },
            /* global */
            {0,                     0,                 0,                   0}
        };
        
        c = getopt_long_only(argc, argv, "", long_options, &option_index);

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
            /* general options */
            case CMD_OPTS_THREADS:
                num_threads = atoi(optarg);
                break;
            /* train/nn options */
            case CMD_OPTS_MODEL:
                model_name = optarg;
                break;
            case CMD_OPTS_TRAIN:
                corpus_file = optarg;
                break;
            case CMD_OPTS_EPOCHS:
                iter = atoi(optarg);
                break;
            case CMD_OPTS_LEARN_RATE:
                learn_rate = atof(optarg);
                break;
            case CMD_OPTS_LEARN_DECAY:
                learn_rate_decay = atof(optarg);
                break;
            case CMD_OPTS_NEGATIVE:
                negative = atoi(optarg);
                break;
            case CMD_OPTS_VECTOR_SIZE:
                vector_size = atoi(optarg);
                break;
            case CMD_OPTS_WINDOW:
                window = atoi(optarg);
                break;
            case CMD_OPTS_SAMPLE:
                sample_rate = atof(optarg);
                break;
            /* supervised document classification */
            case CMD_OPTS_CLASS:
                class_train_file = optarg;
                break;
            case CMD_OPTS_CLASS_TEST:
                class_test_file = optarg;
                break;
            case CMD_OPTS_CLASS_FILE:
                classify_file = optarg;
                break;
            case CMD_OPTS_OUT_CLASS:
                class_out_file = optarg;
                break;
            /* sentence labelling */
            case CMD_OPTS_TRAIN_SENT:
                train_sent_file = optarg;
                break;
            case CMD_OPTS_TEST_SENT:
                test_sent_file = optarg;
                break;
             case CMD_OPTS_EVAL_SENT:
                eval_sent_file = optarg;
                break;
             case CMD_OPTS_OUT_SENT:
                out_sent_file = optarg;
                break;
            /* vocabulary */
            case CMD_OPTS_SAVE_VOCAB:
                vocab_save_file = optarg;
                break;
            case CMD_OPTS_LOAD_VOCAB:
                load_vocab_file = optarg;
                break;
            case CMD_OPTS_MIN_COUNT:
                min_count = atoi(optarg);
                break;
            case CMD_OPTS_IMPORT_VOCAB:
                import_vocab_file = optarg;
                break;
            /* serialization/output */
            case CMD_OPTS_SAVE_NET:
                nn_save_file = optarg;
                break;
            case CMD_OPTS_LOAD_NET:
                nn_load_file = optarg;
                break;
            case CMD_OPTS_OUT_WORDS:
                output_words_file = optarg;
                break;
            case CMD_OPTS_IN_WORDS:
                import_words_file = optarg;
                break;
            case CMD_OPTS_OUT_PVS:
                output_pvs_file = optarg; /* save PVs to this file */
                break;
            case CMD_OPTS_OUT_FORMAT:
                format_name = optarg;
                break;
            /* paragraph vector inference (generate/test) */
            case CMD_OPTS_GEN_PVS:
                gen_paragraphs_file = optarg;
                break;
            case CMD_OPTS_GEN_SAVE:
                pvs_save_file = optarg;
                break;
            /* evaluation */
            case CMD_OPTS_EVAL_QUESTIONS:
                questions_file = optarg;
                break;
            case CMD_OPTS_EVAL_PARAPHRASES:
                paraphrases_file = optarg;
                break;
            case CMD_OPTS_EVAL_PVS:
                pvs_input_file  = optarg;
                break;
            case CMD_OPTS_EVAL_LIMIT:
                eval_limit  = atoi(optarg);
                break;
            /* miscellaneous */
            case '?':
            default:
                printf("unrecognized option \"%s\"\n", argv[optind - 1]);
                print_usage();
                exit(1);
        } /* end of getopt switch */
    } /* end of getopt loop */

    /* check for no options */
    if(optind == 0) {
        printf("Arguments required\n");
        print_usage();
        exit(0);
    }

    /* show help and quit */
    if(show_help) {
        print_help();
        exit(0);
    }

    /* show version and quit */
    if(show_version) {
        print_version();
        exit(0);
    }


    /** @subsection Debug Info
     */
#ifndef NCHECKS
    if(verbose) {
        printf("CHECKS enabled!\n");
    }
#endif
#ifdef CHECK_NANS
    if(verbose) {
        printf("CHECK_NANS enabled\n");
    }
#endif
#ifdef DEBUG
    if(verbose) {
        printf("Running in DEBUG mode!\n");
        printf("MAX WORD SIZE = %d chars\n", NLK_MAX_WORD_SIZE);
        printf("MAX LINE SIZE = %d words\n", NLK_MAX_LINE_SIZE);
    }
#endif


    /** @section Init and general defaults
     */
    /* Global Init */
    nlk_init(); /* initialize random number generator, sigmoid table, locale */
    nlk_set_num_threads(num_threads);
    if(verbose) {
        printf("num threads: %d\n", nlk_get_num_threads());
    }

    /* Model Type */
    if(model_name == NULL) {
        /* do nothing */
    } else {
        lm_type = nlk_lm_model(model_name, concat);
    }

    /* Output File Format */
    format = nlk_format(format_name);

    /* learn rate */
    if(learn_rate <= 0) {
        learn_rate = nlk_lm_learn_rate(lm_type);
    }

    struct nlk_dataset_t *train_set = NULL;
    /* If classifying, load and print datasets early */
    if(class_train_file != NULL) {
        /* load training set */
        if(verbose) {
            printf("Loading dataset from %s\n", class_train_file);
        }
        train_set = nlk_dataset_load_path(class_train_file);
        if(train_set == NULL) {
            NLK_ERROR_ABORT("unable to read class file", NLK_EINVAL);
            /* unreachable */
        }
        if(verbose) {
            printf("Trainset:\n");
            nlk_dataset_print_class_dist(train_set);
        }
    }

    /** @ section Load or Create Neural Network and Corpus
     */
    struct nlk_neuralnet_t *nn = NULL;

    /** @subsection Load the neural network
     */
    if(nn_load_file != NULL) {
        nlk_tic("Loading Neural Network from ", false);
        printf("%s\n", nn_load_file);

        /* load */
        nn = nlk_neuralnet_load_path(nn_load_file, verbose);
        if(verbose) {
            nlk_tic("Neural Network loaded from ", false);
            printf("%s\n", nn_load_file);
        }

    } 
     /** @subsection Create the neural network
     */
    else if(corpus_file != NULL && train) {
        /* create the vocabulary */
        if(verbose) {
            nlk_tic("creating vocabulary for ", false);
            printf("%s min_count = %d\n", corpus_file, min_count);
        }
        vocab = nlk_vocab_create(corpus_file, line_ids, min_count, replace, 
                                 verbose);
        if(verbose) {
            nlk_tic("vocabulary created", true);
        }

        if(hs) {
            nlk_vocab_encode_huffman(&vocab);
            if(verbose) {
                nlk_tic("vocary huffman encoding done", true);
            }
        }

        uint64_t total_lines = nlk_text_count_lines(corpus_file);
        if(verbose) {
            nlk_tic("lines = ", false);
            printf("%"PRIu64"\n", total_lines);
            fflush(stdout);
        }
        /* total vocabularized word count 
         * @TODO this could be a performance problem, i think it exists
         * because of the vocabulary could've been created with a diff
         * file/corpus and loaded - but im not sure. certainly not sure
         * if it needs to be.
         */
        uint64_t total_words = nlk_vocab_count_words(&vocab, corpus_file,
                                                     line_ids,
                                                     total_lines);
        if(verbose) {
            nlk_tic("total words = ", false);
            printf("%"PRIu64"\n", total_words);
            fflush(stdout);
        }
        /* vocab size */
        if(verbose) {
            size_t vocab_size = nlk_vocab_size(&vocab);
            nlk_tic("vocabulary size = ", false);
            double gb = sizeof(nlk_real) * vocab_size * 1e-9 * vector_size;
            printf("%zu (requires: %.2fGB given %zu vector size)\n", 
                    vocab_size, gb, vector_size);
            fflush(stdout);
        }

        if(hs) {
            nlk_vocab_encode_huffman(&vocab);
        }
        

        /* training options */
        struct nlk_nn_train_t train_opts;
        train_opts.model_type = lm_type;
        train_opts.paragraph = nlk_neuralnet_is_paragraph_model(lm_type);
        train_opts.window = window;
        train_opts.sample = sample_rate;
        train_opts.learn_rate = learn_rate;
        train_opts.hs = hs;
        train_opts.negative = negative;
        train_opts.iter = iter;
        train_opts.vector_size = vector_size;
        train_opts.word_count = total_words;
        train_opts.paragraph_count = total_lines;
        train_opts.line_ids = line_ids;

        /* create network */
        nn = nlk_w2v_create(train_opts, concat, vocab, verbose);
    } 

    /**@section Unsupervised Train 
     */
    if(train && nn != NULL && corpus_file != NULL) {
        if(verbose) {
            printf("training %s with\nlearning rate = %f\nsample_rate=%f\n"
                   "window=%d\n", model_name, nn->train_opts.learn_rate, 
                    nn->train_opts.sample, nn->train_opts.window);
        }

        nlk_w2v(nn, corpus_file, verbose);

        if(verbose) { 
            printf("\nTraining finished\n");
        }
    }


    /** @section Supervised Classification
     */

    /**@subsection Train Word Level Classifier
     */
    if(train_sent_file != NULL) {
        struct nlk_supervised_corpus_t *corpus = NULL;
        corpus = nlk_supervised_corpus_load_conll(train_sent_file, NULL);
        if(verbose) {
            size_t max_sentence_size = 0;
            max_sentence_size = nlk_supervised_corpus_max_sentence_size(corpus);
            printf("max sentence size: %zu\n", max_sentence_size); 
        }

        /* import word vectors 
         * @TODO fix this mess
         * */
        FILE *vin = nlk_fopen(import_words_file);
        NLK_ARRAY *wvs;
        if(format == NLK_FILE_BIN) {
            if(verbose) {
                printf("importing word vectors from binary file\n");
            }
            wvs = nlk_array_load(vin);
        } else {
            if(verbose) {
                printf("importing word vectors from text file\n");
            }
            wvs = nlk_array_load_text(vin);
        }
        if(wvs == NULL) {
            NLK_ERROR_ABORT("load failed", NLK_FAILURE);
            /* unreachable */
        }
        struct nlk_layer_lookup_t *lookup_layer;
        lookup_layer = nlk_layer_lookup_create_from_array(wvs);

        if(verbose) {
            printf("loaded %zu word vectors with dim=%zu\n", 
                    lookup_layer->weights->rows, lookup_layer->weights->cols);
        }

        /* import vocab */
        if(import_vocab_file != NULL) {
            vocab = nlk_vocab_import(import_vocab_file, NLK_MAX_WORD_SIZE,
                    false);
        }
        /* load vocab */
        if(load_vocab_file != NULL) {
            vocab = nlk_vocab_import(load_vocab_file, NLK_MAX_WORD_SIZE, 
                                     true);

        }

        /* training options */
        struct nlk_nn_train_t train_opts;
        train_opts.model_type = lm_type;
        train_opts.paragraph = false;
        train_opts.window = window;
        train_opts.sample = 0;
        train_opts.learn_rate = learn_rate;
        train_opts.hs = false;
        train_opts.negative = false;
        train_opts.iter = iter;
        train_opts.vector_size = vector_size;
        train_opts.word_count = 0;
        train_opts.paragraph_count = 0;

        /* create */
        nn = nlk_wv_class_create_senna(train_opts, vocab, lookup_layer, 
                                       corpus->n_classes, verbose);

        /* train */
        nlk_wv_class_senna_train(nn, corpus, verbose);

        /* eval */
        if(eval_sent_file != NULL) {
            if(verbose) {
                printf("evaluating: %s\n", eval_sent_file);
            }
            struct nlk_supervised_corpus_t *test_corpus = NULL;
            test_corpus = nlk_supervised_corpus_load_conll(eval_sent_file,
                                                           corpus->label_map);
 
            nlk_wv_class_senna_test_eval(nn, test_corpus, verbose);
            /* write */
            if(out_sent_file != NULL) {
                if(verbose) {
                    printf("writting to file: %s\n", out_sent_file);
                }
                errno = 0;
                FILE *fout = fopen(out_sent_file, "w");
                if(fout == NULL) {
                    printf("bad file: %s\n", out_sent_file);
                    exit(1);
                }
                nlk_wv_class_senna_test_out(nn, test_corpus, fout);
                fclose(fout);
            }

        }
        if(test_sent_file != NULL) {
            printf("not implemented\n");
        }
    }


    /**@subsection Train Classifier
     */
    if(class_train_file != NULL && nn != NULL) {

        /* do train classifier */
        
        nlk_pv_classifier(nn, train_set, iter,  learn_rate, 
                          learn_rate_decay, verbose);
        
        nlk_dataset_free(train_set);
    }




    /** @section Save & Export Vectors
     */
    if(nn != NULL) {
        /* save paragraph vectors */
        if(output_pvs_file != NULL) {
            nlk_export_pvs(nn->paragraphs, format, output_pvs_file, verbose);

        }

        /* save word vectors */
        if(output_words_file != NULL) {
            nlk_export_words(nn->words, &nn->vocab, format, output_words_file, 
                             verbose);
        }
        
        /* remove or keep paragraph vectors in NN */
        if(remove_pvs) {
            nlk_layer_lookup_free(nn->paragraphs);
            nn->paragraphs = NULL;
            nn->train_opts.paragraph = false;
        }

        /* save neural net */
        if(nn_save_file != NULL) {
            if(verbose) {
                nlk_tic("Saving Neural Network", true);
                printf("%s\n", nn_save_file);
            }
            if(nlk_neuralnet_save_path(nn, nn_save_file) != 0) {
                printf("Unable to save neural network to %s\n", nn_save_file);
            }
        }

        /* export vocabulary */
        if(vocab_save_file != NULL) {
            nlk_vocab_export(vocab_save_file, &nn->vocab);
            if(verbose) {
                printf("Vocabulary saved to: %s\n", vocab_save_file);
            }
        } /* end vocabulary export */
    } /* end of save/export if(nn != null) */


    /** @section Paragraph Vector Inference
     */
    if(gen_paragraphs_file != NULL) {
        if(nn == NULL) {
            nlk_log_message("No neural network created or loaded");
            return NLK_FAILURE;
        }


        /**@subsection Generate Paragraph Vectors
         */
        /* read file */
        struct nlk_corpus_t *corpus_pvs = nlk_corpus_read(gen_paragraphs_file, 
                                                          &nn->vocab, verbose);
        
        /* generate (infer) paragraph vectors */
        NLK_LAYER_LOOKUP *par_table = NULL;
        NLK_ARRAY *pvs = NULL;

        if(corpus_pvs != NULL) {
            if(verbose) { printf("Generating paragraph vectors\n"); }

            /* do generate */
            par_table = nlk_pv_gen(nn, corpus_pvs, iter, verbose);

        }
        if(par_table != NULL) { pvs = par_table->weights; }
        if(verbose) { printf("\n"); }


        /**@subsection Export Generated Paragraph Vectors
         */
        if(pvs_save_file != NULL && pvs != NULL) {
            FILE *fp_pv = NULL;

            if(verbose) {
                printf("Saving generated paragraph vectors to %s\n", 
                        pvs_save_file);
            }
            
            /* use specified format */
            switch(format) {
                case NLK_FILE_W2V_BIN:
                    /* fall through */
                case NLK_FILE_W2V_TXT:
                    nlk_w2v_export_paragraph_vectors(pvs, format,
                                                     pvs_save_file);
                    break;
                default:
                    fp_pv = fopen(pvs_save_file, "wb");
                    if(fp_pv == NULL) {
                        NLK_ERROR_ABORT("unable to open file.", NLK_FAILURE);
                        /* unreachable */
                    }
                    nlk_array_save(pvs, fp_pv);
                    fclose(fp_pv);
                    fp_pv = NULL;
                    break;
            } /* end of format switch */
        } /* end of save */

        if(pvs != NULL) { nlk_array_free(pvs); }
        if(corpus_pvs != NULL) { nlk_corpus_free(corpus_pvs); }
    }


    /** @section Evaluation  (Intrinsic)
     */
    nlk_real accuracy = 0;

    if(questions_file != NULL && nn != NULL) {
        nlk_tic("evaluating word-analogy", true);
        nlk_eval_on_questions(questions_file, &vocab, nn->words->weights, 
                              eval_limit, true, &accuracy);
        printf("accuracy = %f%%\n", accuracy * 100);
    }
    
    accuracy = 0;
    if(pvs_input_file != NULL) {
        NLK_ARRAY *pvs = NULL;
        FILE *fin_pv = NULL;

        errno = 0;
        fin_pv = fopen(pvs_input_file, "rb");
        if(fin_pv == NULL) {
            NLK_ERROR_ABORT(strerror(errno), errno);
        }

        if(format == NLK_FILE_TXT) {
            pvs = nlk_array_load_text(fin_pv);
        } else if(format == NLK_FILE_BIN) {
            pvs = nlk_array_load(fin_pv);
        } else {
            NLK_ERROR_ABORT("invalid format for loading paragraph vectors",
                            NLK_FAILURE);
        }
        fclose(fin_pv);
        fin_pv = NULL;

        nlk_eval_on_paraphrases_pre_gen(pvs, eval_limit, verbose, &accuracy);
        nlk_array_free(pvs);
        printf("accuracy = %f%%\n", accuracy * 100);
    }

    accuracy = 0;
    if(paraphrases_file != NULL && nn != NULL) {
        nlk_tic("evaluating paraphrases", true);

        /* read file */
        struct nlk_corpus_t *corpus_paraphrase = NULL;
        corpus_paraphrase = nlk_corpus_read(paraphrases_file, &nn->vocab, 
                                            verbose);

        nlk_eval_on_paraphrases(nn, corpus_paraphrase, iter, verbose);
        nlk_corpus_free(corpus_paraphrase);
    }

    /**@subsection Test Classifier
     */
    if(class_test_file != NULL && classify_file == NULL) {
        nlk_pv_classify_test(nn, class_test_file, true);
    }

    /**@subsection Classify a file
     */
    if(classify_file != NULL && nn != NULL) {
        size_t *ids = NULL;
        unsigned int *pred = NULL;

        /* create corpus */
        struct nlk_corpus_t *corpus_classify;
        corpus_classify = nlk_corpus_read(classify_file, &nn->vocab, verbose);
        /* gen pvs */
        struct nlk_layer_lookup_t *par_table;
        par_table = nlk_pv_gen(nn, corpus_classify, iter, verbose);

        /* classify */
        pred = nlk_pv_classify(nn, par_table, NULL, 0, verbose);

        /* save classification results */
        if(class_out_file != NULL && pred != NULL) {
            ids = nlk_range(corpus_classify->len);
            nlk_dataset_save_map_path(class_out_file, ids, 
                                      pred, corpus_classify->len);
        }

        if(class_test_file != NULL) {
            struct nlk_dataset_t *tset = NULL;
            tset = nlk_dataset_load_path(class_test_file);
                        float acc = 0;
            acc = nlk_class_score_accuracy(pred, tset->classes, tset->size);
            if(verbose) {
                printf("Test Accuracy: %f (/%zu)\n", acc, tset->size);
                nlk_class_score_cm_print(pred, tset->classes, tset->size); 
            }
            if(tset != NULL) { nlk_dataset_free(tset); }
        }
        if(pred != NULL) { free(pred); }
        if(ids != NULL) { free(ids); }
    }
       
    return 0;
}
